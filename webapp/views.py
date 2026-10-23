import copy
import hashlib
import io
import secrets
from PIL import Image
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_GET, require_POST, require_http_methods
from src.prescription.documents import decode_document, MAX_BYTES
from src.prescription.layout import checked_box
from src.prescription.page_review import assemble_medications
from src.prescription.knowledge import MedicineKnowledge
from src.prescription.annotations import load_annotations, MAX_ANNOTATION_BYTES
from .forms import UploadForm
from .jobs import store, MAX_REGIONS


def owner(request):
    if "owner" not in request.session:
        request.session["owner"] = secrets.token_urlsafe(32)
    return request.session["owner"]


def get_job(request, identifier):
    try:
        return store.get(identifier, owner(request))
    except KeyError:
        raise Http404("Prescription expired or is not in this browser session.")


def editable(job):
    if job.cancelled or job.state in {"queued", "running"}:
        raise ValueError("Wait for reading to finish before changing regions.")


def page(request, job, error=None, status_code=200):
    with job.lock:
        regions = copy.deepcopy(job.regions)
        medicines = assemble_medications(regions)
        context = {
            "job": job,
            "regions": regions,
            "medicines": medicines,
            "medicine_regions": [r for r in regions if r["role"] == "medicine"],
            "roles": ["unreviewed", "medicine", "instruction", "other"],
            "busy": job.state in {"queued", "running"},
            "error": error,
            "unlinked": sum(
                r["role"] == "instruction" and not r.get("medicine_id") for r in regions
            ),
        }
    return render(request, "review.html", context, status=status_code)


@never_cache
@require_http_methods(["GET", "POST"])
def home(request):
    owner(request)
    form = UploadForm(request.POST or None, request.FILES or None)
    if getattr(request, "upload_error", None):
        form.add_error(None, request.upload_error)
    if request.method == "POST" and form.is_valid():
        error_field = "document"
        try:
            image, png = decode_document(
                form.cleaned_data["document"].read(MAX_BYTES + 1)
            )
            annotations = form.cleaned_data.get("annotations")
            regions = None
            if annotations:
                error_field = "annotations"
                regions = load_annotations(
                    annotations.read(MAX_ANNOTATION_BYTES + 1), png, image.size
                )
            job = store.create(
                owner(request),
                png,
                image.size,
                form.cleaned_data["automatic"],
                regions=regions,
            )
            return redirect("detail", identifier=job.id)
        except ValueError as exc:
            form.add_error(error_field, str(exc))
    return render(request, "upload.html", {"form": form})


@never_cache
@require_GET
def detail(request, identifier):
    return page(request, get_job(request, identifier))


@never_cache
@require_GET
def status(request, identifier):
    job = get_job(request, identifier)
    with job.lock:
        return JsonResponse(
            {
                "state": job.state,
                "stage": job.stage,
                "completed": job.completed,
                "total": job.total,
            }
        )


@never_cache
@require_GET
def picture(request, identifier):
    job = get_job(request, identifier)
    with job.lock:
        return HttpResponse(job.image, content_type="image/png")


@never_cache
@require_GET
def crop(request, identifier, region_id):
    job = get_job(request, identifier)
    with job.lock:
        region = next((r for r in job.regions if r["id"] == region_id), None)
        if region is None:
            raise Http404()
        image = Image.open(io.BytesIO(job.image)).convert("RGB").crop(region["box"])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return HttpResponse(buffer.getvalue(), content_type="image/png")


@never_cache
@require_POST
def review(request, identifier):
    job = get_job(request, identifier)
    try:
        with job.lock:
            editable(job)
            regions = copy.deepcopy(job.regions)
            for r in regions:
                text = request.POST.get("text_" + r["id"], "").strip()
                role = request.POST.get("role_" + r["id"], "")
                parent = request.POST.get("parent_" + r["id"], "").strip()
                if len(text) > 1000 or role not in {
                    "unreviewed",
                    "medicine",
                    "instruction",
                    "other",
                }:
                    raise ValueError(
                        "Check region roles and keep each reading below 1,000 characters."
                    )
                r.update(
                    text=text,
                    role=role,
                    medicine_id=parent if role == "instruction" else "",
                    reviewed=request.POST.get("checked_" + r["id"]) == "on"
                    and role != "unreviewed"
                    and bool(text),
                )
            assemble_medications(regions)
            job.regions = regions
            job.stage = "Review saved"
        return redirect("detail", identifier=identifier)
    except ValueError as exc:
        return page(request, job, str(exc), 400)


@never_cache
@require_POST
def add_region(request, identifier):
    job = get_job(request, identifier)
    try:
        with job.lock:
            editable(job)
            if len(job.regions) >= MAX_REGIONS:
                raise ValueError("Maximum 80 regions per page.")
            box = checked_box(
                [float(request.POST.get(k, "")) for k in ("x1", "y1", "x2", "y2")],
                job.size,
                True,
            )
            job.regions.append(
                {
                    "id": str(
                        max(
                            [int(r["id"]) for r in job.regions if r["id"].isdigit()]
                            + [0]
                        )
                        + 1
                    ),
                    "box": box,
                    "ocr_text": "",
                    "text": "",
                    "role": "unreviewed",
                    "medicine_id": "",
                    "suggested_medicine_id": "",
                    "reviewed": False,
                    "alternatives": [],
                }
            )
            job.stage = "Ready for review"
        return redirect("detail", identifier=identifier)
    except (ValueError, TypeError):
        return page(request, job, "Select a nonempty rectangle inside the image.", 400)


@never_cache
@require_POST
def read_region(request, identifier, region_id):
    job = get_job(request, identifier)
    try:
        with job.lock:
            editable(job)
            target = next((r for r in job.regions if r["id"] == region_id), None)
            if target is None:
                raise Http404()
            # Changing OCR invalidates review of this region and its dependants.
            for r in job.regions:
                if r["id"] == region_id or r.get("medicine_id") == region_id:
                    r["reviewed"] = False
                if r.get("medicine_id") == region_id:
                    r["medicine_id"] = ""
            store.submit(job, region_id)
        return redirect("detail", identifier=identifier)
    except ValueError as exc:
        return page(request, job, str(exc), 400)


@never_cache
@require_POST
def read_empty(request, identifier):
    job = get_job(request, identifier)
    try:
        with job.lock:
            editable(job)
            store.submit(job, empty_only=True)
        return redirect("detail", identifier=identifier)
    except ValueError as exc:
        return page(request, job, str(exc), 400)


@never_cache
@require_POST
def remove_region(request, identifier, region_id):
    job = get_job(request, identifier)
    try:
        with job.lock:
            editable(job)
            job.regions = [r for r in job.regions if r["id"] != region_id]
            for r in job.regions:
                if r.get("medicine_id") == region_id:
                    r.update(medicine_id="", reviewed=False)
        return redirect("detail", identifier=identifier)
    except ValueError as exc:
        return page(request, job, str(exc), 400)


@never_cache
@require_POST
def delete(request, identifier):
    store.delete(get_job(request, identifier))
    return redirect("home")


@never_cache
@require_GET
def export(request, identifier):
    job = get_job(request, identifier)
    with job.lock:
        records = assemble_medications(job.regions)
        payload = {
            "schema_version": 1,
            "image_sha256": hashlib.sha256(job.image).hexdigest(),
            "image_size": job.size,
            "status": (
                "reviewed_transcription"
                if records
                and all(r["transcription_reviewed"] for r in records)
                and all(
                    r["reviewed"]
                    and r["role"] != "unreviewed"
                    and (r["role"] != "instruction" or r.get("medicine_id"))
                    for r in job.regions
                )
                else "draft"
            ),
            "regions": copy.deepcopy(job.regions),
            "medications": records,
            "note": "Contains transcription annotations, not verified medicine identities or clinician-approved instructions. Check and de-identify before sharing or using for training.",
        }
    response = JsonResponse(payload, json_dumps_params={"indent": 2})
    response["Content-Disposition"] = 'attachment; filename="prescription-review.json"'
    return response


@never_cache
@require_POST
def lookup(request, identifier, region_id):
    job = get_job(request, identifier)
    with job.lock:
        medicine = next(
            (r for r in assemble_medications(job.regions) if r["id"] == region_id), None
        )
    if not medicine:
        raise Http404()
    name = request.POST.get("name", "").strip()
    label = request.POST.get("setid", "")
    context = {"job": job, "medicine": medicine, "name": name}
    if not medicine["transcription_reviewed"] or request.POST.get("confirmed") != "on":
        return page(
            request,
            job,
            "Review the linked regions and confirm the medicine spelling first.",
            400,
        )
    try:
        if label:
            if request.POST.get("product_confirmed") != "on":
                raise ValueError(
                    "Confirm that the product ingredients, strength and formulation match."
                )
            matching = MedicineKnowledge().lookup(name)
            if not any(
                c.get("term_type") not in {"IN", "PIN", "MIN"}
                and any(l["setid"] == label for l in c.get("labels", []))
                for c in matching["candidates"]
            ):
                raise ValueError(
                    "Specify the exact medicine product, strength and formulation before selecting a label."
                )
            context["uses"] = MedicineKnowledge().indications(name, label)
        else:
            context["matches"] = MedicineKnowledge().lookup(name)
            for candidate in context["matches"]["candidates"]:
                if candidate.get("term_type") in {"IN", "PIN", "MIN"}:
                    candidate["ingredient_only"] = True
                    candidate["labels"] = []
    except ValueError as exc:
        context["error"] = str(exc)
    except Exception:
        context["error"] = (
            "Medicine information is unavailable right now. Your transcription is still available."
        )
    return render(request, "lookup.html", context)
