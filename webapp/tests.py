"""Run: python -m unittest webapp.tests -v (no model download or database)."""

import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "webapp.settings")
import django

django.setup()
from django.test import Client, override_settings, SimpleTestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image
from src.prescription.documents import decode_document
from src.prescription.page_review import propose_regions, assemble_medications
from src.prescription.extraction import extract_line
from scripts.prepare_page_crops import prepare, verify
from webapp.jobs import JobStore


def png():
    buffer = io.BytesIO()
    Image.new("RGB", (200, 200), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def regions():
    return propose_regions(
        [
            {"id": "1", "box": [0, 0, 90, 30], "ocr_text": "Dr Example"},
            {"id": "2", "box": [0, 40, 90, 70], "ocr_text": "Tab Paracetamol 500 mg"},
            {"id": "3", "box": [100, 40, 200, 70], "ocr_text": "twice daily"},
            {"id": "4", "box": [100, 100, 200, 130], "ocr_text": "once daily"},
        ]
    )


class RegionTests(unittest.TestCase):
    def test_same_row_suggestion_is_not_an_applied_link(self):
        rows = regions()
        self.assertEqual(rows[0]["role"], "other")
        self.assertEqual(rows[2]["suggested_medicine_id"], "2")
        self.assertEqual(rows[3]["suggested_medicine_id"], "")
        self.assertEqual(assemble_medications(rows)[0]["frequency_as_written"], [])

    def test_explicit_links_require_all_evidence_reviewed(self):
        rows = regions()
        rows[2]["medicine_id"] = "2"
        rows[1]["reviewed"] = True
        self.assertIsNone(assemble_medications(rows)[0]["times_per_day_as_written"])
        rows[2]["reviewed"] = True
        result = assemble_medications(rows)[0]
        self.assertEqual(result["times_per_day_as_written"], 2)
        self.assertEqual(result["source_regions"], ["2", "3"])
        self.assertFalse(result["identity_confirmed"])

    def test_cross_row_conflict_does_not_produce_frequency(self):
        rows = regions()
        for row in rows:
            row["reviewed"] = True
        rows[2]["medicine_id"] = rows[3]["medicine_id"] = "2"
        self.assertIsNone(assemble_medications(rows)[0]["times_per_day_as_written"])

    def test_dangling_link_is_rejected(self):
        rows = regions()
        rows[2]["medicine_id"] = "missing"
        with self.assertRaises(ValueError):
            assemble_medications(rows)

    def test_conditional_schedules_and_quantity_are_not_daily_counts(self):
        for text in [
            "Take 2 tablets twice daily as needed",
            "Tab X once daily then twice daily",
            "Tab X twice daily if pain",
        ]:
            self.assertIsNone(extract_line(text)["times_per_day_as_written"])
        parsed = extract_line("Take 2 tablets twice daily")
        self.assertEqual(parsed["dose_quantity_as_written"], ["2 tablets"])
        self.assertIsNone(parsed["medicine_candidate"])
        self.assertEqual(parsed["times_per_day_as_written"], 2)


class DocumentTests(unittest.TestCase):
    def test_decoded_bytes_are_canonical_and_not_named_by_user(self):
        image, data = decode_document(png())
        self.assertEqual(image.size, (200, 200))
        self.assertEqual(data, png())

    def test_pdf_single_page_and_rejects_multiple_pages(self):
        first = Image.new("RGB", (200, 200), "white")
        buffer = io.BytesIO()
        first.save(buffer, format="PDF")
        image, _ = decode_document(buffer.getvalue())
        self.assertGreater(image.width, 200)
        buffer = io.BytesIO()
        first.save(buffer, format="PDF", save_all=True, append_images=[first])
        with self.assertRaisesRegex(ValueError, "one prescription page"):
            decode_document(buffer.getvalue())

    def test_rejects_invalid_and_oversized_inputs(self):
        for data in [b"not an image", b"x" * (10 * 1024 * 1024 + 1)]:
            with self.assertRaises(ValueError):
                decode_document(data)


@override_settings(ALLOWED_HOSTS=["testserver"])
class WebsiteTests(SimpleTestCase):
    def setUp(self):
        self.store = JobStore()
        self.patch = patch("webapp.views.store", self.store)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.start = patch.object(self.store, "_ensure_started")
        self.start.start()
        self.addCleanup(self.start.stop)
        self.client = Client()
        response = self.client.post(
            "/",
            {"document": SimpleUploadedFile("private-name.png", png(), "image/png")},
        )
        self.url = response["Location"]
        self.identifier = self.url.strip("/").split("/")[1]
        self.job = self.store.jobs[self.identifier]

    def test_upload_and_cross_session_protection(self):
        self.assertEqual(self.client.get(self.url).status_code, 200)
        stranger = Client()
        for endpoint in ["", "image/", "status/", "export/"]:
            self.assertEqual(stranger.get(self.url + endpoint).status_code, 404)
        self.assertNotIn("private-name", self.client.get(self.url).content.decode())
        self.assertIn("no-store", self.client.get(self.url + "image/")["Cache-Control"])

    def test_csrf_is_required(self):
        client = Client(enforce_csrf_checks=True)
        self.assertEqual(client.post("/", {}).status_code, 403)

    def test_add_review_export_and_delete(self):
        self.client.post(self.url + "add/", dict(x1=0, y1=0, x2=0.8, y2=0.2))
        region_id = self.job.regions[0]["id"]
        response = self.client.post(
            self.url + "review/",
            {
                f"text_{region_id}": "Tab Paracetamol 500 mg twice daily",
                f"role_{region_id}": "medicine",
                f"checked_{region_id}": "on",
            },
        )
        self.assertEqual(response.status_code, 302)
        report = self.client.get(self.url + "export/").json()
        self.assertEqual(report["medications"][0]["times_per_day_as_written"], 2)
        self.assertEqual(report["status"], "reviewed_transcription")
        self.client.post(self.url + "delete/")
        self.assertEqual(self.client.get(self.url).status_code, 404)
        self.assertEqual(self.job.image, b"")

    def test_remove_medicine_invalidates_links(self):
        self.job.regions = regions()
        self.job.regions[2].update(medicine_id="2", reviewed=True)
        self.client.post(self.url + "remove/2/")
        self.assertEqual(self.job.regions[1]["medicine_id"], "")
        self.assertFalse(self.job.regions[1]["reviewed"])

    def test_review_is_atomic_and_escapes_ocr(self):
        self.job.regions = regions()
        before = copy.deepcopy(self.job.regions)
        self.assertEqual(
            self.client.post(self.url + "review/", {"role_1": "invalid"}).status_code,
            400,
        )
        self.assertEqual(self.job.regions, before)
        self.job.regions[0]["text"] = "<script>alert(1)</script>"
        html = self.client.get(self.url).content.decode()
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("<script>alert(1)", html)

    def test_unreviewed_names_do_not_trigger_network_lookup(self):
        self.job.regions = regions()
        with patch("webapp.views.MedicineKnowledge") as knowledge:
            response = self.client.post(
                self.url + "lookup/2/", {"name": "Paracetamol", "confirmed": "on"}
            )
            self.assertEqual(response.status_code, 400)
            knowledge.assert_not_called()

    def test_ingredient_matches_cannot_select_combination_labels(self):
        self.job.regions = regions()
        self.job.regions[1]["reviewed"] = True
        with patch("webapp.views.MedicineKnowledge") as knowledge:
            knowledge.return_value.lookup.return_value = {
                "candidates": [
                    {
                        "name": "ingredient",
                        "term_type": "IN",
                        "ingredients": ["ingredient"],
                        "labels": [
                            {"setid": "fixture", "title": "Combination product"}
                        ],
                    }
                ]
            }
            data = {"name": "ingredient", "confirmed": "on"}
            response = self.client.post(self.url + "lookup/2/", data)
            self.assertIn("Only the ingredient", response.content.decode())
            self.assertNotIn("Combination product", response.content.decode())
            self.client.post(
                self.url + "lookup/2/",
                {**data, "setid": "fixture", "product_confirmed": "on"},
            )
            knowledge.return_value.indications.assert_not_called()

    def test_lookup_source_errors_are_explicit(self):
        self.job.regions = regions()
        self.job.regions[1]["reviewed"] = True
        with patch("webapp.views.MedicineKnowledge") as knowledge:
            knowledge.return_value.lookup.side_effect = RuntimeError(
                "private diagnostic"
            )
            response = self.client.post(
                self.url + "lookup/2/", {"name": "Paracetamol", "confirmed": "on"}
            )
            self.assertContainsText(response, "unavailable right now")
            self.assertNotIn("private diagnostic", response.content.decode())

    def assertContainsText(self, response, text):
        self.assertIn(text, response.content.decode())

    def test_expiry_erases_image(self):
        self.job.created -= 1801
        self.assertEqual(self.client.get(self.url).status_code, 404)
        self.assertEqual(self.job.image, b"")

    def test_background_progress_and_cancellation(self):
        reader = Mock()
        reader.read.return_value = {"text": "Tab Example", "alternatives": []}
        with (
            patch(
                "webapp.jobs.detect_regions",
                return_value=[{"id": "1", "box": [0, 0, 100, 100]}],
            ),
            patch("webapp.jobs.get_reader", return_value=reader),
        ):
            self.store.run(self.job)
            self.assertEqual(self.job.state, "ready")
            self.assertEqual(self.job.completed, 1)
            self.store.delete(self.job)
            self.store.run(self.job)
            self.assertEqual(reader.read.call_count, 1)


@override_settings(ALLOWED_HOSTS=["testserver"])
class SavedReviewTests(SimpleTestCase):
    def setUp(self):
        self.store = JobStore()
        self.patches = [
            patch("webapp.views.store", self.store),
            patch.object(self.store, "_ensure_started"),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.client = Client()
        self.payload = {
            "schema_version": 1,
            "image_sha256": hashlib.sha256(png()).hexdigest(),
            "image_size": [200, 200],
            "regions": regions(),
        }

    def upload(self, payload=None, automatic=False):
        return self.client.post(
            "/",
            {
                "document": SimpleUploadedFile("page.png", png(), "image/png"),
                "annotations": SimpleUploadedFile(
                    "review.json",
                    json.dumps(
                        payload if payload is not None else self.payload
                    ).encode(),
                    "application/json",
                ),
                **({"automatic": "on"} if automatic else {}),
            },
        )

    def test_round_trip_preserves_corrections_links_and_checks(self):
        for r in self.payload["regions"]:
            r["reviewed"] = True
        self.payload["regions"][1]["text"] = "Corrected medicine"
        self.payload["regions"][2]["medicine_id"] = "2"
        response = self.upload(automatic=True)
        self.assertEqual(response.status_code, 302)
        job = next(iter(self.store.jobs.values()))
        self.assertEqual(job.state, "ready")
        self.assertTrue(self.store.queue.empty())
        result = self.client.get(response["Location"] + "export/").json()
        self.assertEqual(result["regions"][1]["text"], "Corrected medicine")
        self.assertEqual(result["regions"][2]["medicine_id"], "2")
        self.assertTrue(result["regions"][2]["reviewed"])
        self.assertFalse(result["medications"][0]["identity_confirmed"])

    def test_wrong_image_and_malformed_regions_do_not_create_jobs(self):
        malformed = []
        wrong = copy.deepcopy(self.payload)
        wrong["image_sha256"] = "0" * 64
        malformed.append(wrong)
        for field, value in [
            ("id", "../../1"),
            ("box", [0, 0, 201, 200]),
            ("reviewed", "false"),
            ("text", {}),
            ("role", []),
        ]:
            bad = copy.deepcopy(self.payload)
            bad["regions"][0][field] = value
            malformed.append(bad)
        duplicate = copy.deepcopy(self.payload)
        duplicate["regions"].append(duplicate["regions"][0])
        malformed.append(duplicate)
        dangling = copy.deepcopy(self.payload)
        dangling["regions"][2]["medicine_id"] = "99"
        malformed.append(dangling)
        for payload in malformed:
            with self.subTest(payload=payload):
                self.assertEqual(self.upload(payload).status_code, 200)
                self.assertEqual(len(self.store.jobs), 0)

    def test_oversized_annotation_is_not_silently_ignored(self):
        response = self.client.post(
            "/",
            {
                "document": SimpleUploadedFile("page.png", png(), "image/png"),
                "annotations": SimpleUploadedFile("huge.json", b" " * (512 * 1024 + 1)),
                "automatic": "on",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("exceeded its size limit", response.content.decode())
        self.assertEqual(len(self.store.jobs), 0)

    def test_imported_ocr_reads_only_empty_regions_and_keeps_roles(self):
        self.payload["regions"][0]["reviewed"] = True
        self.payload["regions"][1].update(text="", ocr_text="", role="medicine")
        response = self.upload(automatic=True)
        self.assertEqual(response.status_code, 302)
        job, region_id, empty_only = self.store.queue.get_nowait()
        self.assertTrue(empty_only)
        reader = Mock()
        reader.read.return_value = {"text": "Draft medicine"}
        with (
            patch("webapp.jobs.detect_regions") as detector,
            patch("webapp.jobs.get_reader", return_value=reader),
        ):
            self.store.run(job, region_id, empty_only)
            detector.assert_not_called()
        self.assertEqual(reader.read.call_count, 1)
        self.assertEqual(job.completed, 1)
        self.assertEqual(job.total, 1)
        self.assertEqual(job.regions[1]["text"], "Draft medicine")
        self.assertEqual(job.regions[1]["role"], "medicine")
        self.assertFalse(job.regions[1]["reviewed"])
        self.assertTrue(job.regions[0]["reviewed"])
        self.assertEqual(job.regions[2]["text"], self.payload["regions"][2]["text"])

    def test_partial_failure_keeps_completed_drafts_and_retry_skips_them(self):
        for row in self.payload["regions"][:2]:
            row.update(text="", ocr_text="", reviewed=False)
        response = self.upload()
        job = next(iter(self.store.jobs.values()))
        reader = Mock()
        reader.read.side_effect = [
            {"text": "First completed"},
            RuntimeError("private data"),
        ]
        with patch("webapp.jobs.get_reader", return_value=reader):
            self.store.run(job, empty_only=True)
        self.assertEqual(job.state, "failed")
        self.assertEqual(job.regions[0]["text"], "First completed")
        self.assertEqual(job.completed, 1)
        self.assertNotIn("private data", job.stage)
        response = self.client.post(response["Location"] + "read-empty/")
        self.assertEqual(response.status_code, 302)
        task = self.store.queue.get_nowait()
        reader.read.side_effect = [{"text": "Second completed"}]
        with patch("webapp.jobs.get_reader", return_value=reader):
            self.store.run(*task)
        self.assertEqual(job.state, "ready")
        self.assertEqual(job.completed, 1)
        self.assertEqual(job.regions[0]["text"], "First completed")

    def test_cancel_during_imported_read_does_not_restore_erased_data(self):
        self.payload["regions"][0].update(text="", ocr_text="")
        self.upload()
        job = next(iter(self.store.jobs.values()))
        reader = Mock()

        def cancel(image):
            self.store.delete(job)
            return {"text": "must not persist"}

        reader.read.side_effect = cancel
        with patch("webapp.jobs.get_reader", return_value=reader):
            self.store.run(job, empty_only=True)
        self.assertEqual(job.image, b"")
        self.assertEqual(job.regions, [])


class PreparationTests(unittest.TestCase):
    def fixture(self, root):
        (root / "image.png").write_bytes(png())
        rows = regions()
        for row in rows:
            row["reviewed"] = True
        rows[2]["medicine_id"] = rows[3]["medicine_id"] = "2"
        review = {
            "schema_version": 1,
            "image_sha256": hashlib.sha256(png()).hexdigest(),
            "image_size": [200, 200],
            "regions": rows,
        }
        (root / "review.json").write_text(json.dumps(review))
        manifest = {
            "permission_confirmed": True,
            "deidentified": True,
            "documents": [
                {
                    "image": "image.png",
                    "review": "review.json",
                    "group": "writer-a",
                    "split": "train",
                }
            ],
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_preparation_is_separate_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            # Different white crops with contradictory labels are intentionally rejected.
            with self.assertRaisesRegex(ValueError, "Duplicate crop"):
                prepare(root / "manifest.json", root / "out")
            review = json.loads((root / "review.json").read_text())
            review["regions"] = review["regions"][:3]
            (root / "review.json").write_text(json.dumps(review))
            report = prepare(root / "manifest.json", root / "out")
            self.assertEqual(report["crops"], 2)
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                prepare(root / "manifest.json", root / "out")

    def test_training_preflight_verifies_images_and_split_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            review = json.loads((root / "review.json").read_text())
            review["regions"] = review["regions"][:3]
            (root / "review.json").write_text(json.dumps(review))
            buffer = io.BytesIO()
            Image.new("RGB", (200, 200), "black").save(buffer, format="PNG")
            (root / "second.png").write_bytes(buffer.getvalue())
            second = copy.deepcopy(review)
            second["image_sha256"] = hashlib.sha256(buffer.getvalue()).hexdigest()
            (root / "second.json").write_text(json.dumps(second))
            manifest["documents"].append(
                {
                    "image": "second.png",
                    "review": "second.json",
                    "group": "writer-b",
                    "split": "validation",
                }
            )
            (root / "manifest.json").write_text(json.dumps(manifest))
            prepare(root / "manifest.json", root / "out")
            report = verify(root / "out")
            self.assertEqual(report["crops"], 4)
            from scripts.train import preflight

            checked = preflight(
                {
                    "dataset_format": "reviewed_pages",
                    "dataset_root": str(root / "out"),
                    "model_id": "fixture",
                    "max_target_length": 64,
                    "output_dir": str(root / "model"),
                }
            )
            self.assertFalse(checked["training_started"])
            first = next((root / "out/images").iterdir())
            first.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                verify(root / "out")

    def test_draft_labels_and_cross_group_splits_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            review = json.loads((root / "review.json").read_text())
            review["regions"][0]["reviewed"] = False
            (root / "review.json").write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError, "reviewed role"):
                prepare(root / "manifest.json", root / "out")
            review["regions"] = review["regions"][:3]
            review["regions"][0]["reviewed"] = True
            (root / "review.json").write_text(json.dumps(review))
            manifest["documents"].append({**manifest["documents"][0], "split": "test"})
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "group crosses"):
                prepare(root / "manifest.json", root / "out")


if __name__ == "__main__":
    unittest.main()
