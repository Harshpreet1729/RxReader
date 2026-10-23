"""Bounded in-memory worker for the LOCAL single-process Django prototype.

No patient image/report is written to disk. Jobs expire after 30 minutes; restart
clears them. A production deployment must replace this with an isolated task queue.
"""

import copy
import io
import queue
import secrets
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from PIL import Image
from src.prescription.data import ROOT, load_config, project_path
from src.prescription.layout import detect_regions
from src.prescription.ocr import TrOCRReader
from src.prescription.page_review import propose_regions
from src.prescription.annotations import MAX_REGIONS

TTL = 30 * 60  # Seconds; temporary uploads expire after 30 minutes.
MAX_JOBS = 6


@lru_cache(maxsize=1)
def get_reader():
    import torch

    config = load_config()
    local = ROOT / "configs/app.json"
    if local.exists():
        path = project_path(load_config(local)["model_path"])
        if not path.is_dir():
            raise RuntimeError("Configured model is missing.")
        model, revision = str(path), None
    else:
        model, revision = config["model_id"], config["model_revision"]
    return TrOCRReader(
        model, revision=revision, device="cuda" if torch.cuda.is_available() else "cpu"
    )


@dataclass
class Job:
    """One uploaded page, its progress, and its editable region readings."""

    owner: str
    image: bytes
    size: tuple
    id: str = field(default_factory=lambda: secrets.token_hex(16))
    created: float = field(default_factory=time.monotonic)
    stage: str = "Queued"
    state: str = "queued"
    completed: int = 0
    total: int = 0
    regions: list = field(default_factory=list)
    cancelled: bool = False
    lock: object = field(default_factory=threading.RLock)


class JobStore:
    """Keep jobs in memory and process one OCR request at a time.

    The request thread updates the page while a worker thread runs the model.
    Locks stop them from changing the same job at the same time.
    """

    def __init__(self):
        self.jobs = {}
        self.lock = threading.RLock()
        self.queue = queue.Queue(maxsize=MAX_JOBS)
        self.started = False
        self.start_lock = threading.Lock()

    def purge(self):
        now = time.monotonic()
        with self.lock:
            for key, job in list(self.jobs.items()):
                if now - job.created >= TTL:
                    with job.lock:
                        job.cancelled = True
                        job.image = b""
                        job.regions = []
                    del self.jobs[key]

    def create(self, owner, image, size, automatic, regions=None):
        self.purge()
        with self.lock:
            if len(self.jobs) >= MAX_JOBS:
                raise ValueError(
                    "Workspace is full. Delete an old prescription and try again."
                )
            job = Job(owner, image, size)
            self.jobs[job.id] = job
            if regions is not None:
                job.regions = copy.deepcopy(regions)
                automatic = automatic and any(
                    not r["text"].strip() and not r["reviewed"] for r in regions
                )
            if not automatic:
                job.state = "ready"
                job.stage = (
                    "Saved review restored"
                    if regions is not None
                    else "Add regions to begin"
                )
        self._ensure_started()
        if automatic:
            try:
                self.submit(job, empty_only=regions is not None)
            except ValueError:
                self.delete(job)
                raise
        return job

    def get(self, identifier, owner):
        self.purge()
        with self.lock:
            job = self.jobs.get(identifier)
            if job is None or not secrets.compare_digest(job.owner, owner):
                raise KeyError(identifier)
            return job

    def delete(self, job):
        with self.lock, job.lock:
            job.cancelled = True
            job.image = b""
            job.regions = []
            self.jobs.pop(job.id, None)

    def _ensure_started(self):
        with self.start_lock:
            if not self.started:
                self.started = True
                threading.Thread(
                    target=self._worker, daemon=True, name="prescription-reader"
                ).start()
                threading.Thread(
                    target=self._reaper, daemon=True, name="prescription-expiry"
                ).start()

    def submit(self, job, region_id=None, empty_only=False):
        self._ensure_started()
        with job.lock:
            if job.cancelled:
                raise ValueError("This prescription has expired.")
            job.state = "queued"
            job.stage = "Queued"
            job.completed = 0
            job.total = 0
            try:
                self.queue.put_nowait((job, region_id, empty_only))
            except queue.Full as exc:
                job.state = "failed"
                job.stage = (
                    "Reader is busy. Try again after the current reading finishes."
                )
                raise ValueError(job.stage) from exc

    def _reaper(self):
        while True:
            time.sleep(30)
            self.purge()

    def _worker(self):
        """Wait for queued work; queue.get() blocks until a job is available."""
        while True:
            job, region_id, empty_only = self.queue.get()
            try:
                self.run(job, region_id, empty_only)
            finally:
                self.queue.task_done()

    def _read_empty(self, job, image):
        with job.lock:
            regions = [
                copy.deepcopy(r)
                for r in job.regions
                if not r["text"].strip() and not r["reviewed"]
            ]
            job.total = len(regions)
            job.stage = (
                "Loading handwriting model" if regions else "No empty regions to read"
            )
        reader = get_reader() if regions else None
        for index, region in enumerate(regions, 1):
            with job.lock:
                if job.cancelled:
                    return
                job.stage = f"Reading empty region {index} of {len(regions)}"
            reading = reader.read(image.crop(region["box"]))
            region.update(
                ocr_text=reading["text"],
                text=reading["text"],
                reviewed=False,
                alternatives=reading.get("alternatives", []),
                possibly_truncated=reading.get("possibly_truncated", False),
            )
            # Keep the imported role/link; it is still an unreviewed annotation.
            with job.lock:
                if job.cancelled:
                    return
                job.regions = [
                    region if r["id"] == region["id"] else r for r in job.regions
                ]
                job.completed = index
        with job.lock:
            if not job.cancelled:
                job.state = "ready"
                job.stage = "Ready for review"

    def run(self, job, region_id=None, empty_only=False):
        try:
            with job.lock:
                if job.cancelled:
                    return
                job.state = "running"
                if empty_only:
                    job.stage = "Preparing saved regions"
                elif region_id:
                    job.stage = "Preparing selected region"
                else:
                    job.stage = "Finding text regions"
                image = Image.open(io.BytesIO(job.image)).convert("RGB")
            if empty_only:
                self._read_empty(job, image)
                return
            if region_id:
                with job.lock:
                    regions = [dict(r) for r in job.regions if r["id"] == region_id]
            else:
                regions = detect_regions(image, "page")
            if len(regions) > MAX_REGIONS:
                raise ValueError("Too many regions. Use manual regions for this page.")
            with job.lock:
                if job.cancelled:
                    return
                job.total = len(regions)
                job.stage = "Loading handwriting model"
            reader = get_reader()
            read = []
            for region in regions:
                with job.lock:
                    if job.cancelled:
                        return
                    job.stage = f"Reading region {len(read)+1} of {len(regions)}"
                reading = reader.read(image.crop(region["box"]))
                region.update(
                    ocr_text=reading["text"],
                    alternatives=reading.get("alternatives", []),
                    possibly_truncated=reading.get("possibly_truncated", False),
                )
                read.append(region)
                with job.lock:
                    job.completed = len(read)
            proposals = propose_regions(read)
            with job.lock:
                if job.cancelled:
                    return
                if region_id:
                    replacement = proposals[0]
                    job.regions = [
                        replacement if r["id"] == region_id else r for r in job.regions
                    ]
                else:
                    job.regions = proposals
                job.state = "ready"
                job.stage = "Ready for review"
        except Exception as error:
            # Do not put exception text or OCR/image data into logs or HTTP errors.
            with job.lock:
                if not job.cancelled:
                    job.state = "failed"
                    job.stage = "Reading could not finish. Check the local model/page dependencies, or add regions manually."
                    job.error_type = type(error).__name__


store = JobStore()
