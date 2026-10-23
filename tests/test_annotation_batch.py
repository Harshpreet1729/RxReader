import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from PIL import Image
from scripts.prepare_annotation_batch import redacted_copy, read_batch


class AnnotationBatchTests(unittest.TestCase):
    def test_redaction_preserves_box_pixels_and_removes_context(self):
        image = Image.new("RGB", (100, 100), "red")
        regions = [{"box": [20, 20, 50, 50]}]
        result = Image.open(io.BytesIO(redacted_copy(image, [10, 10, 60, 60], regions)))
        self.assertEqual(result.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(result.getpixel((30, 30)), (255, 0, 0))
        with self.assertRaisesRegex(ValueError, "cut"):
            redacted_copy(image, [30, 30, 60, 60], regions)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            redacted_copy(image, [10, 10, 60, 60], regions, [[40, 40, 55, 55]])

    def test_interrupted_read_resumes_without_overwriting_corrections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            buffer = io.BytesIO()
            Image.new("RGB", (100, 100), "white").save(buffer, format="PNG")
            png = buffer.getvalue()
            (root / "page.png").write_bytes(png)
            regions = [
                {
                    "id": str(i),
                    "box": [10, 10, 50, 50],
                    "text": "",
                    "ocr_text": "",
                    "role": "medicine",
                    "medicine_id": "",
                    "reviewed": False,
                }
                for i in range(1, 4)
            ]
            regions[0].update(text="Human correction", reviewed=True)
            payload = {
                "schema_version": 1,
                "image_sha256": hashlib.sha256(png).hexdigest(),
                "image_size": [100, 100],
                "regions": regions,
            }
            (root / "review.json").write_text(json.dumps(payload))
            (root / "manifest.draft.json").write_text(
                json.dumps(
                    {
                        "documents": [
                            {
                                "page_id": "1",
                                "image": "page.png",
                                "image_sha256": payload["image_sha256"],
                                "review": "review.json",
                                "regions": 3,
                            }
                        ]
                    }
                )
            )
            reader = Mock()
            reader.read.side_effect = [
                {"text": "First draft"},
                RuntimeError("interrupted"),
            ]
            with self.assertRaises(RuntimeError):
                read_batch(root, reader, "fixture")
            saved = json.loads((root / "review.json").read_text())
            self.assertEqual(saved["regions"][0]["text"], "Human correction")
            self.assertTrue(saved["regions"][0]["reviewed"])
            self.assertEqual(saved["regions"][1]["text"], "First draft")
            reader.read.side_effect = [{"text": "Second draft"}]
            report = read_batch(root, reader, "fixture")
            self.assertEqual(report["nonempty_readings"], 3)
            self.assertEqual(report["reviewed_regions"], 1)
            self.assertEqual(reader.read.call_count, 3)
            (root / "page.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "image changed"):
                read_batch(root, reader, "fixture")


if __name__ == "__main__":
    unittest.main()
