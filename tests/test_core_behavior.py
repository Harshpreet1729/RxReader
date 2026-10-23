"""Regression cases for the readable core flow; no downloads or training."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from app.gradio_app import transcribe
from scripts.train import check_output_directory
from src.prescription.knowledge import MedicineKnowledge
from src.prescription.metrics import metrics


class CoreBehaviorTests(unittest.TestCase):
    def test_gradio_keeps_alternatives_and_literal_fields(self):
        reading = {
            "text": "Tab Example 500 mg twice daily",
            "alternatives": [{"text": "Tab Example 500 mg twice daily"}],
            "possibly_truncated": False,
        }
        with patch("app.gradio_app.get_reader") as load_reader:
            load_reader.return_value.read.return_value = reading
            report = transcribe(
                Image.new("RGB", (100, 40)), "word", None, "local", None
            )

        line = report["lines"][0]
        self.assertEqual(line["medicine_candidate"], "Example")
        self.assertEqual(line["strength_as_written"], ["500 mg"])
        self.assertEqual(line["times_per_day_as_written"], 2)
        self.assertEqual(line["ocr"]["alternatives"], reading["alternatives"])
        self.assertEqual(report["status"], "needs_review")

    def test_metrics_include_insertions_deletions_and_unicode_normalization(self):
        result = metrics(["abcd", "xy"], ["abXcd", "x"])
        self.assertAlmostEqual(result["raw_cer"], 2 / 6)
        self.assertEqual(result["raw_wer"], 1)
        normalized = metrics(["  ＡＢ  C "], ["ab c"])
        self.assertEqual(normalized["normalized_exact_match"], 1)
        self.assertGreater(normalized["raw_cer"], 0)

    def test_lookup_keeps_multiple_ingredients_and_product_labels(self):
        client = MedicineKnowledge(Mock())
        client._get = Mock(
            side_effect=[
                Mock(json=Mock(return_value={"idGroup": {"rxnormId": ["123"]}})),
                Mock(
                    json=Mock(
                        return_value={"properties": {"name": "Example", "tty": "SCD"}}
                    )
                ),
                Mock(
                    json=Mock(
                        return_value={
                            "relatedGroup": {
                                "conceptGroup": [
                                    {
                                        "conceptProperties": [
                                            {"name": "Ingredient A"},
                                            {"name": "Ingredient B"},
                                        ]
                                    },
                                    {"conceptProperties": [{"name": "Ingredient C"}]},
                                ]
                            }
                        }
                    )
                ),
                Mock(
                    json=Mock(
                        return_value={
                            "data": [
                                {"setid": "one", "title": "Product one"},
                                {"setid": "two", "title": "Product two"},
                            ]
                        }
                    )
                ),
            ]
        )
        candidate = client.lookup("Example")["candidates"][0]
        self.assertEqual(
            candidate["ingredients"], ["Ingredient A", "Ingredient B", "Ingredient C"]
        )
        self.assertEqual(
            [label["setid"] for label in candidate["labels"]], ["one", "two"]
        )

    def test_training_resume_checks_data_and_never_overwrites_a_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            checkpoint = output / "checkpoint-10"
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.json").write_text("{}")
            config = {"seed": 42}
            previous = {"config": config}
            for split in ("train", "validation"):
                contents = b"filename,label\na.png,Example\n"
                (root / f"{split}.csv").write_bytes(contents)
                previous[f"{split}_csv_sha256"] = hashlib.sha256(contents).hexdigest()
            (output / "run_config.json").write_text(json.dumps(previous))

            self.assertEqual(
                check_output_directory(config, root, output, checkpoint),
                checkpoint.resolve(),
            )
            with self.assertRaisesRegex(ValueError, "Output is not empty"):
                check_output_directory(config, root, output, None)
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                check_output_directory({"seed": 1}, root, output, checkpoint)

            (root / "validation.csv").write_text("changed labels")
            with self.assertRaisesRegex(ValueError, "dataset differs"):
                check_output_directory(config, root, output, checkpoint)


if __name__ == "__main__":
    unittest.main()
