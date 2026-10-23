import unittest
from unittest.mock import patch, Mock
from PIL import Image
from src.prescription.extraction import extract_line
from src.prescription.layout import checked_box, detect_regions
from src.prescription.metrics import metrics
from src.prescription.pipeline import process_image
from src.prescription.knowledge import MedicineKnowledge
from scripts.prepare_rxhandbd import assign_splits


class ExtractionTests(unittest.TestCase):
    def test_explicit_frequency(self):
        r = extract_line("Tab Paracetamol 500 mg twice daily for 3 days")
        self.assertEqual(r["medicine_candidate"], "Paracetamol")
        self.assertEqual(r["times_per_day_as_written"], 2)
        self.assertEqual(r["strength_as_written"], ["500 mg"])
        self.assertEqual(r["duration_as_written"], "for 3 days")
        self.assertTrue(r["needs_review"])

    def test_no_invented_frequency(self):
        for text in [
            "Napa",
            "Napa BD",
            "Napa 1-0-1",
            "Napa every 8 hours",
            "Napa twice daily OD",
        ]:
            self.assertIsNone(extract_line(text)["times_per_day_as_written"])

    def test_conflicting_frequency(self):
        self.assertIsNone(
            extract_line("Napa once daily twice daily")["times_per_day_as_written"]
        )

    def test_strength_not_times_per_day(self):
        self.assertIsNone(
            extract_line("Napa 3 mg for 2 days")["times_per_day_as_written"]
        )


class PipelineTests(unittest.TestCase):
    def test_templates_and_separate_lines(self):
        reader = Mock()
        reader.read.side_effect = [
            {"text": "Napa", "possibly_truncated": False},
            {"text": "twice daily", "possibly_truncated": False},
        ]
        template = {
            "coordinate_space": "normalized",
            "regions": [
                {"id": "1", "box": [0, 0, 1, 0.5]},
                {"id": "2", "box": [0, 0.5, 1, 1]},
            ],
        }
        result = process_image(Image.new("RGB", (100, 100)), reader, "page", template)
        self.assertEqual(result["lines"][0]["box"], [0, 0, 100, 50])
        self.assertIsNone(result["lines"][0]["times_per_day_as_written"])
        self.assertIsNone(result["lines"][1]["medicine_candidate"])
        self.assertEqual(result["status"], "needs_review")

    def test_invalid_regions(self):
        for box in [[1, 1, 0, 0], [0, 0, 2, 1], [0, float("nan"), 1, 1]]:
            with self.assertRaises(ValueError):
                checked_box(box, (100, 100), True)
        with self.assertRaises(ValueError):
            detect_regions(
                Image.new("RGB", (100, 100)),
                template={"coordinate_space": "pixels", "regions": []},
            )

    def test_detection_adapter_contract(self):
        import types
        import sys
        import numpy  # load compiled extension before the temporary module patch

        detection = Mock()
        detection.return_value.predict.return_value = [
            types.SimpleNamespace(
                json={"res": {"dt_polys": [[[1, 1], [90, 1], [90, 20], [1, 20]]]}}
            )
        ]
        with patch.dict(
            sys.modules, {"paddleocr": types.SimpleNamespace(TextDetection=detection)}
        ):
            regions = detect_regions(Image.new("RGB", (100, 100)), "page")
        self.assertEqual(regions, [{"id": "1", "box": [1, 1, 90, 20]}])


class MetricTests(unittest.TestCase):
    def test_weighted_cer(self):
        r = metrics(["abcd", "x"], ["abce", ""])
        self.assertEqual(r["raw_cer"], 0.4)
        self.assertEqual(r["normalized_exact_match"], 0)

    def test_normalization_and_empty(self):
        self.assertEqual(metrics(["NAPA"], ["napa"])["normalized_cer"], 0)
        self.assertEqual(metrics(["x"], [""])["raw_cer"], 1)
        with self.assertRaises(ValueError):
            metrics([], [])


class KnowledgeTests(unittest.TestCase):
    def test_unknown_has_no_guess(self):
        session = Mock()
        session.get.return_value.json.return_value = {"idGroup": {}}
        r = MedicineKnowledge(session).lookup("UnknownBrandXYZ")
        self.assertEqual(r["status"], "unresolved")
        self.assertEqual(r["candidates"], [])
        self.assertEqual(session.get.call_args.kwargs["params"]["search"], 0)

    def test_dailymed_source_extraction(self):
        client = MedicineKnowledge(Mock())
        setid = "12345678-1234-1234-1234-123456789012"
        label = {
            "setid": setid,
            "title": "Fixture product",
            "url": "https://dailymed.nlm.nih.gov/",
        }
        client.lookup = Mock(return_value={"candidates": [{"labels": [label]}]})
        xml = b'<document xmlns="urn:hl7-org:v3"><section><code code="34067-9"/><text>Fixture indication only.</text></section><section><code code="34068-7"/><text>Do not extract dosing.</text></section></document>'
        client._get = Mock(return_value=Mock(content=xml))
        self.assertEqual(
            client.indications("Fixture", setid)["uses_as_published"],
            ["Fixture indication only."],
        )
        with self.assertRaises(ValueError):
            client.indications("Fixture", "87654321-1234-1234-1234-123456789012")

    def test_failure_propagates_without_fabrication(self):
        client = MedicineKnowledge(Mock())
        client._get = Mock(side_effect=RuntimeError("offline"))
        with self.assertRaisesRegex(RuntimeError, "offline"):
            client.lookup("Fixture")


class TrainingGuardTests(unittest.TestCase):
    def test_default_cli_cannot_train(self):
        from scripts import train

        with (
            patch("sys.argv", ["train"]),
            patch.object(train, "preflight", return_value={"training_started": False}),
            patch.object(train, "train") as run,
        ):
            train.main()
        run.assert_not_called()

    def test_training_rejects_cpu_before_model_download(self):
        from scripts.train import train

        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "disabled on CPU"):
                train({})

    def test_duplicate_groups_stay_out_of_training(self):
        rows = [
            {
                "filename": "a",
                "label": "Napa",
                "source_split": "train",
                "pixel_sha256": "same",
            },
            {
                "filename": "b",
                "label": "Napa",
                "source_split": "test",
                "pixel_sha256": "same",
            },
            {
                "filename": "c",
                "label": "???",
                "source_split": "test",
                "pixel_sha256": "other",
            },
        ]
        assign_splits(rows)
        self.assertEqual(
            [r["split"] for r in rows], ["duplicate", "test", "quarantine"]
        )


if __name__ == "__main__":
    unittest.main()
