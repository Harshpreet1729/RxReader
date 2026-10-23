import tempfile
import unittest
from pathlib import Path

from scripts.prepare_public_pages import parse_box, prepare


class PublicPagePreparationTests(unittest.TestCase):
    def test_source_medicine_box_conversion(self):
        self.assertEqual(parse_box("1 0.5 0.4 0.2 0.2", (100, 200)), [40, 60, 60, 100])

    def test_invalid_or_unexpected_source_boxes_rejected(self):
        for line in [
            "0 0.5 0.5 0.2 0.2",
            "1 nan 0.5 0.2 0.2",
            "1 0.99 0.5 0.2 0.2",
            "1 0.5 0.5 -0.1 0.2",
            "1 0.5 0.5 0.00001 0.00001",
            "1 0.5 0.5",
        ]:
            with self.subTest(line=line), self.assertRaises(ValueError):
                parse_box(line, (100, 200))

    def test_existing_output_and_changed_archive_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "changed.zip"
            archive.write_bytes(b"not the publisher archive")
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare(archive, root)
            with self.assertRaisesRegex(ValueError, "pinned publisher"):
                prepare(archive, root / "prepared")
            self.assertFalse((root / "prepared").exists())


if __name__ == "__main__":
    unittest.main()
