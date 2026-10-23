import unittest
from app.presentation import render_readings, render_matches


class PresentationTests(unittest.TestCase):
    def test_untrusted_ocr_is_escaped_and_absent_fields_are_readable(self):
        value = render_readings(
            {
                "lines": [
                    {
                        "region_id": "1",
                        "raw_text": "<img src=x onerror=alert(1)>",
                        "medicine_candidate": "<script>",
                        "ocr": {"alternatives": [{"text": "<b>fake</b>"}]},
                    }
                ]
            }
        )
        self.assertNotIn("<img", value)
        self.assertNotIn("<script>", value)
        self.assertIn("&lt;img", value)
        self.assertIn("Not found", value)
        self.assertIn("Review required", value)

    def test_source_names_are_escaped(self):
        result = render_matches(
            {"candidates": [{"name": "<script>", "ingredients": ["<b>"], "labels": []}]}
        )
        self.assertNotIn("<script>", result)
        self.assertIn("&lt;script&gt;", result)

    def test_unknown_name_does_not_show_verified_identity(self):
        self.assertIn("No exact match found", render_matches({"candidates": []}))


if __name__ == "__main__":
    unittest.main()
