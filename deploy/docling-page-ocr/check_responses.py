"""Exercise task results and page-selective OCR through docling-serve's public API."""

import io
import threading
import time
import unittest
from unittest.mock import patch

from docling.exceptions import ConversionError
from docling_jobkit.convert.manager import DoclingConverterManager
from docling_serve.app import create_app
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont


class TaskResponses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = cls.enterClassContext(TestClient(create_app()))
        cls.files = {"files": ("sample.pdf", b"%PDF-1.4\n%%EOF", "application/pdf")}

    def test_async_document_failure_has_an_id_and_cause(self):
        release = threading.Event()

        def convert(**kwargs):
            if not release.wait(10):
                raise RuntimeError("test timed out")
            raise ConversionError("sample page dimensions missing")

        with patch.object(DoclingConverterManager, "convert_documents", side_effect=convert):
            queued = self.client.post("/v1/convert/file/async", files=self.files)
            queued.raise_for_status()
            task_id = queued.json()["task_id"]
            try:
                pending = self.client.get(f"/v1/result/{task_id}")
                self.assertEqual(pending.status_code, 202)
                self.assertEqual(pending.json()["task_id"], task_id)
            finally:
                release.set()
            for _ in range(100):
                result = self.client.get(f"/v1/result/{task_id}")
                if result.status_code != 202:
                    break
                time.sleep(0.05)
            self.assertEqual(result.status_code, 200)
            body = result.json()
            self.assertEqual((body["task_id"], body["status"]), (task_id, "failure"))
            self.assertEqual(
                body["errors"],
                [
                    {
                        "code": "conversion_failed",
                        "message": "sample page dimensions missing",
                    }
                ],
            )

    def test_backend_failure(self):
        with patch.object(
            DoclingConverterManager,
            "convert_documents",
            side_effect=RuntimeError("worker unavailable"),
        ):
            result = self.client.post("/v1/convert/file", files=self.files)
        self.assertEqual(result.status_code, 503)
        self.assertTrue(result.json()["task_id"])
        self.assertEqual(result.json()["errors"][0]["code"], "backend_error")

    def test_unknown_task(self):
        self.assertEqual(self.client.get("/v1/result/unknown-task").status_code, 404)

    def test_ocr_distinguishes_page_label_from_blank_page(self):
        pdf = io.BytesIO()
        with Image.new("RGB", (1000, 420), "white") as blank, blank.copy() as drawing:
            font = ImageFont.load_default(size=70)
            ImageDraw.Draw(drawing).text((80, 120), "PLAN 125", fill="black", font=font)
            blank.save(pdf, format="PDF", save_all=True, append_images=[drawing], resolution=150)

        for page, status in ((2, "text"), (1, "no_text")):
            with self.subTest(page=page):
                result = self.client.post(
                    "/v1/convert/pdf-pages/ocr",
                    data={"page_count": "2", "pages": str(page), "languages": ["de", "en"]},
                    files={"file": ("drawing.pdf", pdf.getvalue(), "application/pdf")},
                )
                self.assertEqual(result.status_code, 200)
                rows = result.json()["pages"]
                self.assertEqual([(r["page"], r["status"]) for r in rows], [(page, status)])
                if status == "text":
                    self.assertIn("PLAN 125", " ".join(rows[0]["text"].upper().split()))
                else:
                    self.assertEqual(rows[0]["text"], "")


if __name__ == "__main__":
    unittest.main()
