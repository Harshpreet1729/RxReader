"""Decode bounded single-page uploads without preserving filename or EXIF data."""

import io
import threading
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000
PDF_LOCK = threading.Lock()


def decode_document(data):
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Choose an image or single-page PDF no larger than 10 MB.")
    try:
        if data.startswith(b"%PDF-"):
            import pypdfium2 as pdfium

            # PDFium calls must not run concurrently in this process.
            with PDF_LOCK:
                with pdfium.PdfDocument(data) as pdf:
                    if len(pdf) != 1:
                        raise ValueError("Upload one prescription page at a time.")
                    page = pdf[0]
                    try:
                        w, h = page.get_size()
                        if w <= 0 or h <= 0:
                            raise ValueError("Invalid PDF dimensions.")
                        scale = min(2.5, 2600 / max(w, h))
                        bitmap = page.render(scale=scale)
                        try:
                            image = bitmap.to_pil().convert("RGB").copy()
                        finally:
                            bitmap.close()
                    finally:
                        page.close()
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as original:
                    if (
                        original.format not in {"JPEG", "PNG", "WEBP"}
                        or getattr(original, "n_frames", 1) != 1
                    ):
                        raise ValueError(
                            "Use a still JPG, PNG, WebP or a single-page PDF."
                        )
                    if original.width * original.height > MAX_PIXELS:
                        raise ValueError("Resize the image to 16 megapixels or less.")
                    oriented = ImageOps.exif_transpose(original)
                    if (
                        oriented.mode in {"RGBA", "LA"}
                        or "transparency" in oriented.info
                    ):
                        rgba = oriented.convert("RGBA")
                        image = Image.new("RGB", rgba.size, "white")
                        image.paste(rgba, mask=rgba.getchannel("A"))
                    else:
                        image = oriented.convert("RGB")
        if min(image.size) < 32:
            raise ValueError("The image is too small to read.")
        output = io.BytesIO()
        image.save(output, format="PNG")
        return image, output.getvalue()
    except ValueError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("Could not read this image. Try a clear JPG or PNG.") from exc
    except Exception as exc:
        raise ValueError(
            "Could not open this PDF. Check that it is valid and not password protected."
        ) from exc
