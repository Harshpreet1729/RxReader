"""Reject oversized uploads while streaming; never spool prescriptions to disk."""

import io
from django.core.files.uploadhandler import FileUploadHandler, StopUpload
from django.core.files.uploadedfile import InMemoryUploadedFile
from src.prescription.documents import MAX_BYTES
from src.prescription.annotations import MAX_ANNOTATION_BYTES


class BoundedMemoryUploadHandler(FileUploadHandler):
    def new_file(self, *args, **kwargs):
        super().new_file(*args, **kwargs)
        self.buffer = io.BytesIO()
        self.received = 0

    def receive_data_chunk(self, raw_data, start):
        self.received += len(raw_data)
        if self.received > (
            MAX_ANNOTATION_BYTES if self.field_name == "annotations" else MAX_BYTES
        ):
            self.request.upload_error = "An uploaded file exceeded its size limit (image: 10 MB; saved review: 512 KB)."
            self.buffer.close()
            raise StopUpload(connection_reset=True)
        self.buffer.write(raw_data)
        return None

    def file_complete(self, file_size):
        self.buffer.seek(0)
        return InMemoryUploadedFile(
            self.buffer,
            self.field_name,
            self.file_name,
            self.content_type,
            file_size,
            self.charset,
        )
