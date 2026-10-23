from django import forms
from src.prescription.documents import MAX_BYTES
from src.prescription.annotations import MAX_ANNOTATION_BYTES


class UploadForm(forms.Form):
    document = forms.FileField(
        label="Prescription photo or PDF",
        widget=forms.ClearableFileInput(attrs={"accept": ".jpg,.jpeg,.png,.webp,.pdf"}),
    )
    annotations = forms.FileField(
        required=False,
        label="Saved review or prepared page annotations (optional)",
        help_text="Choose the matching review JSON to restore boxes and corrections. Maximum 512 KB.",
        widget=forms.ClearableFileInput(attrs={"accept": ".json"}),
    )
    automatic = forms.BooleanField(
        required=False,
        initial=True,
        label="Read automatically (with saved annotations: fill empty regions only)",
    )

    def clean_document(self):
        document = self.cleaned_data["document"]
        if document.size > MAX_BYTES:
            raise forms.ValidationError("Maximum file size is 10 MB.")
        return document

    def clean_annotations(self):
        annotations = self.cleaned_data.get("annotations")
        if annotations and annotations.size > MAX_ANNOTATION_BYTES:
            raise forms.ValidationError("Saved review must be no larger than 512 KB.")
        return annotations
