"""Local review interface. Importing this module never downloads a model or trains."""

import argparse
from functools import lru_cache
from pathlib import Path
from PIL import ImageOps
from src.prescription.data import ROOT, load_config, project_path
from src.prescription.ocr import TrOCRReader
from src.prescription.pipeline import process_image
from src.prescription.knowledge import MedicineKnowledge


@lru_cache(maxsize=2)
def get_reader(model, revision):
    """Reuse loaded weights instead of loading a large model on every click."""
    return TrOCRReader(model, revision=revision)


def transcribe(image, mode, template, model, revision):
    if image is None:
        raise ValueError("Upload or photograph a word crop or prescription first.")
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.width * image.height > 30_000_000:
        raise ValueError("Resize images larger than 30 megapixels.")
    model_reader = get_reader(model, revision)
    return process_image(image, model_reader, mode, template)


def make_theme(gr):
    """Appearance only; these settings do not affect OCR or medicine lookup."""
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.slate,
        font=["Segoe UI", "Arial", "sans-serif"],
    )
    palette = dict(
        body_background_fill="#f5f7fa",
        body_text_color="#152438",
        body_text_color_subdued="#62758b",
        background_fill_primary="#ffffff",
        background_fill_secondary="#f5f7fa",
        block_background_fill="#ffffff",
        block_border_color="#d5dee8",
        block_label_text_color="#152438",
        block_label_background_fill="transparent",
        block_title_text_color="#152438",
        input_background_fill="#ffffff",
        input_border_color="#c5d1df",
        input_placeholder_color="#71849a",
        button_primary_background_fill="#087f8c",
        button_primary_background_fill_hover="#076772",
        button_primary_text_color="#ffffff",
        button_primary_border_color="#087f8c",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_hover="#edf7f8",
        button_secondary_text_color="#087f8c",
        button_secondary_border_color="#087f8c",
    )
    dark_palette = {}
    for key, value in palette.items():
        dark_palette[key + "_dark"] = value

    theme = theme.set(
        **palette,
        **dark_palette,
        block_shadow="none",
        block_shadow_dark="none",
        block_radius="6px",
        button_large_radius="6px",
        button_large_text_weight="600",
        input_radius="6px",
        layout_gap="16px",
    )
    return theme


def build_ui(model=None):
    """Connect three steps: upload, review the reading, and look up a name."""
    import gradio as gr

    config = load_config()
    if not model and (ROOT / "configs/app.json").exists():
        model = str(project_path(load_config(ROOT / "configs/app.json")["model_path"]))
        if not Path(model).is_dir():
            raise FileNotFoundError(f"Configured trained model is missing: {model}")
    model_id = model or config["model_id"]
    revision = None if model else config["model_revision"]
    from app.presentation import (
        EMPTY_READING,
        EMPTY_MATCHES,
        render_readings,
        render_matches,
    )

    # Button callbacks run only after the user interacts with the page.
    def run(image, mode, template, progress=gr.Progress()):
        try:
            progress(0.1, desc="Loading reader and checking image...")
            result = transcribe(image, mode, template, model_id, revision)
            progress(1, desc="Reading ready for review")
            return render_readings(result), result
        except Exception as error:
            raise gr.Error(str(error)) from error

    def lookup(name):
        try:
            result = MedicineKnowledge().lookup(name)
            choices = []
            for candidate in result["candidates"]:
                for label in candidate["labels"]:
                    choice = (label["title"], label["setid"])
                    if choice not in choices:
                        choices.append(choice)
            return (
                render_matches(result),
                result,
                gr.update(choices=choices, value=None),
                "",
            )
        except Exception as error:
            raise gr.Error(str(error)) from error

    def uses(name, setid):
        if not setid:
            raise gr.Error("Select the matching product label first.")
        try:
            result = MedicineKnowledge().indications(name, setid)
            published_uses = result["uses_as_published"]
            if not published_uses:
                published_uses = ["No uses section available."]

            explanation = "\n\n".join(published_uses)
            explanation += "\n\nSource: " + result["label"]["url"]
            explanation += "\n" + result["note"]
            return explanation
        except Exception as error:
            raise gr.Error(str(error)) from error

    theme = make_theme(gr)
    css = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")

    def heading(number, label):
        return f'<div class="step-heading"><span>{number}</span><h2>{label}</h2></div>'

    # Layout: each component below becomes an input or output on the page.
    with gr.Blocks(
        title="Prescription reader", theme=theme, css=css, analytics_enabled=False
    ) as ui:
        gr.HTML(
            '<div class="brand-row"><strong>Rx / Reader</strong><span>Prescription workspace</span></div>',
            elem_id="brand",
        )
        gr.HTML(
            '<div class="hero"><h1>Read. Review. Understand.</h1><p>Turn handwritten text into a reading you can check.</p></div>'
        )
        gr.HTML(
            '<div class="notice">Always check the original. This reader can misread medicine names.</div>'
        )
        with gr.Row(elem_id="workspace"):
            with gr.Column(scale=10, min_width=300):
                gr.HTML(heading("01", "Add an image"))
                with gr.Column(elem_classes="workspace-panel"):
                    image = gr.Image(
                        type="pil",
                        sources=["upload", "webcam"],
                        label="Prescription or word image",
                        show_label=False,
                        height=235,
                        elem_id="image-input",
                        show_download_button=False,
                        placeholder="Drop an image here or click to upload",
                    )
                    mode = gr.Radio(
                        [("Word crop", "word"), ("Full page", "page")],
                        value="word",
                        label="Image type",
                        show_label=False,
                        elem_id="image-mode",
                    )
                    with gr.Accordion(
                        "Page regions (optional)",
                        open=False,
                        elem_id="template-accordion",
                    ):
                        template = gr.File(
                            file_types=[".json"],
                            type="filepath",
                            label="Region template",
                        )
                        gr.Markdown(
                            "For a known page layout, supply one medicine/instruction line per region."
                        )
                    scan = gr.Button("Read handwriting", variant="primary")
                    gr.HTML(
                        '<p class="helper">Start with one clearly cropped medicine name.</p>'
                    )
            with gr.Column(scale=11, min_width=300):
                gr.HTML(heading("02", "Review the reading"))
                with gr.Column(elem_classes="workspace-panel", elem_id="review-panel"):
                    reading = gr.HTML(EMPTY_READING, elem_id="reading-result")
                    with gr.Accordion(
                        "Alternative readings & details",
                        open=False,
                        elem_id="reading-details",
                    ):
                        gr.Markdown(
                            "Readings are suggestions, not verified medicine identities. Scores are not confidence percentages."
                        )
                        report = gr.JSON(label="Reading details")
        with gr.Column(elem_id="lookup-section"):
            gr.HTML(heading("03", "Confirm & look up"))
            with gr.Row(elem_id="lookup-columns"):
                with gr.Column(min_width=260):
                    name = gr.Textbox(
                        label="Medicine spelling you have checked",
                        placeholder="Enter the spelling from the original image",
                    )
                    search = gr.Button("Look up medicine")
                with gr.Column(min_width=260):
                    product = gr.Dropdown(
                        label="Matching product label",
                        choices=[],
                        value=None,
                        elem_id="product-select",
                    )
                    show = gr.Button("Show published uses")
            matches = gr.HTML(EMPTY_MATCHES)
            with gr.Accordion("Source details", open=False):
                match_report = gr.JSON(label="RxNorm and DailyMed matches")
            info = gr.Textbox(
                label="Published uses",
                placeholder="Select a matching product label, then choose Show published uses.",
                lines=3,
                max_lines=12,
                interactive=False,
            )
        gr.HTML(
            "Images are processed locally. Confirmed medicine names are sent to NLM. Regional brands may have no match.",
            elem_id="app-footer",
        )
        # The input/output order here must match each callback.
        scan.click(run, [image, mode, template], [reading, report], concurrency_limit=1)
        search.click(lookup, [name], [matches, match_report, product, info])
        show.click(uses, [name, product], [info])

        def reset_lookup():
            return EMPTY_MATCHES, None, gr.update(choices=[], value=None), ""

        name.input(
            reset_lookup, outputs=[matches, match_report, product, info], queue=False
        )

        def clear_uses():
            return ""

        product.input(clear_uses, outputs=info, queue=False)

        def reset_image():
            return (
                EMPTY_READING,
                None,
                "",
                EMPTY_MATCHES,
                None,
                gr.update(choices=[], value=None),
                "",
            )

        for event in (image.input, mode.input, template.change):
            event(
                reset_image,
                outputs=[reading, report, name, matches, match_report, product, info],
                queue=False,
            )
    return ui


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--server_port", type=int, default=7860)
    args = parser.parse_args()
    build_ui(args.model).launch(
        server_name="127.0.0.1", server_port=args.server_port, share=False
    )


if __name__ == "__main__":
    main()
