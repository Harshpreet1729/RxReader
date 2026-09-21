"""Build a self-contained Colab notebook from the current reviewed training code."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def build():
    names = ['scripts/__init__.py','scripts/train.py','scripts/evaluate.py','scripts/prepare_rxhandbd.py',
             'src/__init__.py','src/prescription/__init__.py','src/prescription/data.py',
             'src/prescription/ocr.py','src/prescription/metrics.py',
             'src/prescription/progress.py','src/prescription/training_progress.py',
             'src/prescription/notebook_runner.py','src/prescription/layout.py',
             'src/prescription/extraction.py','src/prescription/pipeline.py','src/prescription/knowledge.py',
             'scripts/infer.py','scripts/lookup_medicine.py','examples/regions.json',
             'configs/colab_base.json','configs/pipeline.json',
             'data/rxhandbd/SOURCE.json','data/rxhandbd/publisher_metadata.json']
    # The dataset itself comes from the checksummed public publisher download.
    payload = io.BytesIO()
    with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = ROOT/name
            data = b'' if not path.exists() and name.endswith('__init__.py') else path.read_bytes()
            archive.writestr(name,data)
    encoded = base64.b64encode(payload.getvalue()).decode()
    digest = hashlib.sha256(payload.getvalue()).hexdigest()
    hashes = json.loads((ROOT/'data/rxhandbd/split_sha256.json').read_text())
    cells=[]
    def md(s): cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
    def code(s): cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(True),'execution_count':None,'outputs':[]})
    md('''# Train TrOCR-base on RxHandBD
Version 2: live progress, streaming logs, and cancellation cleanup.
Training was explicitly requested on 2026-09-21. **Run all starts real GPU training** after GPU selection and Drive authorization. No payment or subscription is requested.

1. Open this notebook in Google Colab (File > Upload notebook).
2. Runtime > Change runtime type > select a GPU (T4 is a reasonable starting point).
3. Runtime > Run all; authorize your own Google Drive when prompted.

Code and provenance are embedded; images are downloaded directly from the checksummed public source. No project ZIP upload or Hugging Face token is needed. Checkpoints and reports are saved to your Drive. Allow several GB of free Drive space (roughly 12 GB headroom recommended, actual usage varies). Free GPU availability/session length is not guaranteed.

We choose TrOCR-base-handwritten as a larger handwriting-specialized candidate. Better results must be established by validation, not assumed from model size. The 444-image validation comparison includes the existing pretrained small baseline and pretrained base. It is NOT a comparison against a separately fine-tuned small model. The 1,111-image test split stays untouched unless explicitly enabled at the end.

**What this project needs:** (1) medicine handwriting recognition trained on word crops, (2) separate page regions, (3) review of uncertain readings, (4) literal strength/schedule extraction, (5) source-backed generic/uses lookup. This notebook covers training, comparison, export and optional inference/lookup. A word-crop dataset does not validate complete prescription understanding.

**Progress:** downloads show bytes, verification shows image counts, training shows optimizer steps plus gradient batches, validation shows batch counts, and every subprocess streams its output into a live log. Unknown-length model loads and checkpoint writes show an indeterminate bar and elapsed time; this is not a fabricated percentage. A silent-process warning is diagnostic, not proof of a hang.

**Stopped old run:** the default v2 run directory preserves v1 checkpoints. Do not delete or reset the runtime until you have checked for saved work. If an old detached training process still consumes GPU memory, use Colab runtime management to end that session before starting this notebook; interrupting v2 normally also stops its child process group.

Sources: [Microsoft TrOCR](https://github.com/microsoft/unilm/tree/master/trocr), [base checkpoint](https://huggingface.co/microsoft/trocr-base-handwritten), [RxHandBD](https://zenodo.org/records/18478741), [Colab limits](https://research.google.com/colaboratory/faq.html).
''')
    code((ROOT/'src/prescription/notebook_runner.py').read_text() + '\n' + '''import sys, subprocess, os, json, hashlib, base64, io, zipfile
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'Select Runtime > Change runtime type > GPU, then rerun.'
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
assert tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) >= (2, 6), 'Use a current Colab PyTorch runtime (2.6+).'
run_live([sys.executable, '-u', '-m', 'pip', 'install', 'transformers==4.57.6', 'accelerate==1.15.0', 'sentencepiece==0.2.2', 'Pillow>=11,<13', 'requests>=2.32,<3', 'protobuf>=4,<8'], title='1. Installing dependencies', log_path='/content/rx_colab_install.log')
# Keep the CUDA-enabled torch/torchvision pair supplied by Colab.
''')
    code('''from google.colab import drive
drive.mount('/content/drive')
RUN_NAME = 'trocr-base-rxhandbd-v2'  # Keep this to resume; change for a separate experiment.
assert RUN_NAME and Path(RUN_NAME).name == RUN_NAME and RUN_NAME not in ('.','..')
RUN_DIR = Path('/content/drive/MyDrive/medical_AI') / RUN_NAME
RUN_DIR.mkdir(parents=True, exist_ok=True)
PROJECT = Path('/content/medical_AI_colab')
PROJECT.mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT)
''')
    code(f'''# Embedded, reviewed project snapshot; contains no credentials, images, or old checkpoints.
BUNDLE = {encoded!r}
BUNDLE_SHA256 = {digest!r}
raw = base64.b64decode(BUNDLE)
assert hashlib.sha256(raw).hexdigest() == BUNDLE_SHA256
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for item in archive.infolist():
        target = (PROJECT / item.filename).resolve()
        assert target.is_relative_to(PROJECT.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(item))
# Provenance for exactly which code was used in this run.
receipt = RUN_DIR / 'bundle_sha256.txt'
if receipt.exists():
    assert receipt.read_text() == BUNDLE_SHA256, 'Code differs from existing run; use a new RUN_NAME.'
else:
    receipt.write_text(BUNDLE_SHA256)
def run(*args):
    return run_live([sys.executable, '-u', '-m', *map(str,args)], cwd=PROJECT, title=str(args[0]), log_path=RUN_DIR/'live.log')
''')
    code(f'''import requests
from tqdm.auto import tqdm
source = json.loads((PROJECT/'data/rxhandbd/SOURCE.json').read_text())
metadata = PROJECT/'publisher_metadata.json'
metadata.write_bytes((PROJECT/'data/rxhandbd/publisher_metadata.json').read_bytes())
DATA = PROJECT/'prepared/RxHandBD'
archive_path = PROJECT/'RxHandBD.zip'
if not DATA.exists():
    if not archive_path.exists() or hashlib.sha256(archive_path.read_bytes()).hexdigest() != source['archive_sha256']:
        temp = archive_path.with_suffix('.part')
        with requests.get(source['download_url'],stream=True,timeout=(15,120)) as response:
            response.raise_for_status()
            with temp.open('wb') as stream, tqdm(total=int(response.headers.get('content-length',0)) or None, unit='B', unit_scale=True, desc='Downloading RxHandBD') as bar:
                for chunk in response.iter_content(1024*1024):
                    stream.write(chunk)
                    bar.update(len(chunk))
        assert hashlib.sha256(temp.read_bytes()).hexdigest() == source['archive_sha256'], 'Download checksum mismatch; rerun.'
        temp.replace(archive_path)
    # Prepare into staging: interrupted downloads/preparation never look complete.
    import tempfile
    with tempfile.TemporaryDirectory(dir=PROJECT) as staging:
        prepared = Path(staging)/'RxHandBD'
        run('scripts.prepare_rxhandbd','--archive',archive_path,'--metadata',metadata,'--output',prepared)
        DATA.parent.mkdir(parents=True,exist_ok=True)
        prepared.rename(DATA)
run('scripts.prepare_rxhandbd','--verify','--output',DATA)
expected = {hashes!r}
for split, expected_hash in expected.items():
    assert hashlib.sha256((DATA/f'{{split}}.csv').read_bytes()).hexdigest() == expected_hash, f'{{split}} differs from reviewed local split'
print('All source images verified; frozen train/validation/test splits match the local project.')
''')
    code('''cfg = json.loads((PROJECT/'configs/colab_base.json').read_text())
cfg['dataset_root'] = str(DATA)
cfg['output_dir'] = str(RUN_DIR/'checkpoints')
CONFIG = PROJECT/'configs/colab_runtime.json'
CONFIG.write_text(json.dumps(cfg,indent=2))
# Batch 1 + accumulation 16 + fp16 + gradient checkpointing limit memory demand.
# Up to 12 epochs, eval/save every 100 optimizer steps, early stop after 5 unimproved evaluations.
run('scripts.train','--config',CONFIG,'--report',RUN_DIR/'preflight.json')
''')
    code('''# Full validation baselines. Each runs in a subprocess, releasing GPU memory afterwards.
for label, config_path in [('small_pretrained',PROJECT/'configs/pipeline.json'),('base_pretrained',CONFIG)]:
    report = RUN_DIR/f'{label}_validation.json'
    if not report.exists():
        baseline_cfg = json.loads(config_path.read_text())
        baseline_cfg['dataset_root'] = str(DATA)
        baseline_cfg['max_target_length'] = cfg['max_target_length']
        path = PROJECT/f'{label}.json'
        path.write_text(json.dumps(baseline_cfg))
        run('scripts.evaluate','--config',path,'--device','cuda','--output',report)
''')
    code('''# This cell STARTS TRAINING. No additional RUN_TRAINING switch is required.
output = Path(cfg['output_dir'])
if (output/'training_complete.json').exists():
    print('This run already completed; proceeding to validation. Change RUN_NAME for a new run.')
else:
    candidates = [p for p in output.glob('checkpoint-*') if (p/'checkpoint_complete.json').exists() and (p/'trainer_state.json').exists()] if output.exists() else []
    resume = max(candidates,key=lambda p:int(p.name.split('-')[-1])) if candidates else None
    if output.exists() and any(output.iterdir()) and resume is None:
        raise RuntimeError('Interrupted before first complete checkpoint. Keep these files; choose a new RUN_NAME to restart.')
    args = ['scripts.train','--config',CONFIG,'--execute']
    if resume:
        print('Resuming',resume)
        args += ['--resume',resume]
    run(*args)
''')
    code('''trained_report = RUN_DIR/'base_finetuned_validation.json'
run('scripts.evaluate','--config',CONFIG,'--model',output/'best','--device','cuda','--output',trained_report)
reports = {name:json.loads((RUN_DIR/f'{name}_validation.json').read_text())['metrics'] for name in ['small_pretrained','base_pretrained','base_finetuned']}
for name, metric in reports.items():
    print(name, json.dumps(metric,indent=2))
improved = reports['base_finetuned']['normalized_cer'] < reports['small_pretrained']['normalized_cer']
comparison = {'metric':'normalized_cer','lower_is_better':True,'improved_vs_existing_pretrained_small':improved,'validation':reports,'test_used':False}
(RUN_DIR/'comparison.json').write_text(json.dumps(comparison,indent=2))
print('Measured improvement over existing pretrained small:',improved)
print('Saved model + processor:',output/'best')
print('Keep this folder together for local inference; reports and checkpoints are in:',RUN_DIR)
''')
    md('''## Optional final test, after model selection
Leave disabled while comparing models/hyperparameters. The held-out test is not used to choose a model. To use the result locally, copy the entire `checkpoints/best` folder (including processor/tokenizer) from Drive into your project and launch `python -m app.gradio_app --model path/to/best`.

If Colab disconnects: reconnect to a GPU and rerun all cells with the same RUN_NAME. Completed checkpoints resume; a partially written checkpoint is ignored. Progress since the last checkpoint can be lost. Do not run the same RUN_NAME simultaneously in two runtimes.
''')
    code('''RUN_FINAL_TEST = False
if RUN_FINAL_TEST:
    run('scripts.evaluate','--config',CONFIG,'--model',output/'best','--split','test','--device','cuda','--output',RUN_DIR/'final_test.json')
else:
    print('Held-out test remains untouched.')
''')
    md("""## Try the trained reader (optional)
Upload a medicine word crop or a page with a matching regions JSON. Review every OCR result. Automatic page detection requires the separate optional Paddle dependencies; manual regions avoid extra installs. Nothing uploads until you enable the cell. General uses lookup below sends only the spelling you explicitly enter to NLM.
""")
    code('''TRY_IMAGE = False
if TRY_IMAGE:
    from google.colab import files
    from PIL import Image
    from IPython.display import display
    uploaded = files.upload()
    image_paths = [Path(name) for name in uploaded if Path(name).suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]
    region_paths = [Path(name) for name in uploaded if Path(name).suffix.lower()=='.json']
    assert len(image_paths)==1 and len(region_paths)<=1, 'Upload one image and optionally one regions JSON.'
    display(Image.open(image_paths[0]))
    args = ['scripts.infer','--image',image_paths[0],'--model',output/'best','--device','cuda','--output',RUN_DIR/'sample_reading.json']
    if region_paths:args += ['--mode','page','--regions',region_paths[0]]
    run(*args)
else:
    print('Optional image upload disabled.')
''')
    code('''CONFIRMED_MEDICINE = ''  # Enter spelling only after checking the original image.
SELECTED_LABEL_ID = ''   # Choose a matching label ID from the first lookup result.
if CONFIRMED_MEDICINE.strip():
    args = ['scripts.lookup_medicine','--confirmed-name',CONFIRMED_MEDICINE]
    if SELECTED_LABEL_ID:args += ['--setid',SELECTED_LABEL_ID]
    run(*args)
else:
    print('No medicine name entered; no external lookup sent.')
''')
    md("""## Download the trained model
The full model and processor are already in Drive under this run's checkpoints/best directory. This optional cell exports a ZIP to your computer. Keep all processor/tokenizer files. Launch the existing local app with `python -m app.gradio_app --model path/to/best`.
""")
    code('''DOWNLOAD_MODEL = False
if DOWNLOAD_MODEL:
    import shutil
    from google.colab import files
    assert (output/'training_complete.json').exists(), 'Training has not completed.'
    bundle = shutil.make_archive('/content/RxHandBD_TrOCR_base_best','zip',root_dir=output/'best')
    files.download(bundle)
else:
    print('Model export optional; checkpoints remain in Drive.')
''')
    titles = iter(['GPU and live dependency setup','Persistent Drive storage','Load the project snapshot','Download and verify RxHandBD','Training configuration and preflight','Measure pretrained baselines','Fine-tune with live progress','Compare validation results'])
    expanded=[]
    for c in cells:
        if c['cell_type']=='code':
            title=next(titles,None)
            if title:expanded.append({'cell_type':'markdown','metadata':{},'source':['## '+title+'\n']})
        expanded.append(c)
    cells=expanded
    for i,c in enumerate(cells):c['id']=f'colab-base-{i:02d}'
    nb={'cells':cells,'metadata':{'accelerator':'GPU','colab':{'name':'RxHandBD_TrOCR_Base_Colab.ipynb','provenance':[]},'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    target=ROOT/'notebooks/RxHandBD_TrOCR_Base_Colab.ipynb'
    target.write_text(json.dumps(nb,indent=2)+'\n',encoding='utf-8')
    print(target, target.stat().st_size, 'bytes')
    return target

if __name__=='__main__':build()
