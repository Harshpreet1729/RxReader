"""Evaluate inference only; never trains or changes source labels."""
import argparse,json,time
from pathlib import Path
from src.prescription.data import load_config, project_path, read_split, load_image
from src.prescription.metrics import metrics
from src.prescription.ocr import TrOCRReader
from src.prescription.progress import emit, progress_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path)
    p.add_argument('--model',help='Saved best checkpoint directory or HF model ID')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--limit',type=int,default=0,help='0: full split; positive: smoke subset, not benchmark')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--output',type=Path,default=Path('outputs/evaluation.json'))
    a=p.parse_args()
    if a.limit<0:p.error('--limit must be nonnegative')
    cfg=load_config(a.config); rows=read_split(project_path(cfg['dataset_root']),a.split)
    total=len(rows); rows=rows[:a.limit] if a.limit else rows
    emit('Loading evaluation model')
    reader=TrOCRReader(a.model or cfg['model_id'],a.device,revision=None if a.model else cfg['model_revision'],max_new_tokens=cfg['max_target_length'])
    results=[]; start=time.perf_counter()
    for row in progress_rows(rows, f'Evaluating {a.split}', every=1):
        prediction=reader.read(load_image(row['image']))
        results.append({'filename':row['filename'],'reference':row['text'],**prediction})
    report={'split':a.split,'full_split_size':total,'subset_only':len(rows)!=total,'training_performed':False,
            'model':reader.model_id,'revision':reader.revision,'seconds':time.perf_counter()-start,
            'metrics':metrics([r['reference'] for r in results],[r['text'] for r in results]),'predictions':results}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='predictions'},indent=2))

if __name__=='__main__': main()
