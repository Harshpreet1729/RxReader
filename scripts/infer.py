"""Read a word crop or prescription page; output candidates for review."""
import argparse,json
from pathlib import Path
from src.prescription.data import load_config,load_image
from src.prescription.ocr import TrOCRReader
from src.prescription.pipeline import process_image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--mode',choices=['word','page'],default='word')
    p.add_argument('--regions',type=Path,help='Explicit per-line boxes; bypass page detector')
    p.add_argument('--model')
    p.add_argument('--config',type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--output',type=Path,default=Path('outputs/prescription.json'))
    a=p.parse_args();cfg=load_config(a.config)
    reader=TrOCRReader(a.model or cfg['model_id'],a.device,revision=None if a.model else cfg['model_revision'],max_new_tokens=cfg['max_target_length'])
    result=process_image(load_image(a.image),reader,a.mode,a.regions)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
