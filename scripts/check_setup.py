"""Check active dependencies and model/data configuration without downloading weights."""
import importlib.metadata as metadata
import json
from src.prescription.data import load_config,project_path,read_split


def main():
    packages={}
    for name in ['torch','torchvision','transformers','accelerate','sentencepiece','Pillow','gradio','paddleocr','paddlepaddle']:
        try:packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError:packages[name]=None
    cfg=load_config()
    counts={s:len(read_split(project_path(cfg['dataset_root']),s)) for s in ['train','validation','test']}
    print(json.dumps({'dependencies':packages,'splits':counts,'model':cfg['model_id'],'training_started':False},indent=2))

if __name__=='__main__':main()
