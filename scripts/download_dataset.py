"""Download and reproduce the pinned RxHandBD splits; existing data is only verified."""
import hashlib
import json
from pathlib import Path
import tempfile
import requests
from src.prescription.data import ROOT
from scripts.prepare_rxhandbd import prepare, verify


def main():
    metadata=ROOT/'data/rxhandbd'
    source=json.loads((metadata/'SOURCE.json').read_text())
    target=ROOT/'Dataset/RxHandBD'
    if not target.exists():
        cache=ROOT/'.cache/datasets';cache.mkdir(parents=True,exist_ok=True)
        archive=cache/'RxHandBD.zip'
        if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest()!=source['archive_sha256']:
            partial=archive.with_suffix('.part');digest=hashlib.sha256();downloaded=0
            print('Downloading pinned RxHandBD archive...',flush=True)
            with requests.get(source['download_url'],stream=True,timeout=(15,120)) as response:
                response.raise_for_status()
                with partial.open('wb') as stream:
                    for chunk in response.iter_content(1024*1024):
                        stream.write(chunk);digest.update(chunk);downloaded+=len(chunk)
                        print(f'\rDownloaded {downloaded/2**20:.1f} MiB',end='',flush=True)
            print()
            if digest.hexdigest()!=source['archive_sha256']:raise ValueError('Archive SHA256 mismatch')
            partial.replace(archive)
        target.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='rxhandbd-',dir=target.parent) as staging:
            prepared=Path(staging)/'RxHandBD'
            prepare(archive,prepared,metadata/'publisher_metadata.json')
            prepared.rename(target)
    report=verify(target)
    for split,expected in json.loads((metadata/'split_sha256.json').read_text()).items():
        if hashlib.sha256((target/f'{split}.csv').read_bytes()).hexdigest()!=expected:
            raise ValueError(f'{split} differs from pinned preparation')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
