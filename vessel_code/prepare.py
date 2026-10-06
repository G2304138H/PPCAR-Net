"""Validate annotated CT NPZs and export seven projection masks per case."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from vessel_code.data.ct_volume import CTProjectionDataset

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--artery-type',required=True,choices=['rca','lca'])
    args=p.parse_args();files=sorted(Path(args.input_dir).glob('*.npz'))
    if not files:raise ValueError('No NPZ files found')
    for case,arrays in CTProjectionDataset(files,args.artery_type):
        dest=Path(args.output_dir)/case.path.stem;dest.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(dest/'projections.npz',**arrays)
        for i,mask in enumerate(arrays['images']):Image.fromarray((mask*255).astype('uint8')).save(dest/f'view_{i}.png')
        print(dest)
if __name__=='__main__':main()
