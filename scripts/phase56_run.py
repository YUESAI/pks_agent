#!/usr/bin/env python3
"""Verified CLI entrypoint for TRAIN-only Phase56 remote-to-onsite audit."""
import argparse
import json
from pathlib import Path
from scripts.phase54_masked_sequences import load_train
from scripts.phase56_remote_to_site_wm import run

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",required=True,type=Path)
    parser.add_argument("--manifest-db",required=True,type=Path)
    parser.add_argument("--report-dir",required=True,type=Path)
    parser.add_argument("--epochs",type=int,default=25)
    parser.add_argument("--width",type=int,default=64)
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    opts=parser.parse_args()
    records,meta=load_train(opts.data_root,opts.manifest_db)
    summary=run(records,opts.report_dir,epochs=opts.epochs,width=opts.width,
                folds=opts.folds,device_name=opts.device)
    print(json.dumps({"data_capacity":meta,"phase56":summary},indent=2))
    print("Export aggregate reports only; do not upload clinical records.")

if __name__=="__main__":
    main()
