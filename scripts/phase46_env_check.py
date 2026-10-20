#!/usr/bin/env python3
"""Fail-fast offline environment diagnostics for Phase46 Qwen on GPU nodes."""
import argparse
import importlib
import os
import platform
import sys
from pathlib import Path


def check(mode):
    problems=[]
    base=Path(sys.prefix).resolve()
    expected=Path("/scratch/ylong66/conda/envs/pks_qwen46")
    if base!=expected:
        problems.append("Python is not in isolated pks_qwen46 environment: "+str(base))
    packages={}
    for name in ("torch","transformers","safetensors","accelerate","tokenizers","huggingface_hub"):
        try:
            module=importlib.import_module(name)
            packages[name]=getattr(module,"__version__","unknown")
        except Exception as e:
            problems.append(name+" import failed: "+str(e))
    if packages.get("torch") and not packages["torch"].startswith("2.5.1"):
        problems.append("Expected PyTorch 2.5.1; found "+packages["torch"])
    if packages.get("transformers")!="4.51.3":
        problems.append("Expected transformers 4.51.3; found "+str(packages.get("transformers")))
    if packages.get("safetensors") and not packages["safetensors"].startswith("0.5.3"):
        problems.append("Expected safetensors 0.5.3; found "+packages["safetensors"])
    print("Python:",sys.executable)
    print("Host:",platform.node(),"glibc:",platform.libc_ver())
    for name,version in sorted(packages.items()):
        print(name+":",version)
    if mode=="gpu":
        if os.environ.get("HF_HUB_OFFLINE")!="1":
            problems.append("GPU run must set HF_HUB_OFFLINE=1")
        if os.environ.get("TRANSFORMERS_OFFLINE")!="1":
            problems.append("GPU run must set TRANSFORMERS_OFFLINE=1")
        try:
            import torch
            if not torch.cuda.is_available():
                problems.append("CUDA is not available")
            else:
                print("CUDA runtime:",torch.version.cuda)
                print("CUDA device:",torch.cuda.get_device_name(0))
                if torch.cuda.get_device_properties(0).total_memory < 14*1024**3:
                    problems.append("GPU has <14GiB, 4B model may not fit")
        except Exception as e:
            problems.append("CUDA check failed: "+str(e))
        model=Path("/scratch/ylong66/models/Qwen--Qwen3-4B-Instruct-2507")
        if not (model/"config.json").is_file() or not list(model.glob("*.safetensors")):
            problems.append("Offline Qwen3 Doctor weights incomplete: "+str(model))
    for issue in problems:
        print("ERROR:",issue,file=sys.stderr)
    if problems:
        raise SystemExit(2)
    print("Phase46",mode,"environment preflight OK")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=("packages","gpu"),required=True)
    args=p.parse_args()
    check(args.mode)


if __name__=="__main__":main()
