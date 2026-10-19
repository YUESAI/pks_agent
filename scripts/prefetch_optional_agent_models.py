#!/usr/bin/env python3
"""LOGIN-NODE ONLY: prefetch optional text LLM / VLM to /scratch/ylong66.

Phase45 CPU proof-of-mechanism needs NEITHER model. This downloader is for
future experiments after deciding that language/multimodal agents help.
Do not run on network-isolated compute nodes.
"""
import argparse,json,os
from datetime import datetime,timezone
from pathlib import Path

ALLOWED={
    "text":"Qwen/Qwen3-0.6B",
    "doctor":"Qwen/Qwen3-4B-Instruct-2507",
    "vlm":"Qwen/Qwen2.5-VL-3B-Instruct",
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind",choices=tuple(ALLOWED),required=True)
    parser.add_argument("--scratch-root",type=Path,
                        default=Path("/scratch/ylong66"))
    parser.add_argument("--revision",default="main",
                        help="Prefer an immutable model commit SHA for published runs")
    args=parser.parse_args()
    root=args.scratch_root.resolve()
    if (root != Path("/scratch/ylong66") and
            Path("/scratch/ylong66") not in root.parents):
        parser.error("Model files must remain under /scratch/ylong66")
    os.environ["HF_HOME"]=str(root/"hf_cache")
    os.environ["HF_HUB_CACHE"]=str(root/"hf_cache"/"hub")
    from huggingface_hub import snapshot_download
    model=ALLOWED[args.kind]
    dest=root/"models"/model.replace("/","--")
    dest.mkdir(parents=True,exist_ok=True)
    downloaded=snapshot_download(
        repo_id=model,revision=args.revision,
        local_dir=str(dest),
        allow_patterns=["*.safetensors","*.json","*.model",
                        "*.txt","*.tiktoken","*.jinja","*.py"],
        ignore_patterns=["*.gguf","*.onnx","*.pt","*.bin"]
    )
    manifest={
        "repo":model,"revision_requested":args.revision,
        "absolute_dir":str(Path(downloaded).resolve()),
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "purpose":"offline Qwen doctor backbone; Phase45 CPU experiment does not use it",
    }
    (dest/"pks_model_provenance.json").write_text(
        json.dumps(manifest,indent=2)+chr(10))
    print(json.dumps(manifest,indent=2))
    print("Compute node load path:",dest)
    print("Use local_files_only=True and HF_HUB_OFFLINE=1.")

if __name__=="__main__":main()
