"""Qwen3 Doctor tool adapter: constrained LLM action selection, no patient labels.

All real patient measurements come solely from PatientEvidenceService;
model output is treated as an untrusted proposed tool name. No internet.
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from scripts.phase45_world_model import PatientEvidenceService

SYSTEM = (
    "You are a research simulation Doctor Agent, not a treating clinician. "
    "Choose ONE next MDS-UPDRS Part III item to request from the allowed "
    "list, using only disclosed past observations. Unrequested values, "
    "the six hidden onsite-only scores, and their outcome are UNKNOWN. "
    "You cannot fabricate observations or change the set of allowed tools. "
    'Reply with EXACTLY one JSON object like {"ask":"NP3SPCH"}, '
    "using an item name from the provided list. No prose."
)


def extract_action(raw,allowed):
    """Parse only JSON ask field; reject out-of-vocabulary or repeats."""
    text=str(raw).strip()
    candidates=[text]
    candidates.extend(re.findall(r"\{[^{}]*\}",text,re.S))
    for candidate in candidates:
        try:obj=json.loads(candidate)
        except (ValueError,TypeError):continue
        if not isinstance(obj,dict) or set(obj)!={"ask"}:continue
        selected=obj.get("ask")
        if isinstance(selected,str) and selected in allowed:
            return selected
    return None


def make_messages(remaining,observed,risk,information_gain,mode):
    """The prompt NEVER receives pid, source dates, hidden truth or labels."""
    if mode not in ("llm_plain","llm_wm"):
        raise ValueError("Unknown LLM Doctor mode")
    parts=[
        "Research task: acquire informative observed Motor scores using "
        "a limited tool-call budget. A score of 0..4 is a recorded item "
        "rating, not a diagnosis.",
        "Available tools (only these names may be requested): "+
        json.dumps(sorted(remaining)),
        "Already measured items: "+
        json.dumps(dict(sorted(observed.items())),sort_keys=True),
    ]
    if mode=="llm_wm":
        ordered=sorted(information_gain,
                       key=lambda k:(-information_gain[k],k))
        parts.extend([
            "Observation World Model current probability of hidden-six "
            "nonzero: "+format(risk,".4f"),
            "World Model estimated conditional information gain "
            "(nats) of the top candidates: "+
            json.dumps({name:round(information_gain[name],5)
                        for name in ordered[:10]}),
            "The model's estimates are uncertain; use them as advice, "
            "not measured clinical facts.",
        ])
    else:
        parts.append("No World Model predictions are available to you.")
    parts.append('Choose your next tool. Output only {"ask":"ONE_ALLOWED_NAME"}.')
    return [{"role":"system","content":SYSTEM},
            {"role":"user","content":"\n".join(parts)}]


class TransformersDoctor:
    """One local Qwen model per GPU process; transformers offline inference."""

    def __init__(self,model_dir,max_new_tokens=48):
        model_dir=Path(model_dir).expanduser().resolve()
        if not (model_dir/"config.json").is_file():
            raise FileNotFoundError("Missing offline model config.json: "+str(model_dir))
        if not any(model_dir.glob("*.safetensors")):
            raise FileNotFoundError("Missing downloaded safetensors: "+str(model_dir))
        if not 1<=max_new_tokens<=128:
            raise ValueError("Limit output tokens to 1..128")
        os.environ["HF_HUB_OFFLINE"]="1"
        os.environ["TRANSFORMERS_OFFLINE"]="1"
        os.environ.setdefault("HF_HOME","/scratch/ylong66/hf_cache")
        os.environ.setdefault("XDG_CACHE_HOME","/scratch/ylong66/.cache")
        try:
            import torch
            from transformers import AutoTokenizer,AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("Install torch, transformers>=4.51 and accelerate "
                               "on login node before compute submission") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Phase46 requires CUDA GPU; submit an L40S/H100 job")
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(
            str(model_dir),local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(
            str(model_dir),local_files_only=True,trust_remote_code=False,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported()
                         else torch.float16),
            device_map="auto",low_cpu_mem_usage=True)
        self.model.eval()
        self.max_new_tokens=max_new_tokens
        self.local_dir=str(model_dir)
        self.config_id=getattr(self.model.config,"_name_or_path","")
        self.total_prompts=0
        self.total_generated_tokens=0

    def generate(self,messages):
        torch=self.torch
        # Explicit attention_mask is essential when pad_token == eos_token.
        # Tokenize the rendered chat template as a batch to obtain a
        # reliable attention mask rather than relying on token inference.
        prompt=self.tokenizer.apply_chat_template(
            messages,add_generation_prompt=True,tokenize=False)
        inputs=self.tokenizer(
            prompt,return_tensors="pt",return_attention_mask=True,
            add_special_tokens=False)
        device=next(self.model.parameters()).device
        inputs={name:value.to(device) for name,value in inputs.items()}
        count=int(inputs["input_ids"].shape[-1])
        # Qwen's generation_config.json may contain sampling defaults.
        # Explicitly reset them for reproducible deterministic decoding.
        from copy import deepcopy
        generation_config=deepcopy(self.model.generation_config)
        generation_config.do_sample=False
        generation_config.temperature=1.0
        generation_config.top_p=1.0
        generation_config.top_k=50
        generation_config.pad_token_id=self.tokenizer.eos_token_id
        with torch.inference_mode():
            outputs=self.model.generate(
                **inputs,max_new_tokens=self.max_new_tokens,
                generation_config=generation_config,use_cache=True)
        self.total_prompts+=1
        tokens=outputs[0,count:]
        self.total_generated_tokens+=int(tokens.shape[-1])
        return self.tokenizer.decode(tokens,skip_special_tokens=True)


def llm_doctor_rollout(world,row,backend,items,questions,mode,checkpoints=()):
    """Doctor -> measured tool -> World Model belief update, strictly q steps.

    Never pass hidden outcome to the LLM. An invalid request is counted
    and replaced by a deterministic, admissible max-information tool.
    """
    if not 1<=questions<=len(items):
        raise ValueError("Invalid question budget")
    marks=tuple(sorted(set(checkpoints)))
    if any(not isinstance(q,int) or not 1<=q<=questions for q in marks):
        raise ValueError("Checkpoints must be within the question budget")
    evidence=PatientEvidenceService(row["features"],items,questions)
    snapshots={}
    invalid=0
    requested=0
    selected_items=[]
    greedy_agreements=0
    for step in range(questions):
        remaining=[k for k in items if k not in evidence.observed]
        gains={k:world.information_gain(evidence.observed,k)
               for k in remaining}
        risk=world.posterior(evidence.observed)
        response=backend.generate(
            make_messages(remaining,evidence.observed,risk,gains,mode))
        requested+=1
        chosen=extract_action(response,set(remaining))
        greedy=sorted(remaining,key=lambda k:(-gains[k],k))[0]
        if chosen is None:
            invalid+=1
            chosen=greedy
        greedy_agreements+=int(chosen==greedy)
        selected_items.append(chosen)
        # Only measured evidence is returned; model output cannot write scores.
        reply=evidence.request(chosen)
        if (step+1) in marks:
            snapshots[step+1]=world.posterior(evidence.observed)
        if reply["provenance"]!="observed_V04_masking_simulation":
            raise AssertionError("Ungrounded tool result")
    return {
        "risk":world.posterior(evidence.observed),
        "queries":requested,
        "invalid_action_fallbacks":invalid,
        "observed_count":len(evidence.observed),
        # Used only to compute aggregate behavior diagnostics in memory.
        # Do not export individual question sequences or patient IDs.
        "selected_items":tuple(selected_items),
        "greedy_agreements":greedy_agreements,
        "risk_at_checkpoints":snapshots,
    }
