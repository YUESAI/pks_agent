#!/usr/bin/env python3
"""Read-only experiment duplication preflight for pks_agent.

Examples:
 python scripts/experiment_preflight.py --query "Qwen abstain"
 python scripts/experiment_preflight.py --mechanism independent-expert-gate --split global80-20-dev
 python scripts/experiment_preflight.py --phase 90
 python scripts/experiment_preflight.py --mechanism partial33-likelihood --split global80-20-dev --fail-on-match

CLI exits 2 with --fail-on-match for an exact precedent or matching design
family. It does not upload patient records, touch original TEST, or run jobs.
"""
from __future__ import annotations
import argparse,json,re,sys
from pathlib import Path

DEFAULT=(Path(__file__).resolve().parents[1]/
         "docs"/"experiment_registry_phase60_90.json")


def load_registry(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data.get("records"),list):
        raise ValueError("Invalid experiment registry: missing records list")
    ids=[int(x["phase"]) for x in data["records"]]
    if len(set(ids))!=len(ids):
        raise ValueError("Duplicate phase entries: "+repr(ids))
    required=("phase","experiment_id","title","status",
              "mechanism","split_protocol","source_doc","verdict")
    for row in data["records"]:
        if any(not str(row.get(field,"")).strip() for field in required):
            raise ValueError("Incomplete registry entry: "+repr(row))
    return data


def _norm(x):
    return " ".join(re.findall(r"[a-z0-9\u4e00-\u9fff]+",
                            str(x).lower()))


def matching_records(data,phase=None,query=None,mechanism=None,split=None):
    matches=[]
    for row in data["records"]:
        if phase is not None and int(row["phase"])!=phase:
            continue
        if mechanism and _norm(row["mechanism"])!=_norm(mechanism):
            continue
        if split and _norm(row["split_protocol"])!=_norm(split):
            continue
        if query:
            text=" ".join(str(row.get(k,"")) for k in (
                "phase","title","verdict","mechanism","source_doc",
                "data_scope","split_protocol"))
            tokens=_norm(query).split()
            if not all(token in _norm(text) for token in tokens):
                continue
        matches.append(row)
    return matches


def cli(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry",type=Path,default=DEFAULT)
    p.add_argument("--phase",type=int)
    p.add_argument("--query",help="All words must appear in historical record")
    p.add_argument("--mechanism",help="Exact historical mechanism family")
    p.add_argument("--split",help="Exact historical split protocol key")
    p.add_argument("--fail-on-match",action="store_true",
                   help="Exit 2 if an existing record matches (pre-submit guard)")
    args=p.parse_args(argv)
    if not any([args.phase is not None,args.query,args.mechanism,args.split]):
        p.error("Specify --phase, --query, --mechanism, or --split")
    data=load_registry(args.registry)
    found=matching_records(data,args.phase,args.query,
                           args.mechanism,args.split)
    if not found:
        print("NO DIRECT MATCH in current historical ledger. "
              "This is NOT proof an experiment is new; inspect protocol details.")
        return 0
    print(f"FOUND {len(found)} documented experiment precedent(s):")
    for row in found:
        print(f"  P{row['phase']} [{row['status']}] {row['title']}")
        print(f"     mechanism={row['mechanism']} split={row['split_protocol']}")
        print(f"     finding={row['verdict']}")
        print(f"     source={row['source_doc']}")
    print("Before resubmitting, identify a NEW hypothesis, endpoint, data "
          "population, split, conditional observation budget, or mechanism.")
    return 2 if args.fail_on_match else 0


if __name__=="__main__":
    sys.exit(cli())
