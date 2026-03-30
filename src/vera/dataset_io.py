from __future__ import annotations
from typing import List, Dict, Any
import json
from fractions import Fraction
from decimal import Decimal


def _json_default(obj: Any):
    if isinstance(obj, Fraction):
        if obj.denominator == 1:
            return obj.numerator
        return float(obj)
    if isinstance(obj, Decimal):
        return float(obj)
    return str(obj)

def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")

def read_gsm8k_jsonl(path: str) -> List[Dict]:
    rows = load_jsonl(path)
    out = []
    for i, r in enumerate(rows):
        q = r.get("question") or r.get("query") or r.get("input") or ""
        a = r.get("answer") or r.get("output") or r.get("label") or ""
        rid = r.get("id") or f"gsm8k-{i:06d}"
        out.append({"id": rid, "question": q, "answer": a, "raw": r})
    return out
