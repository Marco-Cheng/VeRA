from __future__ import annotations
from typing import Optional
import re, random

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def extract_numeric_answer(text: str) -> Optional[float]:
    m = re.search(r"####\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    if m: 
        try: return float(m.group(1))
        except: pass
    nums = list(_NUM_RE.finditer(text))
    if not nums: return None
    try:
        return float(nums[-1].group(0))
    except Exception:
        return None

def set_seed(seed: int):
    random.seed(seed)
