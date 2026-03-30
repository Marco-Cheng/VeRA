# vera/verifier.py
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
import os, sys, math, random, multiprocessing as mp, traceback

# ===== 默认限制（可用环境变量覆盖）=====
TIME_LIMIT_SEC = int(os.getenv("VERA_TIME_SEC", "300"))
MEM_LIMIT_BYTES = int(os.getenv("VERA_MEM_BYTES", str(2 * 1024**3)))

class RNGShim:
    """Drop-in subset of random.Random used by teacher generators."""
    def __init__(self, seed: int):
        self._seed = int(seed)
        self._r = random.Random(self._seed)

    # Core
    def random(self) -> float: return self._r.random()
    def uniform(self, a: float, b: float) -> float: return self._r.uniform(a, b)
    def randint(self, a: int, b: int) -> int: return self._r.randint(a, b)
    def randrange(self, start: int, stop: Optional[int] = None, step: int = 1) -> int:
        return self._r.randrange(start, stop, step)

    # Sequences
    def choice(self, seq): return self._r.choice(seq)
    def choices(self, population, weights=None, cum_weights=None, k: int = 1):
        return self._r.choices(population, weights=weights, cum_weights=cum_weights, k=k)
    def sample(self, population, k: int):
        return self._r.sample(population, k)
    def shuffle(self, x):
        self._r.shuffle(x)
        return x
    def getrandbits(self, k: int) -> int: return self._r.getrandbits(k)

    # Distributions (subset)
    def triangular(self, low=0.0, high=1.0, mode=None):
        return self._r.triangular(low, high, mode)
    def betavariate(self, alpha, beta): return self._r.betavariate(alpha, beta)
    def expovariate(self, lambd): return self._r.expovariate(lambd)
    def gammavariate(self, alpha, beta): return self._r.gammavariate(alpha, beta)
    def gauss(self, mu, sigma): return self._r.gauss(mu, sigma)
    def lognormvariate(self, mu, sigma): return self._r.lognormvariate(mu, sigma)
    def normalvariate(self, mu, sigma): return self._r.normalvariate(mu, sigma)
    def paretovariate(self, alpha): return self._r.paretovariate(alpha)
    def weibullvariate(self, alpha, beta): return self._r.weibullvariate(alpha, beta)

    @property
    def seed(self) -> int:
        return self._seed


# ---------- 顶层：子进程入口（spawn 可 pickle）----------
def _subproc_worker(q: mp.Queue, code: str, fn_name: str, payload: Dict[str, Any],
                    time_limit: int, mem_limit: int) -> None:
    """
    在子进程中 trusted 执行：
      - resource.setrlimit 约束内存/CPU（*nix 有效）
      - 墙钟超时由父进程控制（p.join(timeout)）
    最终把 (ok, result, err_str) 放入队列。
    """
    # 资源限制（类 Unix 有效；Windows 忽略）
    try:
        import resource
        if mem_limit and mem_limit > 0:
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
            except Exception:
                # 某些系统不支持 RLIMIT_AS，忽略
                pass
        if time_limit and time_limit > 0:
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (time_limit, time_limit))
            except Exception:
                pass
    except Exception:
        pass

    try:
        # trusted：提供完整 __builtins__ + 常用模块
        glob: Dict[str, Any] = {"__builtins__": __builtins__, "math": math, "random": random}
        loc: Dict[str, Any] = {}
        exec(code, glob, loc)  # 编译 teacher 代码
        fn = loc.get(fn_name)
        if fn is None or not callable(fn):
            raise ValueError(f"{fn_name} not defined or not callable")

        if fn_name == "generator":
            seed = int(payload["seed"])
            random.seed(seed)
            rng = RNGShim(seed)
            result = fn(rng)  # 期望返回 dict
        else:
            assign = payload["assign"]  # verifier(assign) -> (ok, y)
            result = fn(assign)

        q.put((True, result, None))
    except Exception:
        q.put((False, None, traceback.format_exc()))

# ---------- 父进程封装 ----------
def _run_in_subproc(code: str, fn_name: str, payload: Dict[str, Any],
                    time_limit: int, mem_limit: int) -> Tuple[bool, Any, Optional[str]]:
    q: mp.Queue = mp.Queue(1)
    p = mp.Process(target=_subproc_worker, args=(q, code, fn_name, payload, time_limit, mem_limit))
    p.start()
    p.join(timeout=time_limit if time_limit and time_limit > 0 else None)

    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        return (False, None, f"Timeout: exceeded {time_limit}s wall-clock")

    try:
        ok, res, err = q.get_nowait()
        return (ok, res, err)
    except Exception:
        if p.exitcode != 0:
            return (False, None, f"Subprocess exited with code {p.exitcode}")
        return (False, None, "Unknown subprocess failure")

# ---------- 对外接口（与旧版一致） ----------
def compile_verifier(code: str, time_limit_sec: Optional[int] = None):
    """
    返回 callable: verifier_fn(assign) -> (ok, y)
    每次调用都在受限子进程 trusted 执行。
    """
    def _verifier_fn(assign: Dict[str, Any]) -> Tuple[bool, Any]:
        ok, res, err = _run_in_subproc(
            code=code, fn_name="verifier", payload={"assign": assign},
            time_limit=time_limit_sec if time_limit_sec is not None else TIME_LIMIT_SEC,
            mem_limit=MEM_LIMIT_BYTES
        )
        if not ok:
            if err: sys.stderr.write(f"[verifier error] {err}\n")
            return False, None
        # 期望 (bool, y)；若 teacher 只返回 y，容错视为 (True, y)
        if (isinstance(res, (tuple, list))) and len(res) == 2:
            return bool(res[0]), res[1]
        return True, res
    return _verifier_fn

def run_verifier(verifier_fn, assign: Dict[str, Any]) -> Tuple[bool, Any]:
    return verifier_fn(assign)

def compile_generator(code: str, time_limit_sec: Optional[int] = None):
    """
    返回 callable: gen_fn(rng: RNGShim) -> assign(dict)
    每次调用都在受限子进程 trusted 执行。
    """
    def _gen_fn(rng: RNGShim):
        ok, res, err = _run_in_subproc(
            code=code, fn_name="generator",
            payload={"seed": getattr(rng, "seed", None) or getattr(rng, "_seed", 0)},
            time_limit=time_limit_sec if time_limit_sec is not None else TIME_LIMIT_SEC,
            mem_limit=MEM_LIMIT_BYTES
        )
        if not ok:
            if err: sys.stderr.write(f"[generator error] {err}\n")
            return {}
        if not isinstance(res, dict):
            sys.stderr.write(f"[generator warning] expected dict, got {type(res)}; coercing if possible.\n")
            try:
                res = dict(res)
            except Exception:
                return {}
        return res
    return _gen_fn
