from __future__ import annotations
from typing import Protocol, Dict, Any
import importlib, inspect, json, re
from .prompt_templates import teacher_messages, student_messages, aime_teacher_messages
from .oracle_llm_io import teacher_llm_call, student_llm_call

class TeacherOracle(Protocol):
    def convert_to_spec(self, question: str, answer: str, example_id: str, n_text_templates: int = 3, feedback: str | None = None,) -> Dict[str, Any]:
        ...

class StudentOracle(Protocol):
    def infer(self, prompt: str, example_id: str) -> str:
        ...

def load_impl(path: str):
    if ":" not in path:
        mod, cls = path, None
    else:
        mod, cls = path.split(":", 1)
    m = importlib.import_module(mod)
    if cls is None:
        return m
    T = getattr(m, cls)
    if inspect.isclass(T):
        return T()
    return T

def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"^```(json)?\s*|```$", "", text.strip(), flags=re.IGNORECASE|re.MULTILINE)
    m = re.search(r"\{[\s\S]*\}\s*$", cleaned)
    s = cleaned if m is None else m.group(0)
    return json.loads(s)

class PromptTeacher:
    def __init__(self, profile: str = "aime"):
        self.profile = profile

    def convert_to_spec(self, question: str, answer: str, example_id: str, n_text_templates: int = 3, feedback: str | None = None) -> Dict[str, Any]:
        if self.profile == "aime":
            year = None
            if "::" in example_id:
                _, maybe_year = example_id.split("::", 1)
                try:
                    year = int(maybe_year)
                except ValueError:
                    year = None
            msgs = aime_teacher_messages(example_id, question, answer, year=year, feedback=feedback)
        else:
            msgs = teacher_messages(question, answer, example_id, n_text_templates=n_text_templates, feedback=feedback)
        raw = teacher_llm_call(msgs)
        data = _extract_json(raw)
        generator_block = data.get("generator") or {}
        verifier_block = data.get("verifier") or data.get("validator") or {}
        for key, block in (("generator", generator_block), ("verifier", verifier_block)):
            if block.get("type") != "python":
                raise ValueError(f"{key}.type must be 'python'")
        data.setdefault("text_templates", data.get("text_templates", []))
        data.setdefault("meta", {})
        data["meta"].setdefault("example_id", example_id)
        return data

class PromptStudent:
    def __init__(self): ...
    def infer(self, prompt: str, example_id: str) -> str:
        msgs = student_messages(prompt, example_id)
        return student_llm_call(msgs)
