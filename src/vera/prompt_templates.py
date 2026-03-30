from __future__ import annotations
from typing import List, Dict, Optional

def teacher_messages(question: str, answer: str, example_id: str, n_text_templates: int = 6, feedback: Optional[str] = None) -> List[Dict[str,str]]:
    sys = (
        "You are an expert at parameterizing math word problems in GSM8K. "
        "You need to generalize a problem by parameterizing as many parts of it as possible without changing the core calculation logic."
        "Return ONE JSON OBJECT only. No code fences, no prose."
    )
    schema_example='''
OUTPUT SCHEMA (STRICT JSON, an example below)
---------------------------
{
  "text_templates": [
    "If [name1] has [a] apples and [name2] has [b], how many more does [name1] have?",
    "[name1] possesses [a] apples while [name2] has [b]; compute the difference.",
    "有[name1][a]个苹果，[name2]有[b]个。[name1]比[name2]多几个？",
    "Si [name1] a [a] pommes et [name2] en a [b], quelle est la différence ?",
    "[name1] has [a] marbles; [name2] has [b]. Find [a] - [b].",
    "Suppose [name1] bought [a] stickers and [name2] bought [b]. How many more did [name1] buy?"
  ],
  "slots": {
    "a": {"kind":"int","interval":[0,1], "map": {"kind":"int_range","lo":5,"hi":50,"step":1}, "base_value": 12},
    "b": {"kind":"int","interval":[0,1], "map": {"kind":"int_range","lo":1,"hi":50,"step":1}, "base_value": 3},
    "name1": {"kind":"entity","meta":{"names":["Alice","Xiao Ming","Jean","Lucia"]}, "base_value":"Alice"},
    "name2": {"kind":"entity","meta":{"names":["Bob","Xiao Hong","Marie","Diego"]}, "base_value":"Bob"}
  },
  "verifier": {"type":"python","code":"def verifier(assign):\n    a = int(assign['a']); b = int(assign['b'])\n    if a<=b or a<0 or b<0: return False, None\n    return True, a - b"},
  "generator": {"type":"python","code":"def generator(rng):\n    a = rng.randint(5,50)\n    b = rng.randint(1,a-1)\n    name1 = 'Alice'\n    name2 = 'Bob'\n    return {'a':a,'b':b,'name1':name1,'name2':name2}"},
  "base_assignment": {"a":12, "b":3, "name1":"Alice", "name2":"Bob"},
  "meta": {"source":"gsm8k","example_id":"<EXAMPLE_ID>"}
}
'''.rstrip()
    
    parts = [
        "SEED",
        "----",
        f"question_id: {example_id}",
        f"question: {question}",
        f"answer: {answer}",
        "",
        "GOAL",
        "----",
        "Output a SINGLE JSON object implementing a minimal, coherent augmentation spec of the given seed question above:",
        f"- text_templates: list[str] using [slot_name] placeholders (REQUIRED). Number of templates = {n_text_templates}.",
        "  * Be CREATIVE and DIFFERENT while keeping the same core math logic consistent and validated by the verifier/generator.",
        "  * Allowed changes (not necessarily all, no particular order, must keep slot count and names consistent across ALL templates):",
        "      1) Paraphrase.",
        "      2) Translate to other languages (Chinese, French, Spanish, etc.).",
        "      3) Change/add/remove background story or entities (names, places), as long as slot semantics are preserved.",
        "  * Do NOT leak the final answer y inside templates.",
        "- slots: either a dict {{\"slot_name\":{{...}} }} or a list of objects with fields:",
        "    name, kind in [\"int\",\"float\",\"choice\",\"str\",\"entity\",\"unit\"], optional interval/map, weight, base_value, meta.",
        "- verifier:  { \"type\":\"python\", \"code\": \"def verifier(assign): ... return True, y\" } - A verifier should validate whether the input assign is valid, and return the desired answer for a valid assignment  (if assign is invalid, return False, None).",
        "- generator: { \"type\":\"python\", \"code\": \"def generator(rng): ... return assign\" } - The generator should randomly generate an assign whose format is coherent with the verifier and can pass the verifier.",
        "- base_assignment: assignment corresponding to the original seed question; MUST pass the verifier.",
        "",
        "RULES",
        "-----",
        "- Use ONLY [slot_name] tokens in text_templates (no {{braces}}). Slot names must be ASCII snake_case.",
        "- The set of slot names in text_templates, slots, base_assignment, verifier inputs, and generator outputs MUST MATCH exactly.",
        "- Code restrictions: pure Python 3; no I/O; no imports; we provide 'math' and 'random' at runtime, but be sure to refer to them (e.g. math.gcd, math.lcm).",
        "- y should be numeric when the problem is numeric (usual GSM8K).",
        "- Your output should always guarantee that the output of generator MUST PASS the verifier and contributes to a valid new augmented task.",
        "",
        schema_example,
    ]

    if feedback:
        parts += [
            "",
            "RETRY_FEEDBACK",
            "--------------",
            "The previous attempt failed. Diagnose and fix the issues below.",
            "Ensure generator yields VALID assignments passing the verifier, and that verifier compiles and returns (bool, y).",
            "",
            "Issues to fix:",
            feedback,
        ]

    parts += [
        "",
        "OUTPUT",
        "------",
        "Return ONLY the JSON object.",
    ]

    user = "\n".join(parts)
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]


def student_messages(prompt: str, example_id: str) -> List[Dict[str,str]]:
    sys = (
        "You are solving a math problem. "
        "Respond with ONLY the final numeric answer (retain at most 2 decimal points) in the format: '#### <number>'."
    )
    user = f"""question_id: {example_id}

Problem:
{prompt}

Output format (STRICT):
#### <number>
"""
    return [
        {"role":"system","content": sys},
        {"role":"user","content": user}
    ]


def aime_teacher_messages(example_id: str, question: str, answer: str, year: Optional[int] = None, feedback: Optional[str] = None) -> List[Dict[str, str]]:
    year_info = f"{year}" if year is not None else "Unknown"
    sys = (
        "You are an experienced math olympiad problem setter who specializes in making existing AIME problems harder and more generalizable. "
        "You are given a seed problem from AIME contest, and need to generate a more difficult problem family based on that."
        "Please modify at least one condition (of course you can modify more or change the task entirely) of the seed problem to design a new, more difficult problem family. "
        "This new problem family must require a different and more advanced solution approach (of course can be similar) from the original and should not be solvable by guessing."
        "The problem family must be of such quality and novelty that any instance of the family could be accepted on a venue like IMO, AIME, or at the very least serve as a valuable training exercise."
        "Furthermore, please provide a proof of correctness, and a clear explanation for your generalized family in the metadata."
        "Always return ONE JSON object. No commentary."
    )
    schema = r"""
STRICT JSON SCHEMA
------------------
{
  "language_wrapper": "In a math contest, {alpha} students ...",  # single string with {slot} or [slot] placeholders
  "slots": {                  # optional metadata; keep slot names consistent with wrapper/generator/verifier
    "alpha": {"kind": "int", "description": "total students", "harder_than_seed": true}
  },
  "generator": {
    "type": "python",
    "code": "def generator(rng):\n    # use rng.* for randomness\n    ...\n    return {'alpha': ..., 'beta': ...}"
  },
  "verifier": {
    "type": "python",
    "code": "def verifier(assign):\n    # recompute, validate domain constraints, and return (True, answer)\n    ...\n    return True, answer"
  },
  "hardness_rationale": "Explain briefly why the generated family is more difficult than the seed.",
  "notes": "Optional implementation notes for future maintainers.",
  "meta": {
    "seed_id": "<example_id>",
    "source_year": <year_int>
  }
}
""".strip()

    parts = [
        "SEED PROBLEM",
        "-------------",
        f"id: {example_id}",
        f"year: {year_info}",
        "question:",
        question.strip(),
        "answer:",
        answer.strip(),
        "",
        "GOAL",
        "----",
        "Produce a parametric augmentation that:",
        "1. Each instance of the family preserves the same core logical idea while allowing different numeric instantiations.",
        "2. Boosts the difficulty of the seed (e.g., more steps, trickier algebra, edge cases). As hard as possible, but you should be able to reason and explain your solution.",
        "3. Uses a single language_wrapper string with placeholders; all numeric values must come from code.",
        "4. Supplies generator/validator Python snippets that cooperate: generator(rng) -> dict assignment; verifier(assign) -> (bool, answer).",
        "5. Ensures verifier enforces correctness AND difficulty (no trivial shortcuts).",
        "6. Guarantees generator relies on rng.*, retries internally if necessary, and avoids undefined math operations.",
        "",
        schema,
    ]

    if feedback:
        parts += [
            "",
            "FEEDBACK FROM PREVIOUS ATTEMPT",
            "------------------------------",
            feedback,
        ]

    parts += [
        "",
        "OUTPUT REQUIREMENTS",
        "-------------------",
        "- Return exactly ONE JSON object.",
        "- language_wrapper must use {snake_case} or [snake_case] placeholders that appear in the assignments.",
        "- The verifier must return (bool, answer). When invalid, return (False, None) and explain via comments.",
        "- The generator must frequently produce valid, non-trivial assignments that satisfy the verifier.",
        "- Do not leak the numeric answer inside language_wrapper. All numbers must come from the generator.",
        "- The provided rng shim exposes methods like rng.random(), rng.uniform(a,b), rng.randint(lo, hi), rng.randrange(...), rng.choice(seq), rng.sample(seq,k), rng.shuffle(seq), rng.getrandbits(k), and basic distributions (gauss, gammavariate, etc.). Do NOT call nonexistent methods such as rng.integers or rely on numpy/random imports.",
    ]

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": "\n".join(parts)},
    ]


def judge_messages(question: str, candidate_answer: str, teacher_context: Optional[str] = None) -> List[Dict[str, str]]:
    sys = (
        "You are a meticulous mathematics judge. Determine if a proposed answer solves the problem exactly. "
        "You will see a teacher LLM's reasoning and proposed generalization, but it may contain mistakes. "
        "Evaluate the problem + candidate answer independently; only output 'True' if the reasoning and computations fully justify the answer."
    )
    user_lines = [
        "PROBLEM",
        "-------",
        question.strip(),
        "",
        "CANDIDATE ANSWER",
        "----------------",
        candidate_answer.strip(),
        "",
        "TASK",
        "----",
        "Think carefully. Teacher reasoning (if any) might be flawed, and the candidate answer might still be wrong.",
        "Return an explanation followed by a final line that is exactly True or False, reflecting your final judgement.",
    ]
    if teacher_context:
        user_lines += [
            "",
            "TEACHER CONTEXT (MAY BE INCORRECT)",
            "----------------------------------",
            teacher_context.strip(),
        ]
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


def hardest_variant_messages(seed_id: str, variants: List[Dict[str, str]]) -> List[Dict[str, str]]:
    sys = (
        "You are ranking math contest problems by difficulty. "
        "Given several variants derived from the same seed, choose the hardest one. "
        "Respond with a JSON object {'hardest_variant': '<variant_id>', 'reason': '...'}."
    )
    lines = [
        f"Seed: {seed_id}",
        "",
        "Variants (consider complexity, number of steps, tricky reasoning, boundary cases):",
        "--------------------------------------------------------------------------",
    ]
    for idx, item in enumerate(variants, start=1):
        lines += [
            f"[{item['variant_id']}] (prompt_attempt={item.get('prompt_attempt')}, generator_attempts={item.get('generator_attempts')})",
            f"Question: {item['question']}",
            f"Rationale: {item.get('hardness_rationale', 'N/A')}",
            "",
        ]
    lines += [
        "TASK: Pick the single hardest variant. Return strict JSON with keys 'hardest_variant' and 'reason'.",
    ]
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": "\n".join(lines)},
    ]
