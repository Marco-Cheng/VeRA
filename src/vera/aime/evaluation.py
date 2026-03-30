from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple


class StudentLike(Protocol):
    def infer(self, prompt: str, example_id: str) -> str:
        ...


@dataclass
class RunResult:
    run_index: int
    raw_response: str
    final_chunk: str
    final_value: Optional[float]
    correct: bool


@dataclass
class ProblemResult:
    item_id: str
    question: str
    gold_answer: str
    gold_value: Optional[float]
    runs: List[RunResult]
    avg_at_k: float
    pass_at_k: bool
    majority_vote_correct: bool
    majority_vote_answer: Optional[str]
    majority_vote_value: Optional[float]


def extract_final_answer(response: str) -> Tuple[str, Optional[float]]:
    """
    Extracts the final numeric answer using the repository-specific rule:
      - take the substring after the last whitespace;
      - keep leading sign;
      - respect only the first dot, drop subsequent dots;
      - ignore any other characters.
    Returns the processed chunk and the parsed float (if possible).
    """
    if not response:
        return "", None
    tail = response.rstrip().split()[-1]
    filtered: List[str] = []
    dot_seen = False
    for ch in tail:
        if ch in "+-":
            if not filtered:
                filtered.append(ch)
            continue
        if ch.isdigit():
            filtered.append(ch)
            continue
        if ch == ".":
            if not dot_seen:
                filtered.append(".")
                dot_seen = True
            continue
        # drop all other characters
    chunk = "".join(filtered).strip()
    if not chunk or chunk in {"+", "-"}:
        return tail, None
    try:
        return chunk, float(chunk)
    except ValueError:
        return chunk, None


def is_correct(pred: Optional[float], gold: Optional[float], tol: float = 1e-3) -> bool:
    if pred is None or gold is None:
        return False
    abs_diff = abs(pred - gold)
    if abs_diff <= tol:
        return True
    if abs(gold) > 1e-9 and abs_diff / abs(gold) <= tol:
        return True
    return False


def _majority_vote(runs: List[RunResult], gold_value: Optional[float], tol: float) -> Tuple[bool, Optional[str], Optional[float]]:
    if not runs:
        return False, None, None
    counts: Dict[str, int] = {}
    order: List[str] = []
    for res in runs:
        key = res.final_chunk
        counts[key] = counts.get(key, 0) + 1
        if key not in order:
            order.append(key)
    max_count = max(counts.values())
    candidates = [key for key, cnt in counts.items() if cnt == max_count]
    for key in order:
        if key in candidates:
            chosen = key
            break
    else:
        chosen = order[0]

    chosen_value = None
    for res in runs:
        if res.final_chunk == chosen:
            chosen_value = res.final_value
            break

    correct = is_correct(chosen_value, gold_value, tol)
    return correct, chosen if chosen else None, chosen_value


def evaluate_problem(
    item_id: str,
    question: str,
    gold_answer: str,
    gold_value: Optional[float],
    student: StudentLike,
    runs: int,
    tol: float = 1e-3,
) -> ProblemResult:
    run_results: List[RunResult] = []
    for run_index in range(1, runs + 1):
        response = student.infer(question, f"{item_id}::run{run_index}")
        chunk, value = extract_final_answer(response)
        correct = is_correct(value, gold_value, tol)
        run_results.append(
            RunResult(
                run_index=run_index,
                raw_response=response,
                final_chunk=chunk,
                final_value=value,
                correct=correct,
            )
        )

    avg_at_k = sum(int(res.correct) for res in run_results) / max(1, runs)
    pass_at_k = any(res.correct for res in run_results)
    majority_correct, majority_answer, majority_value = _majority_vote(run_results, gold_value, tol)

    return ProblemResult(
        item_id=item_id,
        question=question,
        gold_answer=gold_answer,
        gold_value=gold_value,
        runs=run_results,
        avg_at_k=avg_at_k,
        pass_at_k=pass_at_k,
        majority_vote_correct=majority_correct,
        majority_vote_answer=majority_answer,
        majority_vote_value=majority_value,
    )


def aggregate_results(results: List[ProblemResult]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {
            "avg_at_k": 0.0,
            "pass_at_k": 0.0,
            "majority_vote": 0.0,
            "details": [],
        }
    avg = sum(res.avg_at_k for res in results) / n
    pass_rate = sum(int(res.pass_at_k) for res in results) / n
    majority = sum(int(res.majority_vote_correct) for res in results) / n
    return {
        "avg_at_k": avg,
        "pass_at_k": pass_rate,
        "majority_vote": majority,
        "details": [
            {
                "item_id": res.item_id,
                "avg_at_k": res.avg_at_k,
                "pass_at_k": res.pass_at_k,
                "majority_vote_correct": res.majority_vote_correct,
                "majority_vote_answer": res.majority_vote_answer,
                "runs": [
                    {
                        "run_index": run.run_index,
                        "raw_response": run.raw_response,
                        "final_chunk": run.final_chunk,
                        "final_value": run.final_value,
                        "correct": run.correct,
                    }
                    for run in res.runs
                ],
            }
            for res in results
        ],
    }
