"""Measure how well the RAG system answers a small test set.

Two simple, honest metrics:
- Retrieval hit rate: did the retrieved chunks contain the expected keywords?
- Answer accuracy: did the final answer contain the expected keywords?
Questions marked "unanswerable" must get an "I don't know" answer.
"""
import json
import time
from pathlib import Path

from rag_engine import answer_question

EVAL_DIR = Path(__file__).parent / "eval"
TEST_FILE = EVAL_DIR / "test_set.json"
RESULTS_FILE = EVAL_DIR / "results.json"


def _clean(text):
    return text.lower().replace("\u2019", "'")


def contains_all(text, keywords):
    text = _clean(text)
    return all(_clean(k) in text for k in keywords)


def run_eval(progress=None):
    tests = json.loads(TEST_FILE.read_text(encoding="utf-8"))
    results = []
    for i, t in enumerate(tests):
        out = answer_question(t["question"])
        unanswerable = t.get("unanswerable", False)
        retrieval_ok = None if unanswerable else contains_all(" ".join(out["chunks"]), t["expected_keywords"])
        answer_ok = contains_all(out["answer"], t["expected_keywords"])
        results.append(
            {
                "question": t["question"],
                "answer": out["answer"],
                "expected_keywords": t["expected_keywords"],
                "unanswerable": unanswerable,
                "retrieval_ok": retrieval_ok,
                "answer_ok": answer_ok,
            }
        )
        if progress:
            progress((i + 1) / len(tests))
        time.sleep(0.5)  # stay under Groq rate limits

    answerable = [r for r in results if not r["unanswerable"]]
    summary = {
        "total_questions": len(results),
        "answer_accuracy": round(sum(r["answer_ok"] for r in results) / len(results), 3),
        "retrieval_hit_rate": round(sum(r["retrieval_ok"] for r in answerable) / max(len(answerable), 1), 3),
    }
    RESULTS_FILE.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    return summary, results


if __name__ == "__main__":
    summary, results = run_eval()
    for r in results:
        mark = "PASS" if r["answer_ok"] else "FAIL"
        print(f"[{mark}] {r['question']}\n       -> {r['answer']}")
    print("\nSummary:", summary)