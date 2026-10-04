"""
eval.py -- score the chatbot on the questions in eval_set.jsonl.

Usage:  python eval.py

Three checks per question:
  1. retrieval hit  -- was the expected source file among the retrieved chunks?
  2. refusal        -- refused when it should, answered when it should
  3. answer facts   -- does the answer contain every phrase in "must_contain"?
"""
import json
from datetime import datetime, timezone

import config
from rag import answer


def load_eval_set(path="eval_set.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    cases = load_eval_set()
    retrieval_hits = facts_ok = refusal_ok = 0
    n_answerable = 0
    total_input_tokens = 0
    failures = []

    for case in cases:
        result = answer(case["question"])
        total_input_tokens += result["input_tokens"]
        sources = {s["source"] for s in result["sources"]}

        if case["answerable"]:
            n_answerable += 1
            hit = case["expected_source"] in sources
            facts = all(p.lower() in result["answer"].lower() for p in case["must_contain"])
            refusal_correct = not result["refused"]          # should NOT refuse
            retrieval_hits += hit
            facts_ok += facts
        else:
            hit = facts = None
            refusal_correct = result["refused"]              # SHOULD refuse

        refusal_ok += refusal_correct
        passed = refusal_correct and (hit is not False) and (facts is not False)
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {case['question']}")
        if not passed:
            failures.append((case, result))

    n = len(cases)
    summary = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "chunking": config.CHUNK_STRATEGY, "chunk_size": config.CHUNK_SIZE,
        "top_k": config.TOP_K, "min_similarity": config.MIN_SIMILARITY,
        "model": config.CHAT_MODEL,
        "retrieval_hit_rate": round(retrieval_hits / n_answerable, 3) if n_answerable else None,
        "answer_fact_rate": round(facts_ok / n_answerable, 3) if n_answerable else None,
        "refusal_accuracy": round(refusal_ok / n, 3),
        "avg_input_tokens": round(total_input_tokens / n, 1),
    }

    print("\n" + "=" * 60)
    for key in ("retrieval_hit_rate", "answer_fact_rate", "refusal_accuracy", "avg_input_tokens"):
        print(f"{key:<22} {summary[key]}")
    print("=" * 60)

    for case, result in failures:
        print(f"\nFAILED: {case['question']}")
        print(f"  expected: {case.get('expected_source', 'refusal')}  must_contain={case.get('must_contain')}")
        print(f"  got:      {result['answer'][:150]}")
        print(f"  retrieved: {[s['id'] for s in result['sources']]}  best_sim={result['best_similarity']}")

    # Keep a history of runs so you can compare settings (your chunking justification!)
    config.EVAL_LOG.parent.mkdir(exist_ok=True)
    with open(config.EVAL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"\nRun summary appended to {config.EVAL_LOG}")


if __name__ == "__main__":
    main()
