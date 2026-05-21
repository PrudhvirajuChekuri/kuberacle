"""Optional RAGAS evaluation integration."""

from k8s_rag.evaluation.runner import EvaluationCaseResult


def run_optional_ragas_metrics(
    case_results: list[EvaluationCaseResult],
) -> dict[str, object]:
    """Run RAGAS metrics if dependencies are available.

    Returns:
        Dict with `enabled` bool and either metric values or skip/error metadata.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except Exception as exc:  # pragma: no cover - dependency optional
        return {
            "enabled": False,
            "status": "skipped",
            "reason": f"RAGAS dependencies unavailable: {exc}",
        }

    rows = []
    for case in case_results:
        if not case.answerable:
            continue
        rows.append(
            {
                "question": case.question,
                "answer": case.answer,
                "contexts": case.retrieved_contexts,
                "ground_truth": case.expected_answer,
            }
        )

    if not rows:
        return {
            "enabled": False,
            "status": "skipped",
            "reason": "No answerable rows available for RAGAS metrics.",
        }

    try:  # pragma: no cover - depends on external libs/model providers
        dataset = Dataset.from_list(rows)
        scores = evaluate(
            dataset=dataset,
            metrics=[faithfulness, context_precision, answer_relevancy],
        )
        return {
            "enabled": True,
            "status": "ok",
            "faithfulness": float(scores.get("faithfulness", 0.0)),
            "context_precision": float(scores.get("context_precision", 0.0)),
            "answer_relevancy": float(scores.get("answer_relevancy", 0.0)),
            "rows_evaluated": len(rows),
        }
    except Exception as exc:
        return {
            "enabled": False,
            "status": "error",
            "reason": f"RAGAS run failed: {exc}",
        }
