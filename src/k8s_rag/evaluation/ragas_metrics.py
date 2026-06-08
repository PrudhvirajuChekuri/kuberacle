"""RAGAS faithfulness, context precision, and answer relevancy evaluation."""

import logging
from dataclasses import dataclass

from k8s_rag.evaluation.runner import EvaluationCaseResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaithfulnessResult:
    """Faithfulness evaluation output.

    Attributes:
        mean: Mean faithfulness score over successfully parsed cases.
        parsed_count: Number of cases successfully scored.
        total_count: Total answerable cases submitted.
    """

    mean: float
    parsed_count: int
    total_count: int


@dataclass(frozen=True)
class AnswerRelevancyResult:
    """Answer relevancy evaluation output.

    Attributes:
        mean: Mean answer relevancy score over successfully parsed cases.
        parsed_count: Number of cases successfully scored.
        total_count: Total answerable cases submitted.
    """

    mean: float
    parsed_count: int
    total_count: int


@dataclass(frozen=True)
class ContextPrecisionResult:
    """Context precision evaluation output.

    Attributes:
        mean: Mean context precision score over successfully parsed cases.
        parsed_count: Number of cases successfully scored.
        total_count: Total answerable cases submitted.
    """

    mean: float
    parsed_count: int
    total_count: int


def _build_llm(judge_model: str, gcp_project: str, gcp_location: str):
    """Build a LangChain LLM wrapper for use as a RAGAS judge.

    Applies a sys.modules patch so ragas 0.4.x can import without error -
    ragas imports ChatVertexAI from a langchain_community path that was
    removed in langchain-community 0.4.

    Args:
        judge_model: Google Generative AI model ID.
        gcp_project: GCP project ID. Passed explicitly to prevent the google.genai
            client from calling google.auth.default() internally without scopes.
        gcp_location: GCP region.

    Returns:
        Configured LangchainLLMWrapper.

    Raises:
        RuntimeError: If required dependencies are not installed.
    """
    try:
        import sys
        from types import ModuleType
        from langchain_google_genai import ChatGoogleGenerativeAI
        _stub = ModuleType("langchain_community.chat_models.vertexai")
        _stub.ChatVertexAI = ChatGoogleGenerativeAI  # type: ignore[attr-defined]
        sys.modules.setdefault("langchain_community.chat_models.vertexai", _stub)

        import google.auth
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise RuntimeError(
            f"RAGAS dependencies are not installed. "
            f"Run: pip install ragas langchain-google-genai. Error: {exc}"
        ) from exc

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=judge_model,
            credentials=credentials,
            project=gcp_project,
            location=gcp_location,
        )
    )


def _build_embeddings(embedding_model: str, gcp_project: str, gcp_location: str):
    """Build a LangChain embeddings wrapper for use by RAGAS answer relevancy.

    Args:
        embedding_model: Google Generative AI embedding model ID.
        gcp_project: GCP project ID. Passed explicitly to prevent the google.genai
            client from calling google.auth.default() internally without scopes.
        gcp_location: GCP region.

    Returns:
        Configured LangchainEmbeddingsWrapper.

    Raises:
        RuntimeError: If required dependencies are not installed.
    """
    try:
        import google.auth
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            f"RAGAS dependencies are not installed. "
            f"Run: pip install ragas langchain-google-genai. Error: {exc}"
        ) from exc

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model=f"models/{embedding_model}",
            credentials=credentials,
            project=gcp_project,
            location=gcp_location,
        )
    )


def _build_rows(case_results: list[EvaluationCaseResult]) -> list[dict]:
    """Build RAGAS input rows from answerable eval cases."""
    return [
        {
            "user_input": case.question,
            "response": case.answer,
            "retrieved_contexts": case.retrieved_contexts,
            "reference": case.expected_answer,
        }
        for case in case_results
        if case.answerable
    ]


def _extract_scores(df, column: str, total_count: int) -> tuple[float, int]:
    """Extract mean and parsed count from a ragas result dataframe column."""
    scores = df[column].dropna().tolist()
    parsed_count = len(scores)
    if parsed_count < total_count:
        logger.warning(
            "%s: %d/%d cases failed to score and will be excluded from the mean.",
            column,
            total_count - parsed_count,
            total_count,
        )
    mean = sum(scores) / parsed_count if parsed_count > 0 else 0.0
    return mean, parsed_count


def compute_faithfulness(
    case_results: list[EvaluationCaseResult],
    gcp_project: str,
    gcp_location: str,
    judge_model: str,
) -> FaithfulnessResult:
    """Compute RAGAS faithfulness over answerable eval cases.

    Args:
        case_results: Per-case evaluation outputs from the main eval run.
        gcp_project: GCP project ID. Forwarded to the LLM client to prevent
            google.auth.default() calls without scopes inside google.genai.
        gcp_location: GCP region.
        judge_model: Google Generative AI model ID to use as the judge.

    Returns:
        FaithfulnessResult with mean score over parsed cases and counts.

    Raises:
        RuntimeError: If ragas or langchain-google-genai are not installed.
        Exception: If the RAGAS evaluation call fails entirely.
    """
    llm = _build_llm(judge_model, gcp_project, gcp_location)

    from datasets import Dataset
    from ragas import evaluate, RunConfig
    from ragas.metrics import faithfulness

    faithfulness.llm = llm

    rows = _build_rows(case_results)
    total_count = len(rows)

    result = evaluate(dataset=Dataset.from_list(rows), metrics=[faithfulness], run_config=RunConfig(max_workers=1))
    mean, parsed_count = _extract_scores(result.to_pandas(), "faithfulness", total_count)
    return FaithfulnessResult(mean=mean, parsed_count=parsed_count, total_count=total_count)


def compute_context_precision(
    case_results: list[EvaluationCaseResult],
    gcp_project: str,
    gcp_location: str,
    judge_model: str,
) -> ContextPrecisionResult:
    """Compute RAGAS context precision over answerable eval cases.

    Args:
        case_results: Per-case evaluation outputs from the main eval run.
        gcp_project: GCP project ID. Forwarded to the LLM client to prevent
            google.auth.default() calls without scopes inside google.genai.
        gcp_location: GCP region.
        judge_model: Google Generative AI model ID to use as the judge.

    Returns:
        ContextPrecisionResult with mean score over parsed cases and counts.

    Raises:
        RuntimeError: If ragas or langchain-google-genai are not installed.
        Exception: If the RAGAS evaluation call fails entirely.
    """
    llm = _build_llm(judge_model, gcp_project, gcp_location)

    from datasets import Dataset
    from ragas import evaluate, RunConfig
    from ragas.metrics import context_precision

    context_precision.llm = llm

    rows = _build_rows(case_results)
    total_count = len(rows)

    result = evaluate(dataset=Dataset.from_list(rows), metrics=[context_precision], run_config=RunConfig(max_workers=1))
    mean, parsed_count = _extract_scores(result.to_pandas(), "context_precision", total_count)
    return ContextPrecisionResult(mean=mean, parsed_count=parsed_count, total_count=total_count)


def compute_answer_relevancy(
    case_results: list[EvaluationCaseResult],
    gcp_project: str,
    gcp_location: str,
    judge_model: str,
    embedding_model: str,
) -> AnswerRelevancyResult:
    """Compute RAGAS answer relevancy over answerable eval cases.

    Args:
        case_results: Per-case evaluation outputs from the main eval run.
        gcp_project: GCP project ID. Forwarded to the LLM and embeddings clients
            to prevent google.auth.default() calls without scopes inside google.genai.
        gcp_location: GCP region.
        judge_model: Google Generative AI model ID to use as the judge.
        embedding_model: Google Generative AI embedding model ID for similarity scoring.

    Returns:
        AnswerRelevancyResult with mean score over parsed cases and counts.

    Raises:
        RuntimeError: If ragas or langchain-google-genai are not installed.
        Exception: If the RAGAS evaluation call fails entirely.
    """
    llm = _build_llm(judge_model, gcp_project, gcp_location)
    embeddings = _build_embeddings(embedding_model, gcp_project, gcp_location)

    from datasets import Dataset
    from ragas import evaluate, RunConfig
    from ragas.metrics import answer_relevancy

    answer_relevancy.llm = llm
    answer_relevancy.embeddings = embeddings

    rows = _build_rows(case_results)
    total_count = len(rows)

    result = evaluate(dataset=Dataset.from_list(rows), metrics=[answer_relevancy], run_config=RunConfig(max_workers=1))
    mean, parsed_count = _extract_scores(result.to_pandas(), "answer_relevancy", total_count)
    return AnswerRelevancyResult(mean=mean, parsed_count=parsed_count, total_count=total_count)
