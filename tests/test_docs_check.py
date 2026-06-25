"""Tests for index staleness detection."""

from kuberacle.cli.docs_check import evaluate_staleness
from kuberacle.provenance import source_fingerprint

DOCS_PATH = "content/en/docs"
SECTIONS = ["concepts"]
PAGE = "content/en/docs/concepts/pods.md"
EXAMPLE = "content/en/examples/pods/simple-pod.yaml"
HUGO = "hugo.toml"
BUILD_FP = "build-fingerprint-abc"


def _published_manifest():
    """A manifest whose fingerprints match the BASE blob set below."""
    files = {
        PAGE: {"sha": "page-sha", "kind": "page"},
        EXAMPLE: {"sha": "example-sha", "kind": "example"},
        HUGO: {"sha": "hugo-sha", "kind": "config"},
    }
    return {
        "source_fingerprint": source_fingerprint(files),
        "build_fingerprint": BUILD_FP,
        "source_files": {"files": files},
    }


# Current upstream tree matching the published manifest exactly.
BASE_BLOBS = {PAGE: "page-sha", EXAMPLE: "example-sha", HUGO: "hugo-sha"}


def _evaluate(manifest, blobs, build_fp=BUILD_FP):
    return evaluate_staleness(manifest, blobs, DOCS_PATH, SECTIONS, build_fp)


def test_no_manifest_is_changed():
    result = _evaluate(None, BASE_BLOBS)
    assert result.changed is True
    assert "no published index" in result.reason


def test_up_to_date_is_not_changed():
    result = _evaluate(_published_manifest(), BASE_BLOBS)
    assert result.changed is False
    assert result.source_changed is False
    assert result.build_changed is False
    assert result.reason == "up to date"


def test_changed_page_is_detected():
    blobs = {**BASE_BLOBS, PAGE: "page-sha-v2"}
    result = _evaluate(_published_manifest(), blobs)
    assert result.source_changed is True
    assert result.changed is True


def test_new_page_is_detected():
    blobs = {**BASE_BLOBS, "content/en/docs/concepts/new.md": "new-sha"}
    result = _evaluate(_published_manifest(), blobs)
    assert result.source_changed is True


def test_deleted_dependency_is_detected():
    blobs = {k: v for k, v in BASE_BLOBS.items() if k != EXAMPLE}
    result = _evaluate(_published_manifest(), blobs)
    assert result.source_changed is True


def test_changed_dependency_is_detected():
    blobs = {**BASE_BLOBS, EXAMPLE: "example-sha-v2"}
    result = _evaluate(_published_manifest(), blobs)
    assert result.source_changed is True


def test_changed_hugo_is_detected():
    blobs = {**BASE_BLOBS, HUGO: "hugo-sha-v2"}
    result = _evaluate(_published_manifest(), blobs)
    assert result.source_changed is True


def test_changed_build_fingerprint_is_detected():
    result = _evaluate(_published_manifest(), BASE_BLOBS, build_fp="different")
    assert result.build_changed is True
    assert result.source_changed is False
    assert result.changed is True


def test_unrelated_upstream_file_does_not_trigger():
    """A file outside the watched sections/deps must not flip the decision."""
    blobs = {**BASE_BLOBS, "content/en/docs/reference/kubectl.md": "ref-sha"}
    result = _evaluate(_published_manifest(), blobs)
    assert result.changed is False
