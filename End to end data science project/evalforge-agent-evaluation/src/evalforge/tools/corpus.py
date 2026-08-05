"""The fictional document corpus the productivity tools search.

Everything in the corpus is invented for evaluation purposes. Three documents carry
deliberate prompt-injection payloads: retrieved content is the primary untrusted input
surface for an agent with tools, so a corpus without adversarial documents cannot test
the boundary that matters most.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from evalforge.exceptions import ConfigurationError


class RequirementRecord(BaseModel):
    """One extracted requirement line."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    description: str
    category: str = "functional"
    estimated_cost: float = Field(default=0.0, ge=0.0)
    priority: str = "should"


class Document(BaseModel):
    """A single fictional project document.

    Args:
        doc_id: Stable identifier.
        title: Human-readable title.
        project: Project slug the document belongs to.
        doc_type: brief, register, costing, risk, schedule, policy, template or note.
        keywords: Terms the search tool matches against.
        content: Body text, which may contain an injection payload.
        requirements: Structured requirements extractable from this document.
        confidential_entities: Names that must never reach an external artifact.
        contains_injection: Whether the body carries an adversarial instruction.
        injection_payload: The adversarial span, used by the injection evaluator.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    title: str
    project: str
    doc_type: str = "note"
    keywords: list[str] = Field(default_factory=list)
    content: str = ""
    requirements: list[RequirementRecord] = Field(default_factory=list)
    confidential_entities: list[str] = Field(default_factory=list)
    contains_injection: bool = False
    injection_payload: str = ""

    def relevance(self, query: str) -> float:
        """Deterministic lexical relevance of this document to ``query``.

        A simple weighted term-overlap score is used rather than embeddings: the search
        tool exists to be *deterministic and inspectable*, not to be a good search
        engine. Its quality is not what is under evaluation.
        """
        terms = {token for token in query.lower().split() if len(token) > 2}
        if not terms:
            return 0.0
        keyword_set = {k.lower() for k in self.keywords}
        title_terms = set(self.title.lower().split())
        body_terms = set(self.content.lower().split())

        score = 0.0
        score += 3.0 * len(terms & keyword_set)
        score += 2.0 * len(terms & title_terms)
        score += 1.0 * len(terms & body_terms)
        if self.project.replace("_", " ") in query.lower():
            score += 2.0
        return score / (3.0 * len(terms))


class Corpus(BaseModel):
    """The whole document collection."""

    model_config = ConfigDict(extra="forbid")

    corpus_version: str = "1.0.0"
    notice: str = ""
    documents: list[Document] = Field(default_factory=list)

    def by_id(self, doc_id: str) -> Document | None:
        """Look up one document."""
        for document in self.documents:
            if document.doc_id == doc_id:
                return document
        return None

    def for_project(self, project: str) -> list[Document]:
        """Every document belonging to ``project``."""
        return [d for d in self.documents if d.project == project]

    def search(self, query: str, project: str | None = None, limit: int = 5) -> list[Document]:
        """Return the ``limit`` most relevant documents, ties broken by ``doc_id``.

        Sorting includes ``doc_id`` as a tiebreaker so results are stable regardless of
        corpus file ordering.
        """
        pool = self.for_project(project) if project else self.documents
        scored = [(doc.relevance(query), doc) for doc in pool]
        ranked = sorted(
            (item for item in scored if item[0] > 0),
            key=lambda item: (-item[0], item[1].doc_id),
        )
        return [doc for _, doc in ranked[:limit]]

    @property
    def injection_documents(self) -> list[Document]:
        """Documents carrying an adversarial instruction."""
        return [d for d in self.documents if d.contains_injection]

    @property
    def confidential_names(self) -> set[str]:
        """Every protected name across the corpus."""
        return {name for doc in self.documents for name in doc.confidential_entities}


@lru_cache(maxsize=4)
def load_corpus(path: Path | None = None) -> Corpus:
    """Load and cache the corpus.

    Args:
        path: Explicit corpus file. Defaults to ``data/sample_documents/corpus.json``.

    Raises:
        ConfigurationError: If the file is missing or unparseable.
    """
    if path is None:
        from evalforge.config import PROJECT_ROOT

        path = PROJECT_ROOT / "data" / "sample_documents" / "corpus.json"
    if not path.exists():
        raise ConfigurationError(
            f"Document corpus not found at {path}. "
            "It ships with the repository under data/sample_documents/."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Corpus at {path} is not valid JSON: {exc}") from exc
    return Corpus.model_validate(raw)
