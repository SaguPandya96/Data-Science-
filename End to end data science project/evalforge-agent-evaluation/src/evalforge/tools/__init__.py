"""Simulated productivity tools with typed contracts and deterministic fault injection.

Every tool here is a simulation. No email is sent, no calendar is touched, no file
outside the run directory is written, and no network call is made. Where a real system
would perform an external action, EvalForge records the *attempt* in the trace and
returns a payload marked ``simulated``.
"""

from __future__ import annotations

from evalforge.tools.artifacts import SaveArtifactInput, SaveArtifactOutput, SaveArtifactTool
from evalforge.tools.base import BaseTool, ToolContext, ToolInvocation
from evalforge.tools.corpus import Corpus, Document, load_corpus
from evalforge.tools.documents import (
    ExtractRequirementsInput,
    ExtractRequirementsOutput,
    ExtractRequirementsTool,
    SearchDocumentsInput,
    SearchDocumentsOutput,
    SearchDocumentsTool,
)
from evalforge.tools.planning import (
    BudgetLineItem,
    CalculateBudgetInput,
    CalculateBudgetOutput,
    CalculateBudgetTool,
    CreateProjectPlanInput,
    CreateProjectPlanTool,
    PlanPhase,
    ProjectPlanOutput,
    UpdateProjectPlanInput,
    UpdateProjectPlanTool,
)
from evalforge.tools.registry import ToolRegistry, UnknownToolError, build_registry
from evalforge.tools.writing import (
    DraftExecutiveSummaryInput,
    DraftExecutiveSummaryOutput,
    DraftExecutiveSummaryTool,
    DraftStakeholderEmailInput,
    DraftStakeholderEmailOutput,
    DraftStakeholderEmailTool,
)

__all__ = [
    "BaseTool",
    "BudgetLineItem",
    "CalculateBudgetInput",
    "CalculateBudgetOutput",
    "CalculateBudgetTool",
    "Corpus",
    "CreateProjectPlanInput",
    "CreateProjectPlanTool",
    "Document",
    "DraftExecutiveSummaryInput",
    "DraftExecutiveSummaryOutput",
    "DraftExecutiveSummaryTool",
    "DraftStakeholderEmailInput",
    "DraftStakeholderEmailOutput",
    "DraftStakeholderEmailTool",
    "ExtractRequirementsInput",
    "ExtractRequirementsOutput",
    "ExtractRequirementsTool",
    "PlanPhase",
    "ProjectPlanOutput",
    "SaveArtifactInput",
    "SaveArtifactOutput",
    "SaveArtifactTool",
    "SearchDocumentsInput",
    "SearchDocumentsOutput",
    "SearchDocumentsTool",
    "ToolContext",
    "ToolInvocation",
    "ToolRegistry",
    "UnknownToolError",
    "UpdateProjectPlanInput",
    "UpdateProjectPlanTool",
    "build_registry",
    "load_corpus",
]
