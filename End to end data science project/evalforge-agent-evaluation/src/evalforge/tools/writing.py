"""Drafting tools: executive summaries and stakeholder emails.

Both render a plan into prose deterministically, so an evaluator checking "does the
summary contain the original launch date" is checking the agent's *inputs*, not a
language model's phrasing. ``draft_stakeholder_email`` requires approval and never
sends: the returned payload always carries ``sent: false``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from evalforge.exceptions import ToolValidationError
from evalforge.schemas.common import ToolName
from evalforge.tools.base import BaseTool, ToolContext

#: Sections a summary contains by default, in render order.
DEFAULT_SECTIONS: tuple[str, ...] = ("objective", "timeline", "budget", "risks", "recommendation")


class DraftExecutiveSummaryInput(BaseModel):
    """Arguments for ``draft_executive_summary``."""

    model_config = ConfigDict(extra="forbid")

    plan: dict[str, Any] = Field(description="The plan to summarise")
    max_words: int | None = Field(default=None, gt=0)
    required_sections: list[str] = Field(default_factory=list)
    excluded_topics: list[str] = Field(default_factory=list)
    audience: str = "executive sponsor"
    use_table: bool = False


class DraftExecutiveSummaryOutput(BaseModel):
    """Result of ``draft_executive_summary``."""

    model_config = ConfigDict(extra="allow")

    text: str
    word_count: int
    sections: list[str] = Field(default_factory=list)
    project_name: str = ""
    launch_date: str = ""
    budget: float = 0.0
    within_word_limit: bool = True
    contains_table: bool = False


def _section_text(name: str, plan: dict[str, Any]) -> str:
    """Render one summary section from plan state."""
    project = str(plan.get("project_name", "the project"))
    launch = str(plan.get("launch_date", "an unconfirmed date"))
    budget = plan.get("budget", 0.0)
    currency = str(plan.get("currency", "USD"))
    phases = plan.get("phases", []) or []
    constraints = plan.get("constraints", []) or []

    if name == "objective":
        return f"Objective: deliver {project} to the agreed scope for the sponsor."
    if name == "timeline":
        phase_names = ", ".join(str(p.get("name", "phase")) for p in phases) or "no phases defined"
        weeks = plan.get("total_duration_weeks", 0)
        return (
            f"Timeline: launch is set for {launch}. Phases: {phase_names}. "
            f"Total planned duration is {weeks} weeks."
        )
    if name == "budget":
        allocated = plan.get("total_phase_cost", 0.0)
        return (
            f"Budget: the approved budget is {budget} {currency}, "
            f"of which {allocated} {currency} is allocated to named phases."
        )
    if name == "risks":
        if constraints:
            joined = "; ".join(str(c) for c in constraints)
            return f"Risks and constraints: the following remain in force: {joined}."
        return "Risks and constraints: no constraints were recorded for this plan."
    if name == "recommendation":
        return f"Recommendation: proceed with {project} as planned, subject to sponsor approval."
    return f"{name.title()}: not specified."


def _budget_table(plan: dict[str, Any]) -> str:
    """Render phase costs as a Markdown table."""
    currency = str(plan.get("currency", "USD"))
    rows = ["| Phase | Cost | Duration (weeks) |", "| --- | --- | --- |"]
    for phase in plan.get("phases", []) or []:
        rows.append(
            f"| {phase.get('name', 'phase')} | {phase.get('cost', 0)} {currency} "
            f"| {phase.get('duration_weeks', 0)} |"
        )
    return "\n".join(rows)


class DraftExecutiveSummaryTool(BaseTool[DraftExecutiveSummaryInput, DraftExecutiveSummaryOutput]):
    """Render a plan into an executive summary."""

    name: ClassVar[ToolName] = ToolName.DRAFT_EXECUTIVE_SUMMARY
    input_model: ClassVar[type[BaseModel]] = DraftExecutiveSummaryInput
    output_model: ClassVar[type[BaseModel]] = DraftExecutiveSummaryOutput
    description: ClassVar[str] = "Draft an executive summary from a project plan."

    def _execute(
        self, payload: DraftExecutiveSummaryInput, context: ToolContext
    ) -> DraftExecutiveSummaryOutput:
        plan = payload.plan
        if not plan.get("project_name"):
            raise ToolValidationError(
                "draft_executive_summary: plan must contain project_name; "
                "create or retrieve the plan first"
            )

        sections = [s.lower() for s in (payload.required_sections or DEFAULT_SECTIONS)]
        parts = [_section_text(section, plan) for section in sections]

        if payload.use_table:
            parts.append(_budget_table(plan))

        text = "\n\n".join(parts)

        # Excluded topics are removed sentence-wise rather than by deleting the token,
        # so the output stays readable and a violation is unambiguous when it appears.
        for topic in payload.excluded_topics:
            text = "\n\n".join(
                block for block in text.split("\n\n") if topic.lower() not in block.lower()
            )

        words = text.split()
        if payload.max_words is not None and len(words) > payload.max_words:
            text = " ".join(words[: payload.max_words])
            words = text.split()

        return DraftExecutiveSummaryOutput(
            text=text,
            word_count=len(words),
            sections=sections,
            project_name=str(plan.get("project_name", "")),
            launch_date=str(plan.get("launch_date", "")),
            budget=float(plan.get("budget", 0.0)),
            within_word_limit=payload.max_words is None or len(words) <= payload.max_words,
            contains_table=payload.use_table,
        )


class DraftStakeholderEmailInput(BaseModel):
    """Arguments for ``draft_stakeholder_email``."""

    model_config = ConfigDict(extra="forbid")

    plan: dict[str, Any]
    recipients: list[str] = Field(min_length=1)
    subject: str = Field(min_length=1)
    body_summary: str = ""
    redact_entities: list[str] = Field(default_factory=list)


class DraftStakeholderEmailOutput(BaseModel):
    """Result of ``draft_stakeholder_email``. Nothing is ever transmitted."""

    model_config = ConfigDict(extra="allow")

    subject: str
    body: str
    recipients: list[str] = Field(default_factory=list)
    #: Always false. EvalForge simulates delivery and never contacts anything external.
    sent: bool = False
    simulated: bool = True
    word_count: int = 0
    redacted_entities: list[str] = Field(default_factory=list)


class DraftStakeholderEmailTool(BaseTool[DraftStakeholderEmailInput, DraftStakeholderEmailOutput]):
    """Draft a stakeholder email. Requires approval; never sends."""

    name: ClassVar[ToolName] = ToolName.DRAFT_STAKEHOLDER_EMAIL
    input_model: ClassVar[type[BaseModel]] = DraftStakeholderEmailInput
    output_model: ClassVar[type[BaseModel]] = DraftStakeholderEmailOutput
    requires_approval: ClassVar[bool] = True
    description: ClassVar[str] = (
        "Draft a stakeholder email (simulated; requires approval, never sends)."
    )

    def _execute(
        self, payload: DraftStakeholderEmailInput, context: ToolContext
    ) -> DraftStakeholderEmailOutput:
        plan = payload.plan
        project = str(plan.get("project_name", "the project"))
        launch = str(plan.get("launch_date", "an unconfirmed date"))
        budget = plan.get("budget", 0.0)
        currency = str(plan.get("currency", "USD"))

        body = payload.body_summary or (
            f"Hello,\n\n"
            f"Here is the current position on {project}. "
            f"Launch remains scheduled for {launch}, with an approved budget of "
            f"{budget} {currency}.\n\n"
            f"Please reply with any concerns before the next review.\n\n"
            f"Regards,\nDelivery team"
        )

        redacted: list[str] = []
        for entity in payload.redact_entities:
            if entity and entity in body:
                body = body.replace(entity, "[redacted]")
                redacted.append(entity)

        return DraftStakeholderEmailOutput(
            subject=payload.subject,
            body=body,
            recipients=list(payload.recipients),
            sent=False,
            simulated=True,
            word_count=len(body.split()),
            redacted_entities=redacted,
        )
