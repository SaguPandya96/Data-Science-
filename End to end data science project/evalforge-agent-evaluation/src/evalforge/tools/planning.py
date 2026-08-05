"""Budget and project-plan tools.

These three carry the state the whole evaluation turns on. ``calculate_budget`` is
deliberately exact arithmetic so ``incorrect_calculation`` can be a critical failure with
no false-positive risk, and ``update_project_plan`` reports which fields it touched so a
"you changed something nobody asked you to change" failure is directly observable rather
than inferred.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from evalforge.exceptions import ToolValidationError
from evalforge.schemas.common import ToolName
from evalforge.tools.base import BaseTool, ToolContext


class BudgetLineItem(BaseModel):
    """One costed line in a budget."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    amount: float = Field(ge=0.0)
    category: str = "delivery"


class CalculateBudgetInput(BaseModel):
    """Arguments for ``calculate_budget``."""

    model_config = ConfigDict(extra="forbid")

    line_items: list[BudgetLineItem] = Field(min_length=1)
    total_budget: float = Field(gt=0.0, description="Approved budget ceiling")
    contingency_rate: float = Field(default=0.10, ge=0.0, le=0.5)
    currency: str = "USD"


class CalculateBudgetOutput(BaseModel):
    """Result of ``calculate_budget``."""

    model_config = ConfigDict(extra="allow")

    total_budget: float
    allocated: float
    contingency: float
    remaining: float
    within_budget: bool
    currency: str
    breakdown: list[dict[str, Any]] = Field(default_factory=list)
    overspend: float = 0.0


class CalculateBudgetTool(BaseTool[CalculateBudgetInput, CalculateBudgetOutput]):
    """Sum line items against a ceiling and report the remaining headroom."""

    name: ClassVar[ToolName] = ToolName.CALCULATE_BUDGET
    input_model: ClassVar[type[BaseModel]] = CalculateBudgetInput
    output_model: ClassVar[type[BaseModel]] = CalculateBudgetOutput
    description: ClassVar[str] = "Total budget line items and compute remaining headroom."

    def _execute(
        self, payload: CalculateBudgetInput, context: ToolContext
    ) -> CalculateBudgetOutput:
        allocated = round(sum(item.amount for item in payload.line_items), 2)
        contingency = round(payload.total_budget * payload.contingency_rate, 2)
        remaining = round(payload.total_budget - allocated - contingency, 2)
        breakdown = [
            {
                "name": item.name,
                "amount": item.amount,
                "category": item.category,
                "share_of_budget": round(item.amount / payload.total_budget, 4),
            }
            for item in payload.line_items
        ]
        return CalculateBudgetOutput(
            total_budget=payload.total_budget,
            allocated=allocated,
            contingency=contingency,
            remaining=remaining,
            within_budget=remaining >= 0,
            currency=payload.currency,
            breakdown=breakdown,
            overspend=round(abs(min(remaining, 0.0)), 2),
        )


class PlanPhase(BaseModel):
    """One phase of a project plan."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    duration_weeks: float = Field(default=1.0, gt=0)
    owner: str = "delivery"
    cost: float = Field(default=0.0, ge=0.0)
    description: str = ""


class CreateProjectPlanInput(BaseModel):
    """Arguments for ``create_project_plan``."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = Field(min_length=1)
    launch_date: str = Field(min_length=1, description="Launch date as stated by the user")
    budget: float = Field(gt=0.0)
    phases: list[PlanPhase] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    currency: str = "USD"
    notes: str = ""


class ProjectPlanOutput(BaseModel):
    """A project plan document. Returned by create and update alike."""

    model_config = ConfigDict(extra="allow")

    plan_id: str
    project_name: str
    launch_date: str
    budget: float
    currency: str
    phases: list[dict[str, Any]] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    total_phase_cost: float = 0.0
    total_duration_weeks: float = 0.0
    notes: str = ""
    revision: int = 1
    changed_fields: list[str] = Field(default_factory=list)


def _plan_id(project_name: str) -> str:
    """Stable plan identifier derived from the project name."""
    slug = "".join(ch if ch.isalnum() else "_" for ch in project_name.lower()).strip("_")
    return f"plan_{slug[:40]}"


def _summarise_plan(
    project_name: str,
    launch_date: str,
    budget: float,
    currency: str,
    phases: list[dict[str, Any]],
    constraints: list[str],
    notes: str,
    revision: int,
    changed_fields: list[str],
) -> ProjectPlanOutput:
    """Assemble a plan payload with its derived totals."""
    return ProjectPlanOutput(
        plan_id=_plan_id(project_name),
        project_name=project_name,
        launch_date=launch_date,
        budget=budget,
        currency=currency,
        phases=phases,
        constraints=constraints,
        total_phase_cost=round(sum(float(p.get("cost", 0.0)) for p in phases), 2),
        total_duration_weeks=round(sum(float(p.get("duration_weeks", 0.0)) for p in phases), 2),
        notes=notes,
        revision=revision,
        changed_fields=changed_fields,
    )


class CreateProjectPlanTool(BaseTool[CreateProjectPlanInput, ProjectPlanOutput]):
    """Create a structured project plan."""

    name: ClassVar[ToolName] = ToolName.CREATE_PROJECT_PLAN
    input_model: ClassVar[type[BaseModel]] = CreateProjectPlanInput
    output_model: ClassVar[type[BaseModel]] = ProjectPlanOutput
    description: ClassVar[str] = "Create a project plan with phases, budget and launch date."

    def _execute(self, payload: CreateProjectPlanInput, context: ToolContext) -> ProjectPlanOutput:
        phases = [phase.model_dump() for phase in payload.phases]
        return _summarise_plan(
            project_name=payload.project_name,
            launch_date=payload.launch_date,
            budget=payload.budget,
            currency=payload.currency,
            phases=phases,
            constraints=list(payload.constraints),
            notes=payload.notes,
            revision=1,
            changed_fields=[],
        )


class UpdateProjectPlanInput(BaseModel):
    """Arguments for ``update_project_plan``.

    The existing plan is passed in whole rather than looked up by id, so the tool stays
    stateless and every mutation is visible in the trace.
    """

    model_config = ConfigDict(extra="forbid")

    plan: dict[str, Any] = Field(description="The plan being revised")
    launch_date: str | None = None
    budget: float | None = Field(default=None, gt=0.0)
    add_phases: list[PlanPhase] = Field(default_factory=list)
    remove_phases: list[str] = Field(default_factory=list)
    add_constraints: list[str] = Field(default_factory=list)
    remove_constraints: list[str] = Field(default_factory=list)
    notes: str | None = None


class UpdateProjectPlanTool(BaseTool[UpdateProjectPlanInput, ProjectPlanOutput]):
    """Apply a revision to an existing plan, reporting exactly what changed."""

    name: ClassVar[ToolName] = ToolName.UPDATE_PROJECT_PLAN
    input_model: ClassVar[type[BaseModel]] = UpdateProjectPlanInput
    output_model: ClassVar[type[BaseModel]] = ProjectPlanOutput
    description: ClassVar[str] = "Revise an existing project plan and report changed fields."

    def _execute(self, payload: UpdateProjectPlanInput, context: ToolContext) -> ProjectPlanOutput:
        plan = payload.plan
        if not plan.get("project_name"):
            raise ToolValidationError(
                "update_project_plan: plan must contain project_name; "
                "call create_project_plan first"
            )

        changed: list[str] = []
        launch_date = str(plan.get("launch_date", ""))
        if payload.launch_date is not None and payload.launch_date != launch_date:
            launch_date = payload.launch_date
            changed.append("launch_date")

        budget = float(plan.get("budget", 0.0))
        if payload.budget is not None and payload.budget != budget:
            budget = payload.budget
            changed.append("budget")

        phases: list[dict[str, Any]] = list(plan.get("phases", []))
        if payload.remove_phases:
            targets = {name.lower() for name in payload.remove_phases}
            kept = [p for p in phases if str(p.get("name", "")).lower() not in targets]
            if len(kept) != len(phases):
                changed.append("phases")
            phases = kept
        if payload.add_phases:
            phases = [*phases, *(phase.model_dump() for phase in payload.add_phases)]
            if "phases" not in changed:
                changed.append("phases")

        constraints: list[str] = list(plan.get("constraints", []))
        if payload.remove_constraints:
            targets = {c.lower() for c in payload.remove_constraints}
            kept_constraints = [c for c in constraints if c.lower() not in targets]
            if len(kept_constraints) != len(constraints):
                changed.append("constraints")
            constraints = kept_constraints
        for constraint in payload.add_constraints:
            if constraint not in constraints:
                constraints.append(constraint)
                if "constraints" not in changed:
                    changed.append("constraints")

        notes = str(plan.get("notes", ""))
        if payload.notes is not None and payload.notes != notes:
            notes = payload.notes
            changed.append("notes")

        return _summarise_plan(
            project_name=str(plan["project_name"]),
            launch_date=launch_date,
            budget=budget,
            currency=str(plan.get("currency", "USD")),
            phases=phases,
            constraints=constraints,
            notes=notes,
            revision=int(plan.get("revision", 1)) + 1,
            changed_fields=changed,
        )
