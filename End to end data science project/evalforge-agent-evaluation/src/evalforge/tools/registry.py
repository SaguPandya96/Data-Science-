"""Tool registry.

One place that knows which tools exist. The agent, the mock provider and the dashboard
all resolve tools through here, so adding a tool is a single-line change rather than
three co-ordinated ones.
"""

from __future__ import annotations

from typing import Any

from evalforge.exceptions import ToolError
from evalforge.schemas.common import ToolName
from evalforge.tools.artifacts import SaveArtifactTool
from evalforge.tools.base import BaseTool
from evalforge.tools.documents import ExtractRequirementsTool, SearchDocumentsTool
from evalforge.tools.planning import (
    CalculateBudgetTool,
    CreateProjectPlanTool,
    UpdateProjectPlanTool,
)
from evalforge.tools.writing import DraftExecutiveSummaryTool, DraftStakeholderEmailTool


class UnknownToolError(ToolError):
    """The agent named a tool that does not exist.

    Modelled as a tool error rather than a lookup failure so a hallucinated tool name
    flows through the same recovery path as any other failed call.
    """

    retryable = False


def build_registry() -> dict[ToolName, BaseTool[Any, Any]]:
    """Instantiate every productivity tool.

    A fresh registry per run keeps tools free of cross-run state, which matters because
    tool instances are shared across every turn of a session.
    """
    tools: list[BaseTool[Any, Any]] = [
        SearchDocumentsTool(),
        ExtractRequirementsTool(),
        CalculateBudgetTool(),
        CreateProjectPlanTool(),
        UpdateProjectPlanTool(),
        DraftExecutiveSummaryTool(),
        DraftStakeholderEmailTool(),
        SaveArtifactTool(),
    ]
    return {tool.name: tool for tool in tools}


class ToolRegistry:
    """Lookup surface over the available tools."""

    def __init__(self, tools: dict[ToolName, BaseTool[Any, Any]] | None = None) -> None:
        """Build a registry, defaulting to the full production tool set."""
        self._tools = tools if tools is not None else build_registry()

    def get(self, name: ToolName | str) -> BaseTool[Any, Any]:
        """Resolve a tool by name.

        Raises:
            UnknownToolError: If no such tool is registered.
        """
        try:
            key = ToolName(name)
        except ValueError as exc:
            available = ", ".join(sorted(t.value for t in self._tools))
            raise UnknownToolError(f"Unknown tool {name!r}. Available tools: {available}") from exc
        if key not in self._tools:
            available = ", ".join(sorted(t.value for t in self._tools))
            raise UnknownToolError(f"Tool {key.value!r} is not registered. Available: {available}")
        return self._tools[key]

    def names(self) -> list[ToolName]:
        """Every registered tool name, in enum order."""
        return [name for name in ToolName if name in self._tools]

    def schemas(self) -> list[dict[str, Any]]:
        """Function-calling schemas for every registered tool."""
        return [self._tools[name].schema() for name in self.names()]

    def requires_approval(self, name: ToolName | str) -> bool:
        """Whether the named tool needs prior user approval."""
        return self.get(name).requires_approval

    def __contains__(self, name: object) -> bool:
        """Whether a tool name is registered."""
        if isinstance(name, ToolName):
            return name in self._tools
        if isinstance(name, str):
            try:
                return ToolName(name) in self._tools
            except ValueError:
                return False
        return False

    def __len__(self) -> int:
        """Number of registered tools."""
        return len(self._tools)
