"""
Model Selection Tool.

Applies a ModelSelectionOutput (produced by either the LLM or the
heuristic fallback) to the PipelineState by populating the
ModelSelectionState sub-state.

This tool is a pure writer — it does not make any decisions itself.
All decision logic lives in the ModelSelectionAgent and the LLM prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from schemas.model_selection_schema import ModelSelectionOutput
from tools.base_tool import BaseTool

if TYPE_CHECKING:
    from state.pipeline_state import PipelineState


class ModelSelectionTool(BaseTool):
    """
    Writes the structured model selection result into PipelineState.

    The execute() method is intentionally narrow: it only transfers
    data from ModelSelectionOutput → ModelSelectionState. It performs
    no analysis, no LLM calls, and no IO.
    """

    def execute(
        self,
        state: "PipelineState",
        output: ModelSelectionOutput,
        dataset_profile_dict: dict | None = None,
    ) -> "PipelineState":
        """
        Populate ModelSelectionState from a ModelSelectionOutput.

        Parameters
        ----------
        state : PipelineState
            The shared pipeline state to update.
        output : ModelSelectionOutput
            Structured result from the LLM or heuristic fallback.
        dataset_profile_dict : dict | None
            Serialised DatasetProfile to store for downstream inspection.

        Returns
        -------
        PipelineState
            Updated state with model_selection fully populated.
        """
        ms = state.model_selection

        # Task type
        ms.task_type = output.task_type

        # Primary model
        ms.primary_model_name = output.primary_model.name
        ms.primary_model_class_path = output.primary_model.class_path
        ms.primary_model_library = output.primary_model.library

        # Candidates — stored as serialised dicts so the state remains
        # JSON-serialisable without importing the schema elsewhere
        ms.candidate_models = [
            candidate.model_dump()
            for candidate in output.candidate_models
        ]

        # Ranking (model names ordered best → worst)
        sorted_candidates = sorted(
            output.candidate_models,
            key=lambda c: c.rank,
        )
        ms.ranking = [c.name for c in sorted_candidates]

        # Ranking criteria
        ms.ranking_criteria = output.ranking_criteria

        # Reasoning and assumptions
        ms.reasoning = output.reasoning
        ms.assumptions = output.assumptions

        # Confidence
        ms.confidence = output.confidence

        # Dataset profile snapshot
        if dataset_profile_dict is not None:
            ms.dataset_profile = dataset_profile_dict

        # Mark as completed
        ms.is_completed = True

        # Build a concise summary string for display / logging
        ms.summary = (
            f"Task: {output.task_type} | "
            f"Primary model: {output.primary_model.name} | "
            f"Candidates: {len(output.candidate_models)} | "
            f"Confidence: {output.confidence:.2f}"
        )

        state.model_selection = ms
        return state
