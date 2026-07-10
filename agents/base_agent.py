"""
Base class for all AI Agents.
"""

from abc import ABC, abstractmethod

from state.pipeline_state import PipelineState


class BaseAgent(ABC):
    """
    Abstract base class for every AI agent.
    """

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        """
        Execute the agent and return the updated PipelineState.
        """
        pass