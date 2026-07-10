"""
Base class for all tools.
"""

from abc import ABC, abstractmethod

from state.pipeline_state import PipelineState


class BaseTool(ABC):
    """
    Abstract base class for every tool.
    """

    @abstractmethod
    def execute(self, state: PipelineState, config):
        """
        Execute the tool and return the updated PipelineState.
        """
        pass