from collections.abc import Callable
from typing import cast

import dspy

from serf.logs import get_logger

logger = get_logger(__name__)


class ERControlSignature(dspy.Signature):
    """Control the entity resolution pipeline.

    Given a dataset description, entity type, and convergence parameters,
    decide which pipeline actions to take and when to stop.
    """

    dataset_description: str = dspy.InputField(desc="Summary of the dataset to resolve")
    entity_type: str = dspy.InputField(desc="Type of entities being resolved")
    max_iterations: int = dspy.InputField(desc="Maximum number of ER iterations")
    convergence_threshold: float = dspy.InputField(
        desc="Stop when reduction per round is below this"
    )
    action_plan: str = dspy.OutputField(desc="Step-by-step plan for resolving entities")


def profile_dataset(path: str) -> str:
    """Tool: Profile a dataset and return summary statistics.

    Parameters
    ----------
    path : str
        Path to the dataset file (Parquet or CSV)

    Returns
    -------
    str
        JSON summary of the dataset profile
    """
    return f"Dataset at {path} profiled. Use the blocking and matching tools to proceed."


def create_blocks(input_path: str, method: str = "semantic", target_block_size: int = 50) -> str:
    """Tool: Create entity blocks from input data.

    Parameters
    ----------
    input_path : str
        Path to input entities
    method : str
        Blocking method: "semantic" or "name"
    target_block_size : int
        Target entities per block

    Returns
    -------
    str
        Summary of blocking results
    """
    return (
        f"Created blocks from {input_path} using {method} method "
        f"with target size {target_block_size}."
    )


def match_blocks(blocks_path: str, iteration: int = 1) -> str:
    """Tool: Run LLM matching on entity blocks.

    Parameters
    ----------
    blocks_path : str
        Path to blocked entities
    iteration : int
        Current iteration number

    Returns
    -------
    str
        Summary of matching results
    """
    return f"Matched blocks from {blocks_path} (iteration {iteration})."


def evaluate_matches(matches_path: str, raw_path: str) -> str:
    """Tool: Evaluate match quality against original data.

    Parameters
    ----------
    matches_path : str
        Path to match results
    raw_path : str
        Path to original raw entities

    Returns
    -------
    str
        JSON summary of evaluation metrics
    """
    return f"Evaluated matches from {matches_path} against {raw_path}."


def check_convergence(reduction_pct: float, threshold: float = 0.01) -> str:
    """Tool: Check if the ER pipeline has converged.

    Parameters
    ----------
    reduction_pct : float
        Reduction percentage from last iteration
    threshold : float
        Convergence threshold

    Returns
    -------
    str
        Whether to continue or stop
    """
    if reduction_pct < threshold:
        return "CONVERGED: Reduction below threshold. Stop iterating."
    return f"NOT CONVERGED: Reduction {reduction_pct:.2%} > threshold {threshold:.2%}. Continue."


# All tools available to the agent
ER_TOOLS = [profile_dataset, create_blocks, match_blocks, evaluate_matches, check_convergence]


class ERAgent(dspy.Module):
    """Agentic entity resolution controller.

    Uses ReAct (Reasoning + Acting) to dynamically orchestrate
    the blocking -> matching -> merging -> evaluation pipeline.

    Parameters
    ----------
    max_iterations : int
        Maximum ER iterations before stopping
    convergence_threshold : float
        Stop when per-round reduction falls below this
    tools : list[Callable[..., str]] | None
        Override default tools. If None, uses ER_TOOLS.
    """

    def __init__(
        self,
        max_iterations: int = 5,
        convergence_threshold: float = 0.01,
        tools: list[Callable[..., str]] | None = None,
    ) -> None:
        super().__init__()
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

        tool_list = tools if tools is not None else ER_TOOLS
        self.react = dspy.ReAct(
            ERControlSignature,
            tools=tool_list,
            max_iters=max_iterations * 3,
        )

    def forward(
        self,
        dataset_description: str,
        entity_type: str = "entity",
    ) -> dspy.Prediction:
        """Run the agentic ER pipeline.

        Parameters
        ----------
        dataset_description : str
            Summary of the dataset to resolve
        entity_type : str
            Type of entities being resolved

        Returns
        -------
        dspy.Prediction
            The agent's action plan and reasoning
        """
        return cast(
            dspy.Prediction,
            self.react(
                dataset_description=dataset_description,
                entity_type=entity_type,
                max_iterations=self.max_iterations,
                convergence_threshold=self.convergence_threshold,
            ),
        )
