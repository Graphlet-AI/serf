from serf.dspy.agents import (
    ER_TOOLS,
    ERAgent,
    ERControlSignature,
    check_convergence,
    create_blocks,
    evaluate_matches,
    match_blocks,
    profile_dataset,
)


def test_er_agent_initialization() -> None:
    """Test ERAgent initializes with default parameters."""
    agent = ERAgent()
    assert agent.max_iterations == 5
    assert agent.convergence_threshold == 0.01
    assert agent.react is not None


def test_er_agent_initialization_custom_params() -> None:
    """Test ERAgent initializes with custom parameters."""
    agent = ERAgent(max_iterations=10, convergence_threshold=0.05)
    assert agent.max_iterations == 10
    assert agent.convergence_threshold == 0.05


def test_er_control_signature_has_correct_fields() -> None:
    """Test ERControlSignature has expected input and output fields."""
    expected_fields = [
        "dataset_description",
        "entity_type",
        "max_iterations",
        "convergence_threshold",
        "action_plan",
    ]
    for field in expected_fields:
        assert field in ERControlSignature.model_fields


def test_check_convergence_converged() -> None:
    """Test check_convergence returns CONVERGED when below threshold."""
    result = check_convergence(reduction_pct=0.005, threshold=0.01)
    assert "CONVERGED" in result
    assert "Stop iterating" in result


def test_check_convergence_not_converged() -> None:
    """Test check_convergence returns NOT CONVERGED when above threshold."""
    result = check_convergence(reduction_pct=0.05, threshold=0.01)
    assert "NOT CONVERGED" in result
    assert "Continue" in result


def test_check_convergence_default_threshold() -> None:
    """Test check_convergence uses default threshold of 0.01."""
    result_converged = check_convergence(reduction_pct=0.005)
    assert "CONVERGED" in result_converged

    result_not_converged = check_convergence(reduction_pct=0.02)
    assert "NOT CONVERGED" in result_not_converged


def test_er_tools_contains_expected_tools() -> None:
    """Test ER_TOOLS contains all expected tool functions."""
    expected = [profile_dataset, create_blocks, match_blocks, evaluate_matches, check_convergence]
    assert expected == ER_TOOLS
    assert len(ER_TOOLS) == 5


def test_er_agent_with_custom_tools() -> None:
    """Test ERAgent can be created with custom tools override."""

    def dummy_tool(x: str) -> str:
        return f"dummy: {x}"

    custom_tools = [dummy_tool]
    agent = ERAgent(tools=custom_tools)
    assert agent.react is not None
    assert "dummy_tool" in agent.react.tools


def test_profile_dataset_tool() -> None:
    """Test profile_dataset tool returns expected string."""
    result = profile_dataset("/path/to/data.parquet")
    assert "Dataset at /path/to/data.parquet profiled" in result


def test_create_blocks_tool() -> None:
    """Test create_blocks tool returns expected string."""
    result = create_blocks("/input", method="semantic", target_block_size=50)
    assert "Created blocks from /input" in result
    assert "semantic" in result
    assert "50" in result


def test_match_blocks_tool() -> None:
    """Test match_blocks tool returns expected string."""
    result = match_blocks("/blocks", iteration=2)
    assert "Matched blocks from /blocks" in result
    assert "iteration 2" in result


def test_evaluate_matches_tool() -> None:
    """Test evaluate_matches tool returns expected string."""
    result = evaluate_matches("/matches", "/raw")
    assert "Evaluated matches from /matches" in result
    assert "/raw" in result
