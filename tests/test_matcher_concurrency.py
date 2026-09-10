"""Regression tests for the concurrent matcher start-up path.

``EntityMatcher.resolve_blocks`` fans blocks out over a thread pool. Before
``warm_up`` existed, the LM and the predictor were built on first use inside those
threads, so every worker built its own LM, minted its own Vertex access token, and
took DSPy's first litellm touch at the same time as its neighbours. Whichever
thread lost that race saw ``partially initialized module 'litellm' has no
attribute 'completion'`` and its whole block fell through to error recovery.
"""

import asyncio
import threading
import time
from typing import Any
from unittest.mock import patch

import dspy
import litellm
import pytest
from dspy.clients._litellm import get_litellm

from serf.dspy.types import BlockResolution, Entity, EntityBlock, MatchDecision
from serf.match.matcher import EntityMatcher
from serf.match.run import ERROR_RECOVERY_REASON, collect_pairs

BLOCK_COUNT = 24
CONCURRENCY = 12
BUILD_SECONDS = 0.05


def _blocks(count: int = BLOCK_COUNT) -> list[EntityBlock]:
    """Build blocks holding one obviously matching pair each.

    Parameters
    ----------
    count : int
        Number of blocks to build

    Returns
    -------
    list[EntityBlock]
        Blocks of two entities each
    """
    return [
        EntityBlock(
            block_key=f"block_{i}",
            block_size=2,
            entities=[
                Entity(id=i * 2, name=f"widget {i}", entity_type="product"),
                Entity(id=i * 2 + 1, name=f"widget {i}", entity_type="product"),
            ],
        )
        for i in range(count)
    ]


class _CountingPredict:
    """Stub predictor that records the threads it ran on."""

    def __init__(self) -> None:
        self.threads: set[int] = set()
        self.calls = 0

    def __call__(self, **_kwargs: Any) -> dspy.Prediction:
        self.threads.add(threading.get_ident())
        self.calls += 1
        # Hold the worker thread long enough that the pool really overlaps.
        time.sleep(0.02)
        return dspy.Prediction(
            resolution=BlockResolution(
                block_key="stub",
                matches=[
                    MatchDecision(
                        entity_a_id=0,
                        entity_b_id=1,
                        is_match=True,
                        confidence=0.99,
                        reasoning="stub",
                    )
                ],
                resolved_entities=[],
                was_resolved=True,
                original_count=2,
                resolved_count=1,
            )
        )


@pytest.fixture
def matcher() -> EntityMatcher:
    """Matcher wired to a stub predictor so no signature work happens."""
    instance = EntityMatcher(max_concurrent=CONCURRENCY)
    instance._predictor = _CountingPredict()  # type: ignore[assignment]
    return instance


def test_resolve_blocks_builds_the_lm_once_for_the_whole_thread_pool(
    matcher: EntityMatcher,
) -> None:
    """Every worker thread shares one LM instead of building its own."""
    created: list[object] = []

    def fake_create_lm(*_args: Any, **_kwargs: Any) -> object:
        # Building the real LM refreshes a Vertex access token over the network,
        # which is slow enough for every thread to enter the unguarded lazy check.
        time.sleep(BUILD_SECONDS)
        created.append(object())
        return created[-1]

    with (
        patch("serf.match.matcher.create_lm", side_effect=fake_create_lm),
        patch.object(dspy, "context"),
    ):
        resolutions = asyncio.run(matcher.resolve_blocks(_blocks()))

    assert len(created) == 1
    assert len(resolutions) == BLOCK_COUNT


def test_resolve_blocks_builds_the_predictor_once_for_the_whole_thread_pool() -> None:
    """The predictor is constructed before the pool starts, not per thread."""
    instance = EntityMatcher(max_concurrent=CONCURRENCY)
    built = 0

    def fake_predict(_signature: Any) -> _CountingPredict:
        nonlocal built
        time.sleep(BUILD_SECONDS)
        built += 1
        return _CountingPredict()

    with (
        patch("serf.match.matcher.create_lm", return_value=object()),
        patch.object(dspy, "Predict", side_effect=fake_predict),
        patch.object(dspy, "context"),
    ):
        asyncio.run(instance.resolve_blocks(_blocks()))

    assert built == 1


def test_concurrent_blocks_reach_litellm_without_falling_into_error_recovery(
    matcher: EntityMatcher,
) -> None:
    """Threads that touch litellm concurrently all resolve their block.

    The stub predictor reaches litellm through DSPy's lazy loader, the same way a
    real completion does, with ``litellm.completion`` mocked so nothing leaves the
    process. A thread that saw a half-built module here would log an LLM failure
    and its block would come back marked ``error_recovery``.
    """
    stub = _CountingPredict()
    original_call = stub.__call__

    def touch_litellm(**kwargs: Any) -> dspy.Prediction:
        get_litellm(feature="dspy.LM").completion(model="stub", messages=[])
        return original_call(**kwargs)

    matcher._predictor = touch_litellm  # type: ignore[assignment]

    with (
        patch("serf.match.matcher.create_lm", return_value=object()),
        patch.object(litellm, "completion", return_value=None) as completion,
        patch.object(dspy, "context"),
    ):
        resolutions = asyncio.run(matcher.resolve_blocks(_blocks()))

    outcome = collect_pairs(resolutions)
    assert completion.call_count == BLOCK_COUNT
    assert outcome.failed_blocks == 0
    assert stub.calls == BLOCK_COUNT
    assert len(stub.threads) > 1
    assert not any(
        e.match_skip_reason == ERROR_RECOVERY_REASON
        for resolution in resolutions
        for e in resolution.resolved_entities
    )
