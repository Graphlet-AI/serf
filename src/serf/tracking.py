"""MLflow tracking and DSPy tracing integration for SERF."""

import mlflow

from serf.config import config
from serf.logs import get_logger

logger = get_logger(__name__)

_initialized = False


def setup_mlflow() -> None:
    """Configure MLflow tracking and enable DSPy autologging.

    Reads configuration from config.yml mlflow section. Sets tracking URI,
    experiment name, and enables mlflow.dspy.autolog() for automatic tracing
    of all DSPy module invocations.

    Safe to call multiple times -- only initializes once.
    """
    global _initialized
    if _initialized:
        return

    tracking_uri = config.get("mlflow.tracking_uri", "http://127.0.0.1:5000")
    experiment_name = config.get("mlflow.experiment_name", "SERF-Entity-Resolution")
    log_traces = config.get("mlflow.autolog.log_traces", True)
    log_traces_from_compile = config.get("mlflow.autolog.log_traces_from_compile", False)
    log_traces_from_eval = config.get("mlflow.autolog.log_traces_from_eval", True)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    mlflow.dspy.autolog(
        log_traces=log_traces,
        log_traces_from_compile=log_traces_from_compile,
        log_traces_from_eval=log_traces_from_eval,
    )

    _initialized = True
    logger.info("MLflow tracking enabled: uri=%s, experiment=%s", tracking_uri, experiment_name)
