"""Read back the matching prompts ``serf train`` wrote.

A trained prompt is a saved DSPy program: the same signature with the
instructions GEPA ended on. Keeping the path convention and the load in their
own module lets the matcher use a trained prompt without importing the training
pipeline, which imports the matcher.
"""

from pathlib import Path
from typing import cast

import dspy

from serf.config import config
from serf.dspy.dataset_signatures import get_dataset_spec
from serf.logs import get_logger

logger = get_logger(__name__)

TRAINED_PROGRAM_SUFFIX = "_gepa.json"


def trained_program_path(dataset: str, directory: str | None = None) -> Path:
    """Return where a dataset's trained program lives.

    Parameters
    ----------
    dataset : str
        Benchmark dataset name
    directory : str | None
        Directory holding trained programs. Defaults to config
        ``optimize.trained_dir``.

    Returns
    -------
    Path
        Path to the dataset's trained DSPy program, which need not exist
    """
    root = directory or str(config.get("optimize.trained_dir", "data/trained_prompts"))
    return Path(root) / f"{dataset}{TRAINED_PROGRAM_SUFFIX}"


def load_trained_predictor(dataset: str, directory: str | None = None) -> dspy.Predict | None:
    """Load a dataset's trained predictor, if one has been written.

    Parameters
    ----------
    dataset : str
        Benchmark dataset name
    directory : str | None
        Directory holding trained programs. Defaults to config
        ``optimize.trained_dir``.

    Returns
    -------
    dspy.Predict | None
        Predictor carrying the trained instructions, or None when no program
        exists for this dataset
    """
    path = trained_program_path(dataset, directory)
    if not path.exists():
        logger.warning(
            f"No trained prompt for {dataset} at {path}; matching with the signature as written"
        )
        return None
    spec = get_dataset_spec(dataset)
    predictor = cast(dspy.Predict, dspy.Predict(spec.signature))
    predictor.load(str(path))
    logger.info(
        f"Loaded trained prompt for {dataset} from {path}: "
        f"{len(predictor_instructions(predictor))} instruction characters against "
        f"{len(spec.signature.instructions)} as written"
    )
    return predictor


def predictor_instructions(predictor: dspy.Predict) -> str:
    """Return a predictor's current instruction text.

    ``Predict.signature`` is optional on the DSPy side, so reading it needs a
    fallback even though a predictor built from a signature always has one.

    Parameters
    ----------
    predictor : dspy.Predict
        Predictor to read

    Returns
    -------
    str
        Instruction text, empty when the predictor carries no signature
    """
    signature = getattr(predictor, "signature", None)
    return str(getattr(signature, "instructions", "") or "")


def trained_instructions(dataset: str, directory: str | None = None) -> str | None:
    """Return the trained instructions for a dataset, if any were written.

    Parameters
    ----------
    dataset : str
        Benchmark dataset name
    directory : str | None
        Directory holding trained programs

    Returns
    -------
    str | None
        Trained instruction text, or None when no program exists
    """
    predictor = load_trained_predictor(dataset, directory)
    return None if predictor is None else predictor_instructions(predictor)
