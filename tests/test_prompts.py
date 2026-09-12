"""Tests for rendering the signatures and prompts the matcher sends."""

import pytest

from serf.dspy.dataset_signatures import (
    DATASET_SIGNATURES,
    SIGNATURE_MODE_GENERIC,
    SIGNATURE_MODE_PER_DATASET,
    AcmPublication,
    DblpPublication,
    get_dataset_spec,
)
from serf.dspy.prompts import (
    all_prompt_reports,
    dataset_prompt_report,
    generic_prompt_report,
    placeholder_records,
    render_report,
    render_reports,
)

DATASETS = sorted(DATASET_SIGNATURES)


@pytest.mark.parametrize("dataset", DATASETS)
def test_report_instructions_are_the_signature_docstring(dataset: str) -> None:
    """The instructions a report shows are the text DSPy sends, not a paraphrase."""
    spec = get_dataset_spec(dataset)
    report = dataset_prompt_report(dataset)
    assert report.instructions == spec.signature.instructions
    assert report.signature_name == spec.signature.__name__
    assert report.signature_mode == SIGNATURE_MODE_PER_DATASET


@pytest.mark.parametrize("dataset", DATASETS)
def test_report_names_both_typed_sides_and_the_candidate_output(dataset: str) -> None:
    """Field reports keep the element type, which is what the adapter renders."""
    spec = get_dataset_spec(dataset)
    report = dataset_prompt_report(dataset)
    inputs = {field.name: field.annotation for field in report.input_fields}
    assert inputs[spec.left_field] == f"list[{spec.left_type.__name__}]"
    assert inputs[spec.right_field] == f"list[{spec.right_type.__name__}]"
    outputs = {field.name: field.annotation for field in report.output_fields}
    assert outputs[spec.candidates_field] == f"list[{spec.candidate_type.__name__}]"


@pytest.mark.parametrize("dataset", DATASETS)
def test_system_prompt_carries_the_instructions(dataset: str) -> None:
    """The rendered system message is where the docstring actually lands."""
    report = dataset_prompt_report(dataset)
    first_line = report.instructions.strip().splitlines()[0]
    assert first_line in report.system_prompt
    assert report.prompt_characters > report.instruction_characters


@pytest.mark.parametrize("dataset", DATASETS)
def test_user_prompt_holds_both_input_fields_and_the_output_skeleton(dataset: str) -> None:
    """The user message is the records plus the nested XML the answer must fill."""
    spec = get_dataset_spec(dataset)
    report = dataset_prompt_report(dataset)
    assert f"<{spec.left_field}>" in report.user_prompt
    assert f"<{spec.right_field}>" in report.user_prompt
    assert f"<{spec.candidates_field}>" in report.user_prompt
    assert "<record_id>" in report.user_prompt


@pytest.mark.parametrize("dataset", DATASETS)
def test_placeholder_records_are_labeled_as_placeholders(dataset: str) -> None:
    """Rendering without real data must not look like it used real data."""
    spec = get_dataset_spec(dataset)
    report = dataset_prompt_report(dataset)
    assert report.records_are_real is False
    for column in spec.left_type.source_columns():
        placeholder = f"<{column}>"
        if isinstance(getattr(placeholder_records(spec.left_type, 1), column), str):
            assert placeholder in report.user_prompt


def test_real_records_are_reported_as_real() -> None:
    """Passing records in flips the flag the report prints under the user message."""
    left = DblpPublication(record_id=1, source_id="conf/vldb/96", title="Mediator Languages")
    right = AcmPublication(record_id=2, source_id="673456", title="Mediator languages")
    report = dataset_prompt_report("dblp-acm", [left], [right])
    assert report.records_are_real is True
    assert "Mediator Languages" in report.user_prompt


def test_generic_report_describes_the_shared_signature() -> None:
    """The generic signature is the one GEPA has far less text to work with."""
    report = generic_prompt_report()
    assert report.signature_mode == SIGNATURE_MODE_GENERIC
    assert report.signature_name == "BlockMatch"
    assert report.dataset == "shared"
    assert "block_records" in report.user_prompt


def test_the_shared_signature_says_far_less_than_any_per_dataset_one() -> None:
    """The instruction budget is the reason per-dataset signatures exist at all."""
    generic = generic_prompt_report()
    for dataset in DATASETS:
        assert dataset_prompt_report(dataset).instruction_characters > (
            generic.instruction_characters
        )


def test_all_reports_covers_every_dataset() -> None:
    """`serf prompts` with no --dataset has to show all five."""
    reports = all_prompt_reports(signature_mode=SIGNATURE_MODE_PER_DATASET)
    assert {report.dataset for report in reports} == set(DATASETS)
    assert len(all_prompt_reports(signature_mode=SIGNATURE_MODE_GENERIC)) == 1


def test_all_reports_can_be_restricted_to_one_dataset() -> None:
    """--dataset narrows the report set."""
    reports = all_prompt_reports(datasets=["abt-buy"])
    assert [report.dataset for report in reports] == ["abt-buy"]


def test_render_report_can_omit_the_rendered_messages() -> None:
    """--instructions-only drops the messages and keeps the instructions."""
    report = dataset_prompt_report("abt-buy")
    short = render_report(report, include_prompt=False)
    full = render_report(report, include_prompt=True)
    assert "### System message" not in short
    assert "### System message" in full
    assert "### Instructions" in short
    assert len(full) > len(short)


def test_render_reports_opens_with_a_summary_table() -> None:
    """The document leads with the number that matters, instruction size."""
    document = render_reports(all_prompt_reports(datasets=["dblp-acm", "abt-buy"]))
    assert "| Signature | Dataset | Instruction chars | Prompt chars |" in document
    assert "`DblpAcmBlockMatch`" in document
    assert "`AbtBuyBlockMatch`" in document
