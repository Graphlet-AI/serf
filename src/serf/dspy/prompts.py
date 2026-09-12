"""Render the DSPy signatures and prompts the matcher sends, before any tuning.

A signature's docstring is its prompt. DSPy copies it into
``Signature.instructions``, the adapter renders it into the system message, and
nothing else in the pipeline writes matching instructions, so reading these
reports is reading what the model is actually told.

What ``dspy.Predict`` sends is more than the docstring: the adapter also emits
the field list, the nested XML skeleton the answer has to fill in, and the input
records themselves. Those three are what GEPA does *not* rewrite, so seeing them
next to the instructions shows how much of a prompt is fixed by the signature's
types and how much is available to optimize.
"""

import re
from dataclasses import dataclass
from typing import Any, get_args

import dspy

from serf.dspy.adapter import RepairingXMLAdapter
from serf.dspy.dataset_signatures import (
    DATASET_SIGNATURES,
    SIGNATURE_MODE_GENERIC,
    SIGNATURE_MODE_PER_DATASET,
    DatasetSignatureSpec,
    get_dataset_spec,
)
from serf.dspy.schemas.base import EntitySide
from serf.dspy.signatures import BlockMatch
from serf.logs import get_logger
from serf.match.few_shot import get_default_few_shot_examples
from serf.match.matcher import SCHEMA_INFO

logger = get_logger(__name__)

PLACEHOLDER_BLOCK_RECORDS = (
    '[{"id": 1, "name": "<record 1 name>", "attributes": {"<column>": "<value>"}},\n'
    ' {"id": 2, "name": "<record 2 name>", "attributes": {"<column>": "<value>"}}]'
)


@dataclass(frozen=True)
class FieldReport:
    """One input or output field of a signature.

    Parameters
    ----------
    name : str
        Field name as the adapter writes it into the prompt
    annotation : str
        Rendered Python annotation
    description : str
        Field description, empty when the signature declares none
    """

    name: str
    annotation: str
    description: str


@dataclass(frozen=True)
class PromptReport:
    """Everything one signature contributes to one matching call.

    Parameters
    ----------
    dataset : str
        Benchmark dataset, or ``shared`` for the generic signature
    signature_mode : str
        ``generic`` or ``per-dataset``
    signature_name : str
        Signature class name
    instructions : str
        Signature instructions, which are its docstring
    input_fields : list[FieldReport]
        Declared input fields
    output_fields : list[FieldReport]
        Declared output fields
    system_prompt : str
        System message the adapter builds from the signature alone
    user_prompt : str
        User message, holding the input records and the output format demand
    records_are_real : bool
        True when the user message was rendered from benchmark records rather
        than from placeholders
    """

    dataset: str
    signature_mode: str
    signature_name: str
    instructions: str
    input_fields: list[FieldReport]
    output_fields: list[FieldReport]
    system_prompt: str
    user_prompt: str
    records_are_real: bool

    @property
    def prompt_characters(self) -> int:
        """Total characters in the two rendered messages.

        Returns
        -------
        int
            Length of the system message plus the user message
        """
        return len(self.system_prompt) + len(self.user_prompt)

    @property
    def instruction_characters(self) -> int:
        """Characters GEPA is free to rewrite.

        Returns
        -------
        int
            Length of the instructions
        """
        return len(self.instructions)


def _annotation_name(annotation: Any) -> str:
    """Render a field annotation the way a reader would write it.

    Parameters
    ----------
    annotation : Any
        Field annotation from a DSPy signature

    Returns
    -------
    str
        Short type name, with module paths and ``typing.`` prefixes dropped
    """
    # A parameterized generic answers __name__ with just "list", which would
    # hide the element type that decides what the adapter renders.
    if get_args(annotation):
        text = str(annotation)
    else:
        text = getattr(annotation, "__name__", None) or str(annotation)
    text = re.sub(r"\b[\w.]+\.(\w+)", r"\1", text)
    return text.replace("typing.", "")


def _field_reports(fields: dict[str, Any]) -> list[FieldReport]:
    """Describe a signature's fields.

    Parameters
    ----------
    fields : dict[str, Any]
        ``input_fields`` or ``output_fields`` of a signature

    Returns
    -------
    list[FieldReport]
        One report per field, in declaration order
    """
    reports: list[FieldReport] = []
    for name, field in fields.items():
        extra: dict[str, Any] = field.json_schema_extra or {}
        description = str(extra.get("desc", "") or "")
        if description.startswith("${"):
            description = ""
        reports.append(
            FieldReport(
                name=name,
                annotation=_annotation_name(field.annotation),
                description=" ".join(description.split()),
            )
        )
    return reports


def placeholder_records(side: type[EntitySide], record_id: int) -> EntitySide:
    """Build one self-labeling record for a side of a match task.

    Every text column is set to its own name in angle brackets, so the rendered
    prompt shows where each source column lands without inventing data that
    could be mistaken for a real record. Numeric columns parse the placeholder as
    missing and come back null, which is also what a blank source cell does.

    Parameters
    ----------
    side : type[EntitySide]
        Typed side model to instantiate
    record_id : int
        Record id to assign

    Returns
    -------
    EntitySide
        Placeholder record
    """
    values: dict[str, Any] = {"record_id": record_id, "source_id": f"<{side.__name__} id>"}
    for column in side.source_columns():
        values[column] = f"<{column}>"
    return side.model_validate(values)


def dataset_prompt_report(
    dataset: str,
    left_records: list[Any] | None = None,
    right_records: list[Any] | None = None,
) -> PromptReport:
    """Render the signature and prompt for one benchmark dataset.

    Parameters
    ----------
    dataset : str
        Benchmark dataset with a per-dataset signature
    left_records : list[Any] | None
        Real left-side records to render. Placeholders are used when omitted.
    right_records : list[Any] | None
        Real right-side records to render. Placeholders are used when omitted.

    Returns
    -------
    PromptReport
        Signature fields, instructions, and the two rendered messages
    """
    spec: DatasetSignatureSpec = get_dataset_spec(dataset)
    real = bool(left_records) and bool(right_records)
    left = left_records or [placeholder_records(spec.left_type, 1)]
    right = right_records or [placeholder_records(spec.right_type, 2)]
    messages = RepairingXMLAdapter().format(
        spec.signature,
        [],
        {spec.left_field: left, spec.right_field: right},
    )
    return PromptReport(
        dataset=dataset,
        signature_mode=SIGNATURE_MODE_PER_DATASET,
        signature_name=spec.signature.__name__,
        instructions=spec.signature.instructions,
        input_fields=_field_reports(spec.signature.input_fields),
        output_fields=_field_reports(spec.signature.output_fields),
        system_prompt=_message(messages, "system"),
        user_prompt=_message(messages, "user"),
        records_are_real=real,
    )


def generic_prompt_report(block_records: str | None = None) -> PromptReport:
    """Render the signature and prompt of the shared ``BlockMatch`` signature.

    Parameters
    ----------
    block_records : str | None
        JSON array of records to render. A placeholder is used when omitted.

    Returns
    -------
    PromptReport
        Signature fields, instructions, and the two rendered messages
    """
    records = block_records or PLACEHOLDER_BLOCK_RECORDS
    messages = RepairingXMLAdapter().format(
        BlockMatch,
        [],
        {
            "block_records": records,
            "schema_info": SCHEMA_INFO,
            "few_shot_examples": get_default_few_shot_examples(),
        },
    )
    return PromptReport(
        dataset="shared",
        signature_mode=SIGNATURE_MODE_GENERIC,
        signature_name=BlockMatch.__name__,
        instructions=BlockMatch.instructions,
        input_fields=_field_reports(BlockMatch.input_fields),
        output_fields=_field_reports(BlockMatch.output_fields),
        system_prompt=_message(messages, "system"),
        user_prompt=_message(messages, "user"),
        records_are_real=block_records is not None,
    )


def _message(messages: list[dict[str, Any]], role: str) -> str:
    """Pull one role's content out of a rendered chat message list.

    Parameters
    ----------
    messages : list[dict[str, Any]]
        Messages returned by an adapter's ``format``
    role : str
        Role to extract

    Returns
    -------
    str
        Concatenated content for that role
    """
    return "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == role)


def all_prompt_reports(
    signature_mode: str = SIGNATURE_MODE_PER_DATASET,
    datasets: list[str] | None = None,
) -> list[PromptReport]:
    """Render every signature for a mode.

    Parameters
    ----------
    signature_mode : str
        ``generic`` for the one shared signature, ``per-dataset`` for all five
    datasets : list[str] | None
        Restrict the per-dataset reports to these datasets

    Returns
    -------
    list[PromptReport]
        One report per signature
    """
    if signature_mode == SIGNATURE_MODE_GENERIC:
        return [generic_prompt_report()]
    names = datasets or list(DATASET_SIGNATURES)
    return [dataset_prompt_report(name) for name in names]


def render_report(report: PromptReport, include_prompt: bool = True) -> str:
    """Format one report as Markdown.

    Parameters
    ----------
    report : PromptReport
        Report to format
    include_prompt : bool
        Include the two rendered messages, not just the instructions

    Returns
    -------
    str
        Markdown section describing the signature and its prompt
    """
    lines: list[str] = [
        f"## {report.dataset} - {report.signature_name}",
        "",
        f"- Signature mode: `{report.signature_mode}`",
        f"- Instructions (what GEPA rewrites): {report.instruction_characters:,} characters",
        f"- Full prompt as sent: {report.prompt_characters:,} characters",
        "",
        "### Input fields",
        "",
    ]
    for field in report.input_fields:
        lines.append(f"- `{field.name}: {field.annotation}` - {field.description}")
    lines += ["", "### Output fields", ""]
    for field in report.output_fields:
        lines.append(f"- `{field.name}: {field.annotation}` - {field.description}")
    lines += ["", "### Instructions", "", "```text", report.instructions.strip(), "```", ""]
    if include_prompt:
        source = "benchmark records" if report.records_are_real else "placeholder records"
        lines += [
            "### System message",
            "",
            "```text",
            report.system_prompt.strip(),
            "```",
            "",
            f"### User message ({source})",
            "",
            "```text",
            report.user_prompt.strip(),
            "```",
            "",
        ]
    return "\n".join(lines)


def render_reports(reports: list[PromptReport], include_prompt: bool = True) -> str:
    """Format a set of reports as one Markdown document.

    Parameters
    ----------
    reports : list[PromptReport]
        Reports to format
    include_prompt : bool
        Include the rendered messages for each report

    Returns
    -------
    str
        Markdown document with a summary table and one section per signature
    """
    header = [
        "# SERF matching signatures and prompts",
        "",
        f"DSPy {dspy.__version__}. Instructions are the signature docstrings, which is the text",
        "GEPA rewrites; the rest of each prompt is fixed by the signature's declared types.",
        "",
        "| Signature | Dataset | Instruction chars | Prompt chars |",
        "| --- | --- | --- | --- |",
    ]
    for report in reports:
        header.append(
            f"| `{report.signature_name}` | {report.dataset} | "
            f"{report.instruction_characters:,} | {report.prompt_characters:,} |"
        )
    header.append("")
    body = [render_report(report, include_prompt=include_prompt) for report in reports]
    return "\n".join(header + body)
