"""Typed record schemas for the DBLP-Google Scholar bibliographic match task.

Source: Köpcke, Thor and Rahm, "Evaluation of entity resolution approaches on
real-world match tasks", PVLDB 3(1), 2010, Table 1 and Section 3.1. The task
joins 2,616 DBLP publications against 64,263 Google Scholar publications with
5,347 gold correspondences over title, authors, venue and year. Unlike DBLP-ACM
this task is only medium difficulty because Scholar extracts its entities
automatically from full-text documents crawled from the web, which the paper
says leaves "duplicate publications, heterogeneous representations of author
lists or venue names, misspellings, and extraction errors". The Scholar side was
collected by querying Scholar with publication titles and venue names, so it
contains many near-miss records that are not matches. Mudgal et al., SIGMOD 2018
(DeepMatcher, Table 3) report 94.7% F1 here versus 98.4% on DBLP-ACM.
"""

from pydantic import Field

from serf.dspy.schemas.base import (
    EntityMatchCandidate,
    EntitySide,
    SourceText,
    SourceYear,
)

DBLP_SOURCE_NAME = "DBLP"
SCHOLAR_SOURCE_NAME = "Google Scholar"


class DblpScholarPublication(EntitySide):
    """The DBLP publication record in the DBLP-Google Scholar task.

    Same curated source as in DBLP-ACM, but this task's DBLP export abbreviates
    author first names to initials, which makes author comparison against the
    Scholar side noisier than it looks. Cited from Köpcke, Thor and Rahm,
    PVLDB 3(1), 2010, Section 3.1.

    Parameters
    ----------
    title : str
        Publication title as curated by DBLP
    authors : str
        Comma-separated author list with initials instead of full first names
    venue : str
        Abbreviated conference or journal name
    year : int | None
        Publication year
    """

    source_name: SourceText = Field(
        default=DBLP_SOURCE_NAME,
        description="Always DBLP: the curated, clean side of this match task",
    )
    title: SourceText = Field(
        default="",
        description=(
            "Publication title as curated by DBLP. Clean and complete, and the "
            "single most reliable field in this task."
        ),
    )
    authors: SourceText = Field(
        default="",
        description=(
            "Comma-separated author list abbreviated to initials in this export, "
            "for example 'M Rusinkiewicz, W Klas, T Tesch'. Scholar abbreviates "
            "similarly but inconsistently, so match surnames, not full strings."
        ),
    )
    venue: SourceText = Field(
        default="",
        description=(
            "Abbreviated venue name such as 'VLDB' or 'SIGMOD Record'. The Scholar "
            "side stores free text here, so absence of overlap means nothing."
        ),
    )
    year: SourceYear = Field(
        default=None,
        description=(
            "Publication year, almost always present on this side. The Scholar side "
            "often omits it, so a missing Scholar year is not evidence against a match."
        ),
    )


class GoogleScholarPublication(EntitySide):
    """A Google Scholar publication record, automatically extracted from the web.

    This is the dirty side of the task. Köpcke, Thor and Rahm, PVLDB 3(1), 2010,
    Section 3.1 documents duplicate publications, heterogeneous author and venue
    representations, misspellings, and outright extraction errors. Real rows in
    the released file include one whose title is a street address
    ("11578 Sorrento Valley Road") with "QD Inc" as its author list, and the
    ``id`` column header carries a UTF-8 byte order mark. Because Scholar holds
    many duplicates, one DBLP publication legitimately matches several Scholar
    records.

    Parameters
    ----------
    title : str
        Extracted title, which may be truncated, misspelled, or not a title at all
    authors : str
        Extracted author list, often initials only, sometimes an organization
    venue : str
        Free-text venue string, often abbreviated with trailing punctuation
    year : int | None
        Publication year, frequently missing on this side
    """

    source_name: SourceText = Field(
        default=SCHOLAR_SOURCE_NAME,
        description=(
            "Always Google Scholar: entities auto-extracted from crawled full text, "
            "with opaque cluster ids such as 'aKcZKwvwbQwJ'"
        ),
    )
    title: SourceText = Field(
        default="",
        description=(
            "Extracted title, and the primary matching signal. May be truncated, "
            "misspelled, lower-cased, double-encoded so that 'â??' stands for a "
            "quotation mark, stripped of its spaces "
            "('Databasearchitecture optimizedforthenewbottleneck: memoryaccess'), "
            "prefixed with the tail of the preceding citation "
            "('andD. Srivastava. HolisticTwigJoins: ...'), page furniture rather "
            "than a paper ('Terms of Usage Privacy Policy Code of Ethics Contact "
            "Us'), or a pure extraction error such as a street address. Compare by "
            "characters, not by words, because the word boundaries are unreliable."
        ),
    )
    authors: SourceText = Field(
        default="",
        description=(
            "Extracted author list, typically initials plus surname. Often a "
            "scraper artefact instead: 'ACMS Anthology', 'portal.acm.org', "
            "'P Geographer, T Geography'. Read those as missing. Overlap of one or "
            "two real surnames with the DBLP side is normal for a true pair, and "
            "the full strings agree on under a third of them."
        ),
    )
    venue: SourceText = Field(
        default="",
        description=(
            "Free-text venue string, for example 'Phil. Mag,' or a publisher name, "
            "and often empty, HTML-entity-encoded or wrong. Agrees on about a "
            "quarter of true pairs, so it is weak evidence in both directions."
        ),
    )
    year: SourceYear = Field(
        default=None,
        description=(
            "Publication year, missing on more than half of this side's rows "
            "because extraction failed, which is not evidence against a match. "
            "Written as a float, so Scholar's '2002.0' is DBLP's '2002': compare "
            "as numbers, never as strings, and allow one year of slack since "
            "Scholar may record a preprint or reprint year."
        ),
    )


class DblpScholarCandidate(EntityMatchCandidate):
    """A DBLP record paired with a Google Scholar record.

    Parameters
    ----------
    left : DblpScholarPublication
        The DBLP publication
    right : GoogleScholarPublication
        The Google Scholar publication
    """

    left: DblpScholarPublication = Field(description="The DBLP publication, copied from the input")
    right: GoogleScholarPublication = Field(
        description="The Google Scholar publication, copied from the input"
    )
