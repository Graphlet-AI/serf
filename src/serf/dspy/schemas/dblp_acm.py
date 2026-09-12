"""Typed record schemas for the DBLP-ACM bibliographic match task.

Source: Köpcke, Thor and Rahm, "Evaluation of entity resolution approaches on
real-world match tasks", PVLDB 3(1), 2010, Table 1 and Section 3.1. The task
joins 2,616 DBLP publications against 2,294 ACM Digital Library publications
with 2,224 gold correspondences over the attributes title, authors, venue and
year. The paper calls it the easiest of its four tasks: both sources are
well-structured, partially manually curated, and cover the same set of computer
science conferences and journals, and every evaluated approach except one
exceeded 91% F-measure. Mudgal et al., SIGMOD 2018 (DeepMatcher, Table 3)
report 98.4% F1 on the same task, so residual errors are few and specific.
"""

from pydantic import Field

from serf.dspy.schemas.base import (
    EntityMatchCandidate,
    EntitySide,
    SourceText,
    SourceYear,
)

DBLP_SOURCE_NAME = "DBLP"
ACM_SOURCE_NAME = "ACM"


class DblpPublication(EntitySide):
    """A DBLP publication record from the DBLP-ACM task.

    DBLP is a curated computer science bibliography, so its values are clean and
    consistently formatted. Its ids are human-readable citation keys that encode
    venue, first author and year, and its venues are the short abbreviated names
    the community uses. Cited from Köpcke, Thor and Rahm, PVLDB 3(1), 2010,
    Section 3.1.

    Parameters
    ----------
    title : str
        Publication title as curated by DBLP
    authors : str
        Comma-separated author list with full first names
    venue : str
        Abbreviated conference or journal name
    year : int | None
        Publication year
    """

    source_name: SourceText = Field(
        default=DBLP_SOURCE_NAME,
        description="Always DBLP: a curated computer science bibliography",
    )
    title: SourceText = Field(
        default="",
        description=(
            "Publication title as curated by DBLP. Clean and correctly cased, but "
            "titles in this snapshot are sometimes cut off mid-word, so a title "
            "that is a prefix of the other side's title is still strong evidence."
        ),
    )
    authors: SourceText = Field(
        default="",
        description=(
            "Comma-separated author list, usually with full first names "
            "(for example 'Viswanath Poosala, Yannis E. Ioannidis'). Author order "
            "matches the publication. A bare '?' is DBLP's way of writing 'no "
            "author list recorded' and must be read as missing, not as an author."
        ),
    )
    venue: SourceText = Field(
        default="",
        description=(
            "Abbreviated venue name as used by DBLP, for example 'VLDB', "
            "'SIGMOD Record' or 'SIGMOD Conference'. The ACM side spells the same "
            "venue out in full, so the two never agree as strings. Map it through "
            "the crosswalk in the instructions and use it only to separate a "
            "conference paper from its journal version."
        ),
    )
    year: SourceYear = Field(
        default=None,
        description=(
            "Publication year. Present on essentially every row and equal to the "
            "ACM year on every true pair, so require equality rather than "
            "allowing slack: an unequal year rules the pair out even when the "
            "titles and the authors both match."
        ),
    )


class AcmPublication(EntitySide):
    """An ACM Digital Library publication record from the DBLP-ACM task.

    The ACM side covers the same conferences and journals as the DBLP side but
    identifies records by numeric ACM article ids and spells venue names out in
    full. Cited from Köpcke, Thor and Rahm, PVLDB 3(1), 2010, Section 3.1.

    Parameters
    ----------
    title : str
        Publication title as recorded by the ACM Digital Library
    authors : str
        Comma-separated author list
    venue : str
        Full, spelled-out conference or journal name
    year : int | None
        Publication year
    """

    source_name: SourceText = Field(
        default=ACM_SOURCE_NAME,
        description="Always ACM: the ACM Digital Library, identified by numeric article ids",
    )
    title: SourceText = Field(
        default="",
        description=(
            "Publication title as recorded by the ACM Digital Library. Sentence-"
            "cased where DBLP is title-cased, so lowercase before comparing. ACM "
            "keeps the full subtitle after a colon where DBLP truncates it, so an "
            "ACM title that starts with the whole DBLP title is the same paper."
        ),
    )
    authors: SourceText = Field(
        default="",
        description=(
            "Comma-separated author list. Uses different first-name forms and "
            "orderings from DBLP often enough that the two strings are equal on "
            "under a third of true pairs, so compare surnames only. Carries "
            "numeric character references, so 'Oliver G&#252;nther' is "
            "'Oliver Günther'."
        ),
    )
    venue: SourceText = Field(
        default="",
        description=(
            "Full venue name, for example 'International Conference on Management "
            "of Data' for the venue DBLP abbreviates as 'SIGMOD Conference'. "
            "Carries raw HTML entities such as '&mdash;'. Treat the abbreviated "
            "and spelled-out forms of one venue as equal."
        ),
    )
    year: SourceYear = Field(
        default=None,
        description=(
            "Publication year. Equal to the DBLP year on every true pair and on "
            "only about an eighth of the hardest non-pairs, which makes it the "
            "cheapest filter available. Require equality."
        ),
    )


class DblpAcmCandidate(EntityMatchCandidate):
    """A DBLP record paired with an ACM record.

    Parameters
    ----------
    left : DblpPublication
        The DBLP publication
    right : AcmPublication
        The ACM Digital Library publication
    """

    left: DblpPublication = Field(description="The DBLP publication, copied from the input")
    right: AcmPublication = Field(
        description="The ACM Digital Library publication, copied from the input"
    )
