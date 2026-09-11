"""Typed record schemas for the Walmart-Amazon electronics match task.

Source: the UW-Madison Magellan data repository, produced by Konda et al.,
"Magellan: Toward Building Entity Matching Management Systems" (Magellan
technical report / PVLDB 9(12), 2016), a project run with @WalmartLabs, whose
how-to-guide-driven workflows generated the labelled product match tasks in that
repository. Mudgal et al., SIGMOD 2018 (DeepMatcher, Table 2) package it as a
structured electronics task with five attributes -- title, category, brand,
modelno and price -- joining 2,554 Walmart.com products against 22,074
Amazon.com products, with 962 gold matches in the released candidate set.

Table 3 of the DeepMatcher paper reports 66.9% F1 for the best deep model against
71.9% for the string-similarity baseline Magellan, one of the few tasks where the
simpler baseline wins. That is the signature of a task decided by exact
identifiers rather than by paraphrase: the model number carries the decision.
The paper's saliency analysis (Section 6.2 and Appendix C) agrees, finding that
the strongest tokens are product serial numbers. DeepMatcher also drew its error
sample for structured data from this task, and its "dirty" variant is built by
moving values such as brand into the title while blanking the real column, which
is a good reminder that brand and model number often hide inside the title.

In the DeepMatcher packaging used here, ``exp_data/tableA.csv`` is the Walmart
side and ``exp_data/tableB.csv`` is the Amazon side. This was verified against
the raw archive, whose ``tableA`` rows carry walmartimages.com image URLs and a
Walmart ``groupname`` taxonomy while its ``tableB`` rows carry amazon.com URLs
and ASINs, and by row counts (2,554 versus 22,074).
"""

from pydantic import Field

from serf.dspy.schemas.base import (
    EntityMatchCandidate,
    EntitySide,
    SourcePrice,
    SourceText,
)

WALMART_SOURCE_NAME = "Walmart.com"
AMAZON_SOURCE_NAME = "Amazon.com"


class WalmartProduct(EntitySide):
    """A Walmart.com electronics product listing.

    Categories come from Walmart's own shelf taxonomy, which does not line up
    with Amazon's browse nodes. Cited from Konda et al., Magellan technical
    report, and Mudgal et al., SIGMOD 2018, Table 2.

    Parameters
    ----------
    title : str
        Lower-cased Walmart catalog title, usually brand first and model last
    category : str
        Walmart shelf taxonomy label
    brand : str
        Brand name, lower-cased
    modelno : str
        Manufacturer model number as listed by Walmart
    price : float | None
        Walmart price in US dollars
    """

    source_name: SourceText = Field(
        default=WALMART_SOURCE_NAME,
        description="Always Walmart.com, the smaller side of this match task",
    )
    title: SourceText = Field(
        default="",
        description=(
            "Lower-cased Walmart catalog title, for example 'epson 1500 hours 200w "
            "uhe projector lamp elplp12'. Terse where the Amazon title is "
            "keyword-stuffed, so a true pair can share almost no words. Leads with "
            "the brand and ends with the model number, sometimes truncated: 'hp "
            "cb40 toner' for the cartridge whose code is 'cb400a'."
        ),
    )
    category: SourceText = Field(
        default="",
        description=(
            "Walmart shelf taxonomy label such as 'electronics - general' or "
            "'monitors'. Ignore it. It agrees on under five per cent of true pairs, "
            "barely above the rate for non-pairs, and it is frequently wrong rather "
            "than merely coarse: an HP Ultrium data cartridge is filed under 'mp3 "
            "accessories'. An incompatible-looking category is not evidence against "
            "a match."
        ),
    )
    brand: SourceText = Field(
        default="",
        description=(
            "Brand name, lower-cased. Use it only as a gate: it agrees on most true "
            "pairs but also on two fifths of near misses, so it can rule a pair out "
            "and never rule one in. May be blank even when the brand is visible in "
            "the title."
        ),
    )
    modelno: SourceText = Field(
        default="",
        description=(
            "Manufacturer model number as listed by Walmart, for example 'elplp12'. "
            "The sharpest exact-equality signal in any of these tasks: normalised "
            "equality with the Amazon model number holds on about two thirds of true "
            "pairs and on roughly one in four hundred near misses, so equality "
            "decides the pair. Compare case-insensitively, ignoring dashes and "
            "spaces, and when two codes disagree compare them character by "
            "character, because one changed character always means a different "
            "capacity, colour or revision. One true pair in three has no usable "
            "code on one side or the other, so an absent model number is not "
            "evidence against a match; decide those pairs on the title."
        ),
    )
    price: SourcePrice = Field(
        default=None,
        description=(
            "Walmart price in US dollars, using 0.0 as a null sentinel that must be "
            "read as missing. When both sides carry a real price, a gap inside a "
            "quarter is about three times more common on true pairs than on near "
            "misses, which makes it a tie-breaker only."
        ),
    )


class AmazonElectronicsProduct(EntitySide):
    """An Amazon.com electronics product listing.

    The Amazon side is nearly nine times larger than the Walmart side, so most
    Amazon records have no partner, and it is full of accessories that share a
    brand and category with the product they accompany. Cited from Konda et al.,
    Magellan technical report, and Mudgal et al., SIGMOD 2018, Table 2 and
    Section 6.2.

    Parameters
    ----------
    title : str
        Lower-cased Amazon listing title, usually brand first and model last
    category : str
        Amazon browse-node category label
    brand : str
        Brand name, lower-cased
    modelno : str
        Manufacturer part number as listed by Amazon
    price : float | None
        Current Amazon price in US dollars
    """

    source_name: SourceText = Field(
        default=AMAZON_SOURCE_NAME,
        description=(
            "Always Amazon.com, the much larger side: most of its records have no "
            "Walmart counterpart at all"
        ),
    )
    title: SourceText = Field(
        default="",
        description=(
            "Lower-cased Amazon listing title, for example 'kodak black ink cartridge "
            "10b 1163641'. Usually leads with the brand and ends with the model or "
            "part number. Watch for accessory wording (cable, case, replacement lamp, "
            "refill, mount) that makes an otherwise similar title a different product."
        ),
    )
    category: SourceText = Field(
        default="",
        description=(
            "Amazon browse-node category label such as 'headphone accessories' or "
            "'inkjet printer ink'. Ignore it: Amazon uses hundreds of fine nodes "
            "where Walmart uses dozens of coarse shelves, so the two agree on under "
            "five per cent of true pairs."
        ),
    )
    brand: SourceText = Field(
        default="",
        description=(
            "Brand name, lower-cased. Occasionally the seller name rather than the "
            "manufacturer, and may be blank while the brand is present in the title. "
            "A gate, not evidence for a match."
        ),
    )
    modelno: SourceText = Field(
        default="",
        description=(
            "Manufacturer part number as listed by Amazon, frequently repeated at "
            "the end of the title. Normalised equality with the Walmart model number "
            "decides the pair. Two *different* values are only evidence against a "
            "match once you have checked that this one is really a code: it is blank "
            "on more than a quarter of rows and often holds leftover descriptive "
            "text instead ('high power', 'with csr', 'high contrast matte white'). A "
            "multi-word value with no digits is prose and tells you nothing. When "
            "this value is blank or prose the pair has to be decided on the title, "
            "not rejected."
        ),
    )
    price: SourcePrice = Field(
        default=None,
        description=(
            "Current Amazon price in US dollars, a street price rather than a list "
            "price and so usually below Walmart's. Never zero on this side; null on "
            "about one row in eight. Use it as a tie-breaker with a quarter of "
            "tolerance."
        ),
    )


class WalmartAmazonCandidate(EntityMatchCandidate):
    """A Walmart.com product paired with an Amazon.com product.

    Parameters
    ----------
    left : WalmartProduct
        The Walmart.com listing
    right : AmazonElectronicsProduct
        The Amazon.com listing
    """

    left: WalmartProduct = Field(
        description="The Walmart.com product listing, copied from the input"
    )
    right: AmazonElectronicsProduct = Field(
        description="The Amazon.com product listing, copied from the input"
    )
