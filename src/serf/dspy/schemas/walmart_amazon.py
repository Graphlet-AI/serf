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
            "uhe projector lamp elplp12'. Normally leads with the brand and ends "
            "with the model number, and often restates brand or model that is also "
            "in its own column."
        ),
    )
    category: SourceText = Field(
        default="",
        description=(
            "Walmart shelf taxonomy label such as 'electronics - general' or "
            "'monitors'. Walmart and Amazon use different taxonomies, so a category "
            "mismatch is weak evidence against a match; only a clearly incompatible "
            "product kind matters."
        ),
    )
    brand: SourceText = Field(
        default="",
        description=(
            "Brand name, lower-cased. Should agree with the Amazon brand for a true "
            "match, allowing for aliases and sub-brands. May be blank even when the "
            "brand is visible in the title."
        ),
    )
    modelno: SourceText = Field(
        default="",
        description=(
            "Manufacturer model number as listed by Walmart, for example 'elplp12'. "
            "The highest-precision field in this task: normalised equality with the "
            "Amazon model number is close to decisive. Compare case-insensitively "
            "and ignore separators such as dashes and spaces."
        ),
    )
    price: SourcePrice = Field(
        default=None,
        description=(
            "Walmart price in US dollars. Retail prices differ between the two "
            "retailers, so price cannot decide a match; a very large gap can hint at "
            "an accessory versus the main product."
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
            "'inkjet printer ink'. Finer-grained and worded differently from the "
            "Walmart shelf label, so do not require the two to be equal."
        ),
    )
    brand: SourceText = Field(
        default="",
        description=(
            "Brand name, lower-cased. Occasionally the seller name rather than the "
            "manufacturer, and may be blank while the brand is present in the title."
        ),
    )
    modelno: SourceText = Field(
        default="",
        description=(
            "Manufacturer part number as listed by Amazon, frequently repeated at the "
            "end of the title. Normalised equality with the Walmart model number is "
            "close to decisive; a clear mismatch of two present model numbers is "
            "strong evidence against a match."
        ),
    )
    price: SourcePrice = Field(
        default=None,
        description=(
            "Current Amazon price in US dollars, which is a street price rather than "
            "a list price and therefore usually below Walmart's."
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
