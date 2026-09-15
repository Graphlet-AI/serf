"""Typed record schemas for the Abt-Buy e-commerce match task.

Source: Köpcke, Thor and Rahm, "Evaluation of entity resolution approaches on
real-world match tasks", PVLDB 3(1), 2010, Table 1 and Section 3.1. The task
joins 1,081 Abt.com products against 1,092 Buy.com products with 1,097 gold
correspondences. Only products carrying a valid UPC were kept so that a perfect
mapping exists, but the UPC is deliberately not part of the released attributes,
which leaves product name, description, manufacturer and price. The paper finds
e-commerce matching far harder than bibliographic matching: no evaluated
approach exceeded 70% F-measure on Abt-Buy.

Mudgal et al., SIGMOD 2018 (DeepMatcher) treat Abt-Buy as a *textual* task
because every attribute is a text blob, and report 62.8% F1 with the best deep
model against 43.6% for the string-similarity-based Magellan baseline. They also
show the name/title attribute is the "informative" one that packs brand and
model number: removing it drops the best F1 to 47.7%.

The two sides do not share a schema. Buy.com publishes a manufacturer column and
Abt.com does not, so on the Abt side the brand has to be read out of the product
name.
"""

from pydantic import Field

from serf.dspy.schemas.base import (
    EntityMatchCandidate,
    EntitySide,
    SourcePrice,
    SourceText,
)

ABT_SOURCE_NAME = "Abt.com"
BUY_SOURCE_NAME = "Buy.com"


class AbtProduct(EntitySide):
    """An Abt.com product listing.

    Abt names follow a "Brand Product Description - MODELNO" convention and the
    description repeats the name before appending slash-separated specifications.
    There is no manufacturer column on this side. Cited from Köpcke, Thor and
    Rahm, PVLDB 3(1), 2010, Section 3.1 and Mudgal et al., SIGMOD 2018, Table 4.

    Parameters
    ----------
    name : str
        Product name, usually ending in the manufacturer model number
    description : str
        Long specification blob, slash-separated, repeating the name first
    price : float | None
        Listed price in US dollars, frequently missing
    """

    source_name: SourceText = Field(
        default=ABT_SOURCE_NAME,
        description="Always Abt.com, an electronics and appliance retailer with no brand column",
    )
    name: SourceText = Field(
        default="",
        description=(
            "Product name, conventionally 'Brand Product Description - MODELNO', "
            "for example 'Sony White Earbud Style Headphones - MDREX55WH'. The "
            "trailing model code is the highest-precision evidence in this task, "
            "worth far more than the words around it: the full names are identical "
            "on under two per cent of true pairs. The leading token is normally "
            "the brand, which has no column of its own on this side."
        ),
    )
    description: SourceText = Field(
        default="",
        description=(
            "Long specification blob that repeats the name verbatim and then lists "
            "features separated by '/'. Because it restates the name it is not "
            "independent evidence and agreeing with the name proves nothing; mine "
            "it only for a model code or a capacity the name left out."
        ),
    )
    price: SourcePrice = Field(
        default=None,
        description=(
            "Listed price in US dollars, stored with a currency symbol and missing "
            "on most rows, so four gold pairs in five have no comparable price at "
            "all. Where both sides do have one, prices within a quarter of each "
            "other are three times more common on true pairs than on near misses, "
            "which makes it a tie-breaker and never a reason to decide."
        ),
    )


class BuyProduct(EntitySide):
    """A Buy.com product listing.

    Buy.com carries a manufacturer column that the Abt side lacks, and its
    descriptions are short spec fragments rather than long blobs. Cited from
    Köpcke, Thor and Rahm, PVLDB 3(1), 2010, Section 3.1 and Mudgal et al.,
    SIGMOD 2018, Table 4.

    Parameters
    ----------
    name : str
        Product name, often including the model number
    description : str
        Short specification fragment, frequently nearly empty
    manufacturer : str
        Brand, usually upper-cased; absent from the Abt side entirely
    price : float | None
        Listed price in US dollars, frequently missing
    """

    source_name: SourceText = Field(
        default=BUY_SOURCE_NAME,
        description="Always Buy.com, an online retailer that does publish a manufacturer column",
    )
    name: SourceText = Field(
        default="",
        description=(
            "Product name, for example 'Ex Series Earbuds Wht - MDR EX55/WHI'. "
            "Shorter and worded differently from the Abt name for the same product, "
            "and it interleaves the model code with spaces and slashes where Abt "
            "appends it unbroken. Strip every separator out of this name before "
            "testing whether an Abt code appears inside it."
        ),
    )
    description: SourceText = Field(
        default="",
        description=(
            "Short specification fragment such as '5 x 10/100Base-TX LAN', missing "
            "two times in five and sometimes just a colour word. Worth reading only "
            "for a model code the name omitted."
        ),
    )
    manufacturer: SourceText = Field(
        default="",
        description=(
            "Brand, usually upper-cased such as 'LINKSYS'. The Abt side has no such "
            "column, so this can only constrain a candidate against the leading "
            "tokens of the Abt name, never be compared field to field."
        ),
    )
    price: SourcePrice = Field(
        default=None,
        description=(
            "Listed price in US dollars, missing on nearly half of these rows. "
            "Useful only as a tie-breaker when the Abt side also has one, with "
            "roughly a quarter of tolerance."
        ),
    )


class AbtBuyCandidate(EntityMatchCandidate):
    """An Abt.com product paired with a Buy.com product.

    Parameters
    ----------
    left : AbtProduct
        The Abt.com listing
    right : BuyProduct
        The Buy.com listing
    """

    left: AbtProduct = Field(description="The Abt.com product listing, copied from the input")
    right: BuyProduct = Field(description="The Buy.com product listing, copied from the input")
