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
            "for example 'Sony Turntable - PSLX350H'. This is the most informative "
            "field on this side: the trailing model number is the highest-precision "
            "evidence available, and the leading token is normally the brand, which "
            "has no column of its own here."
        ),
    )
    description: SourceText = Field(
        default="",
        description=(
            "Long specification blob that repeats the name and then lists features "
            "separated by '/'. Contains model numbers, capacities and dimensions "
            "worth checking, but is verbose and unaligned with the Buy.com side."
        ),
    )
    price: SourcePrice = Field(
        default=None,
        description=(
            "Listed price in US dollars, stored with a currency symbol in the source "
            "and often missing. Two retailers rarely list the same price, so a price "
            "difference is weak evidence against a match and a missing price is none."
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
            "Product name, for example 'Linksys EtherFast EZXS88W Ethernet Switch - "
            "EZXS88W'. Shorter and worded differently from the Abt name for the same "
            "product, so match on brand plus model number plus product type rather "
            "than on string overlap."
        ),
    )
    description: SourceText = Field(
        default="",
        description=(
            "Short specification fragment such as '5 x 10/100Base-TX LAN', often "
            "nearly empty. Much less informative than the Abt description."
        ),
    )
    manufacturer: SourceText = Field(
        default="",
        description=(
            "Brand, usually upper-cased such as 'LINKSYS'. The Abt side has no such "
            "column, so compare this against the leading tokens of the Abt name."
        ),
    )
    price: SourcePrice = Field(
        default=None,
        description=(
            "Listed price in US dollars, often missing. Prices differ between the two "
            "retailers, so they cannot decide a match."
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
