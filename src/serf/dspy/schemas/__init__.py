"""Per-dataset, strongly-typed record schemas for entity matching.

One module per benchmark match task, each with two side models (one per source)
and an outer candidate type. The side models carry the exact columns of their
own source CSV and field descriptions drawn from the entity resolution
literature, so the LLM sees what each source actually is instead of a single
generic entity description.
"""

from serf.dspy.schemas.abt_buy import AbtBuyCandidate, AbtProduct, BuyProduct
from serf.dspy.schemas.amazon_google import (
    AmazonGoogleCandidate,
    AmazonSoftwareProduct,
    GoogleSoftwareProduct,
)
from serf.dspy.schemas.base import (
    EntityMatchCandidate,
    EntitySide,
    entity_side,
    strip_side_prefix,
)
from serf.dspy.schemas.dblp_acm import AcmPublication, DblpAcmCandidate, DblpPublication
from serf.dspy.schemas.dblp_scholar import (
    DblpScholarCandidate,
    DblpScholarPublication,
    GoogleScholarPublication,
)
from serf.dspy.schemas.walmart_amazon import (
    AmazonElectronicsProduct,
    WalmartAmazonCandidate,
    WalmartProduct,
)

__all__ = [
    "AbtBuyCandidate",
    "AbtProduct",
    "AcmPublication",
    "AmazonElectronicsProduct",
    "AmazonGoogleCandidate",
    "AmazonSoftwareProduct",
    "BuyProduct",
    "DblpAcmCandidate",
    "DblpPublication",
    "DblpScholarCandidate",
    "DblpScholarPublication",
    "EntityMatchCandidate",
    "EntitySide",
    "GoogleScholarPublication",
    "GoogleSoftwareProduct",
    "WalmartAmazonCandidate",
    "WalmartProduct",
    "entity_side",
    "strip_side_prefix",
]
