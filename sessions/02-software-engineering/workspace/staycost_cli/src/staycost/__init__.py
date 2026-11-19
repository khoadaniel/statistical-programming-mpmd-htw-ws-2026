"""staycost: the cost of a stay in a Berlin listing.

This file makes the folder `staycost` a package. It runs once, when the package is first imported,
and re-exports the main function so that users can write `from staycost import stay_cost`.
"""

from staycost.costs import stay_cost

__all__ = ["stay_cost"]
