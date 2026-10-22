"""MCP server (stdio) with one read-only tool, value_parcel. See docs/mcp.md.

Usage:
    python -m avm.mcp_server            # serves <data>/serving/value_parcel_2026.sqlite
    python -m avm.mcp_server --sample   # the 500-parcel sample database

Startup refuses to serve unless the prediction file and the database match
their recorded hashes (avm.lookup.verify). The tool only reads: it never
trains, writes or regenerates anything, and returns no owner names and no
over/under-assessment verdicts.
"""

import argparse
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations
from pydantic import Field

from avm.lookup import Lookup, db_path

DESCRIPTION = (
    "Value a home in the City of Milwaukee, Wisconsin. Covers single-family and duplex homes only. "
    "Returns the value as of January 1, 2026 from a frozen model fit only on data through 2025, with an 80% range "
    "(rounded to the nearest $1,000), the three largest drivers of the estimate as % effects (SHAP), and 3 to 5 "
    "comparable sales with the search radius and time window used. This is a model estimate with an 80% range, "
    "not an appraisal or an assessment. It gives no over- or under-assessment verdicts and no assessed values. "
    "Accepts either a street address (house number and street; duplex ranges such as 2876-78 work; units are "
    "ignored) or a 10-digit tax key, not both. Statuses: ok; multiple_matches (pick a tax key from the candidates "
    "and call again); no_match (with up to 5 nearby addresses); not_covered (not a single-family or duplex "
    "parcel); no_estimate; not_found; invalid_input."
)
INSTRUCTIONS = ("Milwaukee home values from a public-data model. Always present results as a model estimate with its "
                "80% range, not an appraisal or assessment, and do not infer whether a home is over- or "
                "under-assessed.")


def build_server(lookup: Lookup) -> MCPServer:
    server = MCPServer("milwaukee-avm", instructions=INSTRUCTIONS)

    @server.tool(name="value_parcel", title="Milwaukee home value (model estimate)", description=DESCRIPTION,
                 annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True,
                                             openWorldHint=False),
                 structured_output=True)
    def value_parcel(
        address: Annotated[str | None, Field(description="Street address in the City of Milwaukee, e.g. house "
                                         "number, direction, street and type; city, ZIP and unit are optional")] = None,
        tax_key: Annotated[str | None, Field(description="10-digit City of Milwaukee tax key (leading zeros "
                                         "optional)")] = None,
    ) -> dict[str, Any]:
        return lookup.value_parcel(address=address, tax_key=tax_key)

    return server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="serve the 500-parcel sample database")
    a = ap.parse_args()
    build_server(Lookup(db_path(a.sample))).run("stdio")


if __name__ == "__main__":
    main()
