# Ask Claude about any Milwaukee home

A local MCP server (stdio, Python MCP SDK) with one read-only tool, `value_parcel`. Give it a City of Milwaukee street address or tax key and it returns the frozen model's value as of Jan 1 2026 with its 80% range, the home's recorded characteristics and last sale, the three largest drivers of that estimate, and 3 to 5 comparable sales with the search tier used.

Every response carries the label **"model estimate, not an appraisal or assessment"**. Values and ranges are rounded to the nearest $1,000 (exact values stay in the database). The tool returns no owner names, no assessed values and no over- or under-assessment verdicts. It covers single-family (land use 8810) and duplex (8820) parcels only.

## Setup

1. Build the serving database once (it reads the raw data, so run the pipeline first):

   ```
   python -m avm.serving build      # writes <data>/serving/value_parcel_2026.sqlite and its .sha256
   python -m avm.serving check      # the same hash checks the server runs at startup
   ```

   `--sample` builds a 500-parcel database for a quick test. Timings, table sizes and row counts are written to `<data>/serving/build_report.json`. The database stays in the data folder and is never committed: it holds addresses and tax keys.

2. Connect a client. The server is `python -m avm.mcp_server` run with the project's virtual environment; pass `AVM_DATA_DIR` if the data folder is not the default.

### Claude Code

```
claude mcp add milwaukee-avm -e AVM_DATA_DIR=<data> -- <venv>\Scripts\python.exe -m avm.mcp_server
claude mcp list        # milwaukee-avm should show as connected
```

Add `--scope user` to make it available in every project. Inside a session, `/mcp` shows the server and its one tool.

### Claude Desktop

Edit `claude_desktop_config.json` (Settings, Developer, Edit Config; on Windows it is in `%APPDATA%\Claude\`, on macOS in `~/Library/Application Support/Claude/`):

```json
{
  "mcpServers": {
    "milwaukee-avm": {
      "command": "<venv>\\Scripts\\python.exe",
      "args": ["-m", "avm.mcp_server"],
      "env": {"AVM_DATA_DIR": "<data>"}
    }
  }
}
```

Restart Claude Desktop; Settings, Developer then lists the server and whether it is running. On macOS or Linux the command is `<venv>/bin/python`.

## What the server checks at startup

The server refuses to start (and the client shows it as failed) unless:

- `reports/prospective/predictions_2026.parquet` has the sha256 recorded in `prospective_meta.json` (the file tagged `prospective-2026` and timestamped with OpenTimestamps);
- the database was built from that same file (its `meta` table records the hash);
- the database matches its `.sha256` sidecar.

After that it only reads: an indexed SQLite lookup per call, standard library only (no pandas, no model).

## Recovering the model for explanations

No fitted booster was saved when the 2026 predictions were made, and SHAP needs one. `avm.serving build` refits the frozen primary spec (`reports/frozen/model_spec.json`) exactly as `avm.prospective predict` did: same parcel rows, frame, training rows, configuration and seed. It **aborts unless the refit reproduces every value in the hashed file exactly** (rounded dollars, every parcel), and it checks the training row count against the one recorded at predict time. The recovered booster is saved next to the database and its sha256 is stored in the database's `meta` table. Values and ranges served come from the hashed file, never from the refit; the refit only supplies SHAP values.

## Drivers: grouping and why it differs from Stage 9

Drivers are LightGBM TreeSHAP values in log price, shown as % effects relative to the model's average home (exp of the summed SHAP values, minus one). Correlated features split credit between them, so some are grouped before ranking:

| Driver | Features |
|---|---|
| location | coordinates, the five distance features, and the census tract effect |
| nearby sale prices | nearby sales per sq ft, distance to them, neighborhood prices per sq ft |
| this home's prior sale | its own prior sale (price, time since, whether one exists); the response adds that sale's date and price |
| nearby vacant or city-owned lots | counts of vacant and city-owned parcels within 200 m |
| structure features | individual: finished sq ft, lot size, age, bedrooms, baths, stories, style, duplex, missing-value flags |

The top three by size are returned. The Stage 9 review list ranks single features instead, so there "location" can be split across eight correlated columns, each looking small. Stage 7 already leads its global importance with groups for that reason; the tool applies the same idea per home but keeps structure features individual, because each is a fact about the home a reader can check.

## Subject details and last sale

`subject_details` gives the home's finished sq ft, bedrooms, baths (full.half), year built and lot size in sq ft, as read from the 2025 roll (the Master Property File) for the model. If a field had to come from an earlier roll, a `note` names it.

`subject_last_sale` (date and recorded price) is the home's latest sale in the city's sales files from 2014 to 2025; normally that is an arm's-length sale from the same cleaned set the model trains on. It is separate from the comps and from the "this home's prior sale" driver, which shows only the sale the model used as a feature (that one must be more than 12 months old). The field is left out when the home has no sale on record. If the latest sale on record was removed by cleaning (a resale within 12 months, a sale under $10,000, a multi-parcel sale and so on), `subject_last_sale` shows that sale's date and recorded price with the note "excluded from the model's training data: <reason>; shown for context, not used in the estimate", and the latest arm's-length sale follows inside it as `latest_clean_sale`. A multi-parcel price covers every parcel in the sale, not this home alone.

Comps never include the home's own sales: the subject parcel is removed from its comp pool before the search.

## Address matching

Addresses are matched against the city's Master Address Index:

- Case, punctuation, city, state and ZIP are ignored; a ZIP only breaks ties.
- Full words become the index's abbreviations (North to N, Avenue to AV), and numbered streets become ordinals (40 to 40TH).
- Units (Apt, Unit, #, Upper, Lower, Rear) are ignored and noted, because values are per parcel.
- Short duplex ranges such as 2876-78 are expanded. A range matches only numbers on the same side of the street (odd and even numbers face each other). A range wider than 20 matches only its two end numbers.
- A plain number that also exists with a letter suffix on another parcel (124 and 124A) returns every match for the user to choose; so do addresses shared by several valued parcels.

| Status | Meaning |
|---|---|
| `ok` | one parcel with a model value |
| `multiple_matches` | candidates (tax key, address, land use); call again with a tax key |
| `no_match` | up to 5 nearby addresses on that street, same side first |
| `not_covered` | the parcel's land use is not single-family or duplex |
| `no_estimate` | single-family or duplex, but the model has no value (the missing input is named) |
| `not_found` | tax key not in the 2025 roll |
| `invalid_input` | neither or both inputs, or unreadable input |

## Demo

Three prompts that show the tool's range:

1. "Use value_parcel on `<a duplex, written as a short range such as 2876-78 plus the street>`. What's the Jan 1 2026 estimate and its 80% range, and which three things move it most?" Shows range parsing, the label, rounding and the drivers.
2. "Look up tax key `<10 digits>`. Are the comparable sales a good match on size and distance, and how far did the search have to go?" Shows the comps and the search tier as a confidence signal.
3. "Is `<a condominium address>` over-assessed? And try `<a misspelled address>`." Shows `not_covered`, that the tool gives no assessment verdicts, and `no_match` with nearby addresses.

Talking point: the values were fit only on data through 2025; the prediction file's hash was timestamped in Bitcoin block 972397 on Oct 21 2026, before the city released its 2026 sales file; and the server re-checks that hash every time it starts.
