# The XML adapter was losing most blocks

`serf benchmark -d abt-buy --sample-records 1000` logged this on 22 of its 137
blocks:

```
serf.match.matcher - ERROR - LLM failure for block block_5: Adapter JSONAdapter failed to parse the LM response.
serf.match.run - WARNING - 6 of 60 blocks failed their LLM call and contributed no matches, so recall for this run is understated
```

The matcher is configured with `dspy.XMLAdapter`, so a *JSONAdapter* error is the first clue. DSPy's
`ChatAdapter.__call__` catches any parse failure and silently re-runs the whole request through
`JSONAdapter`, discarding the original error. Every one of those log lines is a block that failed XML
parsing first and then failed the fallback too.

## How often XML parsing actually failed

Replaying the 33 iteration-1 blocks through `XMLAdapter(use_json_adapter_fallback=False)`, against the
DSPy cache so the LM responses are fixed:

| | Blocks parsed | Blocks needing the JSON fallback |
|---|---|---|
| Before | 6 / 33 | 27 |
| After the type fixes | 20 / 33 | 13 |
| After the metacharacter repair | **29 / 33** | 4 |

Six of 33. The fallback was not an edge case, it was the main path: 82% of blocks were paying for two
inferences instead of one, and the ones whose fallback response was also malformed contributed no
matches at all.

## Three causes

**Dicts arrive as strings.** `XMLAdapter` renders a `dict` output field as one flat
`<attributes>...</attributes>` tag, so the model writes a JSON object into the tag body and the adapter
hands the text back as `str`. Pydantic wants a dict:

```
resolved_entities.0.attributes
  Input should be a valid dictionary [type=dict_type, input_value='{"price": "$24.95"}', input_type=str]
```

**XML has no null.** Asked to fill in `bool | None` and `list[int] | None` tags, the model writes the
literal word:

```
resolved_entities.0.match_skip
  Input should be a valid boolean, unable to interpret input [type=bool_parsing, input_value='null', input_type=str]
```

One block produced 34 of these at once. Both are handled now by `field_validator`s on `Entity`, which
is where the constraint belongs: an empty or placeholder tag body means `None`, and a JSON object in a
tag body is a dict.

**Product names contain ampersands.** `XMLAdapter` parses with `xml.etree.ElementTree`, which rejects
the entire document over a single bare `&`. Entity resolution echoes source text back through the
output fields, and the Abt-Buy catalog is full of `Office Home & Student`, `AT&T` and `< 30 lbs`:

```
Failed to parse XML: not well-formed (invalid token): line 16, column 85
```

This one is not fixable by prompting — a model writing free-form product text will not reliably escape
XML entities. `serf.dspy.adapter.RepairingXMLAdapter` instead escapes stray `&` and `<` and retries the
parse. It only ever runs after an unrepaired parse has already failed, and it re-raises unchanged when
escaping makes no difference, so well-formed markup takes exactly the same path it did before.

## What is left

Four of 33 blocks still fail XML and still use the JSON fallback, and those are genuine model errors:
one truncated `attributes` value with a stray quote after the closing brace, one mismatched tag. The
fallback exists for exactly that, and at 4 blocks it costs what it was meant to cost.

## Why it stayed hidden

DSPy's fallback raises the *fallback's* error, not the original, so `logger.error` in the matcher only
ever showed a JSONAdapter message for a pipeline that does not use JSONAdapter. Nothing in the logs
pointed at the XML parse. Finding it took replaying real blocks with `use_json_adapter_fallback=False`;
the synthetic four-entity block used in earlier smoke tests reproduced the doubled call count but
looked like a success from the outside, because the fallback answered correctly.

If a future change makes blocks start failing again, run a replay with the fallback disabled before
reading anything into the error message.
