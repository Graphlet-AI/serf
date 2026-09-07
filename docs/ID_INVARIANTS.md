# Identifier Conservation: The SERF Match/Merge Contract

> **Every integer identifier that enters a block must be accounted for on the way out — either as the identifier of an output record, or inside the merge list of exactly one output record. Nothing is dropped. Nothing is merged into a record the model did not explicitly say it belonged to.**

This is the correctness backbone of the system. A resolution run that violates it is a failed run.

The semantics below were established in the predecessor system, **Abzu** (`/Users/rjurney/Software/weave`), where getting them right was the single hardest part of the work. They are reproduced here as a specification so that SERF can implement them faithfully. The implementation may be improved — cleaner, better tested, better named. **The semantics may not be changed.**

Reference implementation: `abzu/er/uuid.py` (`process_block_with_uuid_mapping`, ~574 lines) and `abzu/er/match.py`. Current SERF implementation: `src/serf/match/uuid_mapper.py` and `src/serf/match/matcher.py`. Section 8 lists every place they currently diverge.

---

## 1. Why identifiers get swapped at all

Language models are measurably worse at manipulating UUIDs than small integers. A UUID is ~36 characters of high-entropy text; a block of 100 records carrying UUIDs plus accumulated provenance can spend more context on identifiers than on the data being matched, and the model transcribes them wrong.

So the pipeline runs an **identifier swapper**: before the LLM call, every record's stable identifier is replaced with a small consecutive integer local to that block. After the call, the integers are swapped back. The model only ever sees integers.

**Integers in, integers out.** This is the contract at every boundary of the matching stage, regardless of what the source dataset uses for keys.

## 2. The four sets

Everything follows from keeping four sets straight. Get these right and the rest is bookkeeping.

| Set | Definition |
| --- | --- |
| **Input universe** | Every identifier that enters the block. Critically, this is _not_ just the record identifiers — it also includes every identifier already sitting in an incoming record's `source_ids` from a previous round. |
| **Output identifiers** | Identifiers appearing in the `source_ids` merge lists of the records the LLM returned. |
| **Master identifiers** | The top-level identifier of each returned record. Under the MDM convention (§3) a master's own identifier is _not_ in its own `source_ids`, so masters must be counted separately or every master looks missing. |
| **Missing** | `input_universe − (output_identifiers ∪ master_identifiers)`. |

The input universe is where implementations most often go wrong. On iteration 2 and later, an incoming record already carries the identifiers it absorbed in iteration 1. If those are not in the input universe, a model that drops the record takes its whole merge history with it, silently, and coverage validation will not catch it because it was never counted as present.

Abzu builds the universe from both sources:

```python
# abzu/er/uuid.py:186-203
all_input_uuids: set[str] = set()
...
    if "uuid" in comp_data and comp_data["uuid"]:
        input_companies_by_uuid[comp_data["uuid"]] = comp_data
        all_input_uuids.add(comp_data["uuid"])
    ...
    if "source_uuids" in comp_data and isinstance(comp_data["source_uuids"], list):
        for source_uuid in comp_data["source_uuids"]:
            if source_uuid:
                all_input_uuids.add(source_uuid)
                source_uuid_to_company[source_uuid] = comp_data
```

## 3. The MDM master-record convention

When the model merges a group of records:

- The record with the **lowest input integer identifier** becomes the master.
- **All other identifiers in the group** go into the master's `source_ids`.
- The master's own identifier is **not** in its own `source_ids`.
- `source_ids` accumulate transitively: if record 1 arrives carrying `source_ids=[3, 7]` and record 22 arrives carrying `source_ids=[2, 4]`, and the model merges them, the output is master `id=1` with `source_ids=[22, 3, 7, 2, 4]`.

That last example is the canonical few-shot demonstration and it is worth keeping verbatim, because transitive accumulation is the part models get wrong.

Downstream evaluation joins on the master identifier and assumes the lowest-id convention. Changing it breaks cross-iteration validation.

## 4. Strip before the call, restore after

Provenance is expensive to send and the model does not need it to make a decision. After many rounds a single record can carry hundreds of accumulated identifiers.

**Stripped before the call**, held in a local cache keyed by mapped integer:

- the stable identifier (UUID or equivalent)
- `source_ids` — **must be cleared, not just the UUIDs**
- `source_uuids`
- `match_skip`, `match_skip_history`

Clearing `source_ids` is not optional and is not merely an optimization. The mapped integers are block-local, in the range assigned by the swapper. If a record arrives already carrying `source_ids=[3, 7]` from a previous round and those are left in place, they occupy the same numeric space as the block-local mapped integers, and the unmapping step will resolve them against the wrong records. This is a silent data-corruption bug that only appears from iteration 2 onward.

**Restored after the call**, from the cache: original identifiers, and the union of every merged record's cached provenance, deduplicated, with the master's own identifier removed.

## 5. Two-phase recovery

After the model returns and provenance is restored, compute the missing set (§2). If it is empty, done. If not, every missing identifier falls into one of two cases and each gets its own phase.

### Phase 1 — patch provenance onto records that survived

A missing identifier was carried in the `source_ids` of an incoming record, and **that record's parent is present in the output.** The model returned the record but dropped part of its merge history. Nothing needs to be re-emitted; the provenance just needs to be put back.

```python
# abzu/er/uuid.py:493-507 — Step 1: Add back missing UUIDs to existing output companies
for resolved_company in resolved_companies:
    company_source_uuids = resolved_company.get("source_uuids") or []
    for source_uuid in company_source_uuids:
        if source_uuid in uuids_to_add_back:
            for uuid_to_add in uuids_to_add_back[source_uuid]:
                if uuid_to_add not in company_source_uuids:
                    company_source_uuids.append(uuid_to_add)
    if company_source_uuids:
        resolved_company["source_uuids"] = sorted(list(set(company_source_uuids)))
```

If the _parent_ is also missing, this is not a Phase 1 case — the parent is promoted into Phase 2 and recovering the parent recovers its children with it.

### Phase 2 — re-emit records the model dropped entirely

A missing identifier belonged to a record the model simply did not return. The record is restored from the input cache and re-emitted as a standalone, unmerged record.

```python
# abzu/er/uuid.py:509-543 — Step 2: Recover entire companies that are missing
for company_uuid in companies_to_recover:
    if company_uuid in input_companies_by_uuid:
        missing_company = copy.deepcopy(input_companies_by_uuid[company_uuid])

        # IMPORTANT: Remove any integer IDs from previous iterations
        missing_company.pop("id", None)
        missing_company.pop("source_ids", None)

        # Ensure source_uuids contains the company's own UUID
        if "source_uuids" not in missing_company or not missing_company["source_uuids"]:
            missing_company["source_uuids"] = [company_uuid]
        elif company_uuid not in missing_company["source_uuids"]:
            missing_company["source_uuids"].append(company_uuid)

        missing_company["match_skip"] = True
        missing_company["match_skip_reason"] = "missing_in_match_output"

        skip_history = missing_company.get("match_skip_history", []) or []
        if iteration not in skip_history:
            skip_history.append(iteration)
        missing_company["match_skip_history"] = skip_history

        resolved_companies.append(missing_company)
```

Three details in there are load-bearing:

1. **Stale integer identifiers are dropped** (`pop("id")`, `pop("source_ids")`) so the recovered record cannot collide with the next round's block-local numbering.
2. **The record's own identifier is forced into its own provenance list.** A recovered record is not a master of a merge, so the MDM exception in §3 does not apply, and coverage validation counts provenance lists.
3. **The iteration number is appended to** `match_skip_history`, giving per-record skip frequency across rounds. A record skipped in five consecutive rounds is a signal about the blocking, not about the model.

### The rule behind both phases

A dropped record is **never** assumed to have been merged into something. Absence from the output is absence of evidence, not evidence of a match. The system re-emits it unmerged and marks why. This is what makes it safe to run at block sizes where the model measurably drops records.

## 6. Skip reasons

`match_skip_reason` is the audit trail. Every record that did not go through a normal LLM merge carries one.

| Value | Set where | Meaning |
| --- | --- | --- |
| `singleton_block` | Match orchestration, before any LLM call | The block had one record. Nothing to match against. Passed through untouched. |
| `error_recovery` | Match orchestration, on exception | The LLM call failed. All input records passed through unchanged. |
| `missing_in_match_output` | Phase 2 recovery | The LLM dropped this record. Restored from cache, unmerged. |
| `missing_primary_uuid` | Evaluation only — a counter, never written to a record | A record identifier went missing. |
| `missing_source_uuid` | Evaluation only — a counter, never written to a record | A provenance identifier went missing. |

The distinction in the last two matters: they are diagnostics computed during evaluation, not states a record can be in. Do not write them onto records.

**Singletons must be short-circuited before the LLM.** A block of one has no possible match. Sending it costs money, adds a chance for the model to mangle it, and pollutes the results. Abzu splits singleton blocks out at the DataFrame level (`block_size == 1`) and never calls the model on them.

## 7. New identifiers on merged output

When a block is genuinely resolved (`was_resolved == True`), **every** output record from that block gets a freshly generated stable identifier, including Phase 2 recoveries. This makes round _N_ output distinguishable from round _N−1_ input, which is what allows cross-round validation to detect a pipeline that accidentally re-emits its own input.

```python
# abzu/er/match.py:52-58
if result.get("was_resolved") and "resolved_companies" in result:
    for company in result["resolved_companies"]:
        company["uuid"] = str(uuid.uuid4())
```

Singleton blocks and error-recovered blocks have `was_resolved == False` and **keep their original identifiers.** They did not change, so they do not get a new identity. Assigning new identifiers unconditionally destroys that signal.

## 8. Where SERF currently diverges

Audited against `src/serf/match/uuid_mapper.py` and `src/serf/match/matcher.py`. Each of these is a task in [.cursor/scratchpad.md](../.cursor/scratchpad.md) (Milestone 1).

| # | Divergence | Consequence | Severity |
| --- | --- | --- | --- |
| D1 | `source_ids` are **not** stripped before the LLM call (`uuid_mapper.py:57-60` clears `uuid` and `source_uuids` only) | Pre-existing identifiers from earlier rounds collide with block-local mapped integers; unmapping resolves them against the wrong records | **Corruption, iteration ≥ 2** |
| D2 | Input universe is only `_int_to_original.keys()` (`uuid_mapper.py:98`) — it excludes provenance carried in on incoming records | Dropped merge history is invisible to the missing-set computation; coverage validation cannot detect it | **Silent loss** |
| D3 | Only Phase 2 is implemented. There is no `uuids_to_add_back` map and no Phase 1 | Records that survive with truncated provenance are never repaired | **Silent loss** |
| D4 | `singleton_block` is never set; singleton blocks are sent to the LLM | Wasted spend, unnecessary exposure to model error, skip analysis undercounts | High |
| D5 | `_assign_uuids` (`matcher.py:178-180`) assigns new identifiers to **all** output including error-recovery pass-throughs | Round _N_ output is indistinguishable from unchanged input; overlap checks lose their meaning | High |
| D6 | Mapped integers start at 0 (`uuid_mapper.py:46-47`); Abzu starts at 1 | Models conflate `0` with null/absent in structured output | Medium |
| D7 | `config.yml` declares `er.matching.max_retries` and `retry_delay_ms`; the matcher implements neither | A transient API failure becomes a whole block of `error_recovery` records | Medium |
| D8 | `config.yml` declares `er.blocking.min_block_size`; nothing reads it | Dead configuration | Low |

D1, D2 and D3 are the ones that can lose data. They are the first thing to fix, and each needs a test that fails against the current implementation before the fix lands.

## 9. Required tests

One test per invariant, named for the invariant. These are the tests that must fail if someone breaks the contract.

1. **Round trip.** Every input identifier appears exactly once in the output, as a master or in exactly one merge list.
2. **Dropped record.** A mock LLM omits a record entirely. It comes back with `match_skip_reason == "missing_in_match_output"`, its original field values, and its own identifier in its provenance list.
3. **Dropped provenance, surviving parent.** A mock LLM returns a record but truncates its `source_ids`. Phase 1 restores the missing entries. (Fails today — D3.)
4. **Second round.** A record carrying `source_ids` from round 1 goes through round 2. Its round-1 provenance survives, and none of those identifiers is resolved against a block-local record. (Fails today — D1, D2.)
5. **Transitive accumulation.** Master `id=1` `source_ids=[3, 7]` merged with `id=22` `source_ids=[2, 4]` yields master `id=1` with `source_ids == [22, 3, 7, 2, 4]` as a set, and `1` is not in its own list.
6. **Singleton short-circuit.** A block of one produces no LLM call and one output record with `match_skip_reason == "singleton_block"` and its original identifier. (Fails today — D4.)
7. **Error recovery.** The LLM raises. All input records come back unchanged with `match_skip_reason == "error_recovery"`, `was_resolved == False`, and **original** identifiers. (Partially fails today — D5.)
8. **Skip history accumulates.** A record skipped in rounds 1, 2 and 3 ends with `match_skip_history == [1, 2, 3]`.
9. **Validation gates.** A synthetic run that drops 1% of records fails the coverage gate; a clean run passes all four checks.
