# BENCHMARKS

The five datasets SERF is scored on all come out of one research group at UW-Madison, and
they exist in the shape they do because of two systems built there: **Magellan**, which
defined how an entity matching pipeline is assembled and debugged, and **DeepMatcher**,
which defined the train/valid/test splits most later papers report against. Both technical
reports are in this repository, converted to Markdown:

- [`docs/papers/magellan-tr.md`](docs/papers/magellan-tr.md) — Konda et al., *Magellan: Toward
  Building Entity Matching Management Systems*
- [`docs/papers/deepmatcher-tr.md`](docs/papers/deepmatcher-tr.md) — Mudgal et al., *Deep
  Learning for Entity Matching: A Design Space Exploration*, SIGMOD 2018

Everything below the "Method" section is measured, not quoted. Reproduce it with:

```bash
uv run serf profile-benchmark --output BENCHMARKS_PROFILE.md --json profiles.json
```

That command is implemented in `src/serf/analyze/benchmarks.py` and runs entirely in Spark
SQL over the two source tables and the gold mapping.

---

## Lessons from Magellan

Magellan's contribution is a *how-to guide*: not a better matching algorithm, but an
opinionated sequence of steps with tools for each. Four of its findings shape how SERF
should approach these datasets.

**Blocking is the recall ceiling, and you cannot see it without a debugger.** Magellan's
blocking debugger takes the pairs a blocker *removed* and ranks the top 200 most
match-looking ones for a human to read. Users stop tuning when that list comes back empty.
In the 24-team study, all 24 teams debugged their blockers, taking 1 to 10 iterations
(average 5), and 12 teams explicitly used "the debugger returned no matches" as their
stopping rule. The same logic is why SERF measures blocking recall separately from
end-to-end F1: a pair that blocking never proposes cannot be recovered downstream.

**Attribute discriminativeness is cheap and predicts usefulness.** To decide which
attributes to compare, Magellan scores each one with

```
unique(x, A)  = distinct non-empty values / non-empty values
missing(x, A) = empty values / rows
s(x, A)       = unique(x, A) + 1 - missing(x, A)
s(x)          = s(x, A) * s(x, B)
```

A score near 2.0 per table means "present everywhere and nearly a key". This whole document
reports `s(x)` for every shared attribute, because it separates the columns worth putting in
a prompt from the ones that are decoration.

**Escalate blockers in order of cost.** Overlap first ("must share *k* tokens"), then
attribute equivalence ("must share a value"), then sorted-neighbourhood or hash, and only
then rule-based blocking. Stop when the surviving candidate set is small enough. Teams
pruned away more than 95% of `|A x B|` in 21 of 24 cases, averaging 97.3%.

**Most accuracy comes from fixing data and features, not from the learner.** Across the
24 teams, the initially selected learner scored P 56–100%, R 37.5–100%, F1 56–99.5%. After
debugging — which meant adding and removing features (21 teams), cleaning data (12 teams),
and correcting wrong labels (16 teams) — and then adding 1–5 rules, the same teams reached
P 91.3–100%, R 64.7–100%, F1 78.6–100%. The average gain was 18.8 F1 and the largest was
72.5 F1. At WalmartLabs, debugging a proxy random forest in place of an opaque production
matcher raised recall 34% while costing 0.65% precision. Three teams *lost* F1 by adding
rules, all three starting from a baseline above 94%, because they overfit the development
set.

## Lessons from DeepMatcher

DeepMatcher compared four deep learning architectures (SIF, RNN, Attention, Hybrid) against
Magellan across 11 structured, 6 textual and 6 dirty matching tasks. Three findings matter
here.

**Deep learning does not beat feature engineering on clean structured data, but it wins
big when the strings are synonyms.** On our five datasets:

| Dataset | Magellan F1 | Best DL F1 | Difference |
|---|---|---|---|
| DBLP-ACM | 98.4 | 98.4 (several) | 0.0 |
| DBLP-Scholar | 92.3 | 94.7 (Hybrid) | +2.4 |
| Walmart-Amazon | 71.9 | 67.6 (RNN) | **-4.3** |
| Amazon-Google | 49.1 | 69.3 (Hybrid) | **+20.2** |
| Abt-Buy (textual) | 43.6 | 62.8 (Hybrid) | +19.2 |

The Amazon-Google gap is explained in the report: "the product titles across matching pairs
from the source datasets correspond to synonyms of one another... semantically similar but
have large string similarity distances". The Walmart-Amazon loss is the opposite failure —
the deep model overfits the quirks of the training set where Magellan's string-similarity
features cannot.

**Three error categories account for the majority of mistakes.** Sampling 150 misclassified
pairs, 80 fell into three groups, quoted verbatim from section 6.2:

1. *Linguistic variations of domain-specific terms* (false negatives) — "wooster brush r097
   sherlock **gt**" against "wooster brush r097 sherlock **grip tip**".
2. *Missing highly-informative tokens* (false negatives) — "ocz tech vertex 2 60gb 2.5 sata2
   SSD" against "ocz tech 60gb vertex 2 sata2 2.5-inch SSD **oczssd22vte60g**", where the
   model over-weights a code that adds no information.
3. *Similar but semantically different tokens* (false positives) — "nelson sprinkler
   **50571** brass pipe & hose fitting" against "nelson sprinkler brass pipe & hose fitting
   **50575**", an edit distance of one between different products.

Every one of these three shows up in the measured examples below, which is the main reason
to trust the categorisation.

**The datasets are candidate sets, not cross products.** Section B.1: the structured
datasets were built by "first apply[ing] blocking using [Magellan] to get a candidate set",
then labelling every pair in it. The dirty variants were manufactured by moving each
attribute's value into `title` with 50% probability. This has a direct consequence for how
we read our own precision, quantified per dataset below.

---

## Method

Three measurements recur in every dataset section.

**Discriminativeness** is Magellan's `s(x) = s(x, A) * s(x, B)`, defined above.

**Agreement on matches against near misses.** For each shared attribute we report the
fraction of pairs whose normalised values are equal, computed twice: once over the gold
pairs, and once over *near misses* — the highest token-overlap non-gold pair for each left
record, after dropping tokens held by more than 1% of the right table. An attribute is
only useful as evidence if these two numbers differ. `unusable_` is the fraction of pairs
where at least one side is empty, which is what stops a high-agreement attribute from being
worth much.

**Ground truth kind.** Abt-Buy, DBLP-ACM and DBLP-Scholar ship a complete Leipzig mapping,
so any pair outside it is a genuine non-match. Amazon-Google and Walmart-Amazon ship a
labelled candidate set covering 0.26% and 0.018% of the cross product respectively, so a
pair outside the gold set is usually a pair nobody ever looked at. **A matcher that
correctly finds an unlabelled true match on those two datasets is scored as a false
positive.** Of the near misses we surface, only 20.8% (Amazon-Google) and 10.8%
(Walmart-Amazon) are confirmed non-matches.

## The five at a glance

| Dataset | Domain | A x B | Gold | Density | 1:1 | Ground truth | Sharpest signal |
|---|---|---|---|---|---|---|---|
| dblp-acm | bibliographic | 2,616 x 2,294 | 2,224 | 0.0371% | 100% | complete mapping | title + year |
| dblp-scholar | bibliographic | 2,616 x 64,263 | 5,347 | 0.0032% | 21% | complete mapping | title, after repair |
| abt-buy | products | 1,081 x 1,092 | 1,097 | 0.0929% | 96% | complete mapping | model code containment |
| amazon-google | software | 1,363 x 3,226 | 1,167 | 0.0265% | 74% | labelled candidates | price + semantics |
| walmart-amazon | electronics | 2,554 x 22,074 | 962 | 0.0017% | 77% | labelled candidates | `modelno` equality |

---

## dblp-acm

Two clean bibliographic catalogues of the same five database venues, 2,616 DBLP records
against 2,294 ACM records, with 2,224 gold pairs that are **100% strictly one to one**.
Every attribute is present on essentially every row: the only gap anywhere is 14 missing ACM
author lists (0.61%). This is the easy benchmark, and the published ceiling is 98.4 F1 for
both Magellan and the best deep model.

**Discriminativeness.** `title` 3.87, `authors` 3.53, `year` 1.01, `venue` 1.00. Title and
authors are near-keys; year and venue are five- and ten-valued enumerations that can only
ever act as filters.

**Quirks.** DBLP uses `?` as a null sentinel in `authors` (23 rows) rather than leaving the
field empty. ACM titles are sentence-cased where DBLP titles are title-cased, so any
comparison must lowercase first. ACM venue strings carry HTML entities
(`The VLDB Journal &mdash; The International Journal on Very Large Data Bases`) and ACM
author names carry numeric character references (`Oliver G&#252;nther` for
`Oliver Günther`). The most common titles on both sides are recurring SIGMOD Record columns
— `Editor's Notes` appears 30 times in DBLP, `Reminiscences on Influential Papers` 19 times
— which is where nearly all the false-positive pressure comes from.

**SQL analyses.**

```sql
-- Year is a hard constraint here, and the single cheapest filter available.
SELECT avg(CASE WHEN a.year = b.year THEN 1 ELSE 0 END) FROM gold_pairs;      -- 1.0000
SELECT avg(CASE WHEN a.year = b.year THEN 1 ELSE 0 END) FROM near_miss_pairs; -- 0.1283

-- Venue never agrees as a string, but is a perfect 5-way bijection.
SELECT a_venue, b_venue, count(*) FROM gold_pairs GROUP BY 1, 2 ORDER BY 3 DESC;
```

| DBLP venue | ACM venue | pairs |
|---|---|---|
| SIGMOD Conference | International Conference on Management of Data | 793 |
| VLDB | Very Large Data Bases | 638 |
| SIGMOD Record | ACM SIGMOD Record | 460 |
| VLDB J. | The VLDB Journal &mdash; The International Journal on Very Large Data Bases | 201 |
| ACM Trans. Database Syst. | ACM Transactions on Database Systems (TODS) | 132 |

Exact normalised title equality holds for 91.3% of matches and 0.69% of near misses; mean
title-token Jaccard is 0.977 against 0.134, and the match distribution is 1.0 at every
decile from p10 up. Author-list equality is only 28.3% on matches, because the two sources
abbreviate given names differently and reorder authors.

**Match pattern: ACM keeps the subtitle, DBLP truncates it.** Every hard match is this one
shape, and year and authors carry the pair.

- DBLP: `Mediator Languages - a Proposal for a Standard` / `Peter Buneman, Louiqa Raschid, Jeffrey D. Ullman` / 1997
- ACM: `Mediator languages-a proposal for a standard: report of an I3/POB working group held at the University of Maryland, April 12 and 13, 1996` / `Peter Buneman, Louiqa Raschid, Jeffrey Ullman` / 1997

**Mismatch pattern: the conference paper and its journal extension.** Identical titles, same
authors reordered, different year and venue tier — and they are *not* a match.

- `Lineage Tracing for General Data Warehouse Transformations` / `Yingwei Cui, Jennifer Widom` / VLDB / **2001**
- `Lineage tracing for general data warehouse transformations` / `Y. Cui, J. Widom` / The VLDB Journal / **2003**

**What to tell the matcher.** Require `year` equality. Treat title as the primary evidence
but expect one side to be a prefix of the other. Never compare venue as a string; map it
through the crosswalk above and use it only to distinguish a conference paper from its
journal version. When two records share a generic editorial title, decide on authors and
year alone.

---

## dblp-scholar

The same 2,616 DBLP records, matched against a 64,263-row Google Scholar crawl. 5,347 gold
pairs, and only **21% are one to one**: 1,238 of the 2,408 matched DBLP rows have more than
one Scholar counterpart, averaging 2.22 each, and one has 20. Scholar holds many versions of
the same paper — preprint, proceedings, extended journal — and the mapping links all of
them. Any matcher that assumes a bijection loses most of the recall here.

**Discriminativeness.** `title` 3.89, `authors` 3.34, `venue` 0.94, `year` 0.46. Year scores
worst of any attribute in any of the five datasets, because it is missing from 54.1% of
Scholar rows. Compare the same column on DBLP-ACM, where it scores 1.01 and is a hard
constraint.

**Quirks, and they are the whole story.** Scholar is a raw crawl of the open web, not a
catalogue. Its top venues are `Cochrane Database Syst Rev` (910), `New Directions for Higher
Education` (894) and `Phil. Mag` (832), and only 5,218 of its 64,263 rows appear anywhere in
the gold mapping. Its `authors` column contains scraper artefacts: `ACMS Anthology` (107 rows),
`portal.acm.org` (30), `P Geographer, T Geography` (27). Its `title` column contains page
chrome: `Source ACM SIGMOD Record archive` (19), `Terms of Usage Privacy Policy Code of
Ethics Contact Us` (13). Text is double-encoded, so 3.87% of gold-matched Scholar titles
contain `â??` where a quote should be, and 7,601 venue strings contain an HTML entity.
Whitespace is collapsed in 1.05% of gold-matched titles, producing
`Databasearchitecture optimizedforthenewbottleneck: memoryaccess`. And 4.66% begin with the
tail of the preceding citation rather than the title: `andD. Srivastava. HolisticTwigJoins:
OptimalXMLPatternMatching`. The DBLP side is clean except for one truncated venue value,
`ecord`, which should read `SIGMOD Record`.

**SQL analyses.**

```sql
-- The year trap: Scholar stores year as a float, so string equality never fires.
SELECT avg(CASE WHEN trim(a_year) = trim(b_year) THEN 1 ELSE 0 END) FROM gold_pairs; -- 0.0000
SELECT avg(CASE WHEN abs(cast(a_year AS double) - cast(b_year AS double))
                   / greatest(cast(a_year AS double), cast(b_year AS double)) <= 0.05
            THEN 1 ELSE 0 END) FROM gold_pairs;                                      -- 0.9996

-- How much of the right side is even in scope.
SELECT count(*) FROM b;                                             -- 64,263
SELECT count(DISTINCT b_row) FROM gold;                             --  5,218
```

Year agreement is *literally zero* on matches by string comparison and *99.96%* by numeric
comparison, purely because DBLP writes `2002` and Scholar writes `2002.0`. Exact title
equality holds for 54.2% of matches against 1.47% of near misses; mean Jaccard is 0.851
against 0.186. Venue agrees on 26.1% of matches, authors on 28.8%.

**Match pattern: the same paper, one side mangled by the crawler.** Token Jaccard is exactly
0.000 and the records are still the same paper. Author overlap is the only surviving signal.

- `Database Architecture Optimized for the New Bottleneck: Memory Access` / `P Boncz, S Manegold, M Kersten` / VLDB / 1999
- `Databasearchitecture optimizedforthenewbottleneck: memoryaccess` / `P Boncz, S Manegold, M Kersten` / `Proc. 25th International Conference on Very Large Data Bases &hellip;,` / *(no year)*

**Mismatch pattern: the recurring column, decades apart.** Identical title, and the year is
the only thing separating them.

- `Reminiscences on Influential Papers` / *(no authors)* / 2002
- `Reminiscences on Influential Papers` / `E Bertino` / `ACM Transactions on Database Systems,` / **1976.0**

**What to tell the matcher.** Strip HTML entities and repair mojibake before comparing
anything. Compare titles with a character-level or subword measure, not a word-token one,
because whitespace is unreliable. Ignore a leading `and<Initial>. <Surname>.` fragment.
Cast year to a number and allow one year of slack; never string-compare it. Allow one DBLP
record to match many Scholar records. Expect most of the right table to be irrelevant.

---

## abt-buy

Two retail catalogues, 1,081 Abt products against 1,092 Buy.com products, 1,097 gold pairs,
96.2% one to one. DeepMatcher classifies this as a **textual** rather than structured task,
and it is the only one of our five it treats that way. It is also the dataset where our own
name-similarity signal is weakest: exact normalised name equality holds for **1.46%** of
matches, and mean name-token Jaccard is 0.432, with a p10 of 0.20.

**Discriminativeness.** `name` 3.98, `description` 2.91, `price` 1.12. `manufacturer` exists
only on the Buy side (116 distinct values, `Sony` 170, `Panasonic` 99, `Canon` 86), so it can
constrain a candidate but can never be compared across the pair.

**Quirks.** Abt writes one long marketing name and then repeats it verbatim at the head of a
slash-delimited spec list, so `description` averages 249 characters and is 100% present but
almost entirely redundant with `name`. Buy writes a terse name and a 57-character spec
fragment that is missing 40.4% of the time and is often just `Black` (11 rows). Price is a
dollar-prefixed string missing from 61.3% of Abt rows and 46.0% of Buy rows, so 79.4% of
gold pairs have no comparable price at all. Both sides put the model number in the name, but
in different shapes: Abt appends ` - MDREX55WH`, Buy interleaves it as `MDR EX55/WHI`.

**SQL analyses.** This is the finding that matters most in the whole document.

```sql
-- Exact model-code equality, against containment after stripping every separator.
WITH coded AS (
    SELECT
        filter(split(regexp_replace(lower(a_name), '[^a-z0-9]+', ' '), ' +'),
               t -> length(t) >= 4 AND t rlike '[0-9]')          AS a_codes,
        regexp_replace(lower(b_name), '[^a-z0-9]+', '')          AS b_squash
    FROM gold_pairs
)
SELECT avg(CASE WHEN exists(a_codes, c -> contains(b_squash, c)) THEN 1 ELSE 0 END) FROM coded;
```

| measure | matches | near misses |
|---|---|---|
| both sides carry a code | 0.8541 | 0.8445 |
| codes are exactly equal | 0.4731 | 0.0144 |
| **one code contained in the other** | **0.8195** | **0.0246** |

Containment finds 82% of matches where exact equality finds 47%, at a false-positive rate of
2.5%. It is a stronger discriminator than the entire name. Price, where it exists, is worth
having: median relative gap 0.175 on matches against 0.585 on near misses, 61.5% within 25%
against 20.1%.

**Match pattern: the same model code, punctuated differently.** Token Jaccard 0.000; the code
is the only link.

- Abt: `Sony White Earbud Style Headphones - MDREX55WH`
- Buy: `Ex Series Earbuds Wht - MDR EX55/WHI`

**Mismatch pattern: colour and variant siblings.** The near misses are almost all a product
family where one character of the code encodes the variant.

- Abt `Omnimount TV Top Shelf Mount - CCH1B` (black finish) against Buy `OmniMount TV Top Shelf Mount - CCH1P` (platinum), Jaccard 0.714
- Abt `Sony DVP-FX820 Black 8' Portable DVD Player - DVPFX820` against Buy `Sony DVP-FX820/L Portable DVD Player - DVPFX820/L`, Jaccard 0.875
- Abt `Canon EOS Rebel XSi Silver Digital SLR Camera - XSIREB1855S` against Buy `Canon EOS Rebel XSi Digital SLR Camera - Silver - 2757B001`, Jaccard 0.800, where the two sides use unrelated part numbers for what may or may not be the same camera

**A labelling quirk worth knowing.** Buy contains three byte-identical rows named
`LG 2.0 cu.ft. Over the Range Microwave Oven` (ids 208156877, 208156878, 208156879), and the
gold mapping assigns each to a *different* Abt colour variant — `LMVM2085BK`, `LMVM2085WH`
and `LMVM2085SS` respectively. Nothing in the text distinguishes them. A matcher that pairs
all three the same way scores one true positive and two false positives no matter which way
it chooses.

**What to tell the matcher.** Extract every digit-bearing token of four characters or more
from both names, strip all punctuation and whitespace from both names, and treat substring
containment of a code as near-decisive evidence. Then check the variant suffix character by
character before accepting: a one-character difference in a model code means a different
colour or capacity, not a match. Ignore Abt's description as evidence — it is the name
again. Use price only when both sides have one.

---

## amazon-google

1,363 Amazon software listings against 3,226 Google Shopping listings, 1,167 gold pairs,
74.0% one to one, with up to 5 Google listings per Amazon product. This is the dataset with
the largest published gap between string-similarity and semantic matching: Magellan reaches
49.1 F1 and DeepMatcher's Hybrid model reaches 69.3.

**Discriminativeness.** `title` 3.85, `price` 1.52, `manufacturer` 0.57 — the second lowest
score of any attribute in any of the five datasets, behind only DBLP-Scholar's year, because
Google's `manufacturer` is **88.96% missing**.

**Quirks.** Google titles are constructed differently from Amazon's: they prepend the
publisher and an internal SKU, then abbreviate aggressively —
`intuit inc 284216 qckbks prem nonprofit ed 2005` for Amazon's
`quickbooks premier non-profit edition 2005`. Both sides have been pre-tokenised by whoever
built the dataset, so punctuation is spaced out into its own tokens: `( r )`,
`( jewel case )`, `clipart & more 250000 ( jc )`. Google's `manufacturer`, when present, is
hyphen-joined (`sony-pictures-digital-entertainment`, `global-software-publishing`) where
Amazon's is spaced. Many Google rows are not products at all but stationery listings —
`blank laser checks` (8), `deposit slips` (6), `multipurpose continuous checks` (6).

**SQL analyses.**

```sql
-- Manufacturer is the strongest attribute and almost never available.
SELECT avg(CASE WHEN trim(a_manufacturer) = trim(b_manufacturer) THEN 1 ELSE 0 END),
       avg(CASE WHEN nullif(trim(b_manufacturer), '') IS NULL THEN 1 ELSE 0 END)
FROM gold_pairs;   -- agreement 0.8029 when comparable, unusable 0.8218 of the time

-- Model codes do not exist in this domain.
-- both sides carry a code: 0.1525 of matches
```

| measure | matches | near misses |
|---|---|---|
| exact title | 0.0540 | 0.0040 |
| mean title Jaccard | 0.5124 | 0.1588 |
| manufacturer agrees | 0.8029 | 0.1610 |
| manufacturer unusable | 0.8218 | 0.9020 |
| price within 25% | 0.7992 | 0.2397 |
| median relative price gap | 0.1074 | 0.5348 |

Price is the most useful non-title attribute of any product dataset here: 79.9% of matches
are within 25% against 23.97% of near misses, and it is only unusable on 10.4% of pairs.

**Match pattern: abbreviation and reordering, with no code to fall back on.**

- Amazon: `emc retrospect 7.5 professional for windows upgrade` / `dantz`
- Google: `retrospect 7.5 prof win upg only pz24a0075` / `emc` / $54.68

This is DeepMatcher's "linguistic variations of domain-specific terms" category exactly:
`professional` to `prof`, `windows` to `win`, `upgrade` to `upg`, and the manufacturer has
moved from one field to the other.

**Mismatch pattern: the version number is the whole difference.**

- Amazon: `photo explosion deluxe **3.0**` / `nova development` / $49.99
- Google: `photo explosion deluxe ( r ) **2.0**` / $49.99

**Read precision carefully here.** Only 0.26% of the cross product was ever labelled, and
only **20.8%** of the near misses we surface are confirmed non-matches. Several of the
highest-Jaccard "non-matches" look like true matches nobody labelled —
`elementary school success deluxe 2006` appears identically on both sides, as does
`myinvoices & estimates deluxe` and `clifford the big red dog thinking adventures`.

**What to tell the matcher.** Expect abbreviations and expand them: `prem` premier, `ed`
edition, `upg` upgrade, `jc` jewel case, `win` windows. Expect the publisher to appear inside
the Google title rather than in its own field, and treat an Amazon `manufacturer` value found
anywhere in the Google title as agreement. Compare version numbers and edition years
exactly — they are the most common reason a near-identical pair is not a match. Use price as
a tie-breaker with a 25% tolerance. Do not look for model codes; only 15% of pairs have one.

---

## walmart-amazon

2,554 Walmart electronics against 22,074 Amazon listings, 962 gold pairs, 77.3% one to one.
This is the one structured dataset where Magellan *beats* every deep model (71.9 against
67.6), and the measurements explain why: it has a near-key attribute that rewards exact
comparison and punishes generalisation.

**Discriminativeness.** `title` 3.97, `modelno` 3.29, `price` 1.95, `brand` 1.24,
`category` 1.00.

**SQL analyses.** `modelno` is the sharpest exact-equality signal anywhere in these five
datasets.

```sql
SELECT
    avg(CASE WHEN lower(trim(a_modelno)) = lower(trim(b_modelno)) THEN 1 ELSE 0 END) AS agree,
    avg(CASE WHEN nullif(trim(b_modelno), '') IS NULL THEN 1 ELSE 0 END)             AS unusable
FROM gold_pairs;   -- agree 0.6784, unusable 0.3181
```

| measure | matches | near misses |
|---|---|---|
| `modelno` agrees | **0.6784** | **0.0026** |
| `modelno` unusable | 0.3181 | 0.3744 |
| `brand` agrees | 0.8654 | 0.4146 |
| `category` agrees | 0.0443 | 0.0221 |
| exact title | 0.0644 | 0.0000 |
| mean title Jaccard | 0.5902 | 0.1947 |
| price within 25% | 0.6346 | 0.2200 |

Brand is necessary but nowhere near sufficient: it agrees on 41% of near misses. Category is
worthless as a comparison — Walmart uses 63 coarse buckets and Amazon 706 fine ones, so they
agree on 4.4% of true matches, barely above the 2.2% they agree on for non-matches.

**Quirks.** Amazon's `modelno` is missing from 28.5% of rows and, where present, is often not
a model number at all but leftover descriptive text: `high power` (8 rows),
`cosmopolitan electrol` (6), `with csr` (6), `high contrast matte white` (5),
`with keystone eliminator` (5). Walmart's `price` uses `0.0` as a null sentinel in 64 rows
while Amazon leaves price genuinely null in 13.0% of rows and never writes zero. Walmart's
`category` is frequently wrong rather than merely coarse:
`hp c7973w ultrium 800 gb worm data cartridge` is filed under `mp3 accessories`, and
`lorex lw1002 live wireless security cameras` under `garden - general`.

**Match pattern: Walmart's terse title against Amazon's keyword-stuffed one, joined by
`modelno`.** Token Jaccard 0.067 and the model numbers are identical.

- Walmart: `hp cb40 toner 7500 page-yield` / printers / hp / **cb400a** / $249.23
- Amazon: `hp color laserjet cb400a black print cartridge in retail packaging` / laser printer toner / hp / **cb400a** / $145.99

Note that the title itself is wrong on the Walmart side — `cb40` is a truncation of `cb400a`.

**Mismatch pattern: one character of the model number.** This is DeepMatcher's "similar but
semantically different tokens" category, and it is the dominant failure mode here.

- `microsoft wireless laser desktop 3000 keyboard and mouse combo` / **nud-00001** against
  `microsoft wireless desktop 3000 keyboard and mouse combo` / **mfc-00001** (Jaccard 0.889)
- `buffalo technology drivestation axis **1.5 tb** usb 2.0 desktop external hard drive` /
  **hd-lb1 .5 tu2** against the same title reading **1 tb** / **hd-lb1 .0 tu2** (Jaccard 1.000,
  different capacity)
- `ampad evidence glue top narrow ruled pads` / **21118** against the same title with
  **21218** (Jaccard 0.857)

**Read precision carefully here too.** Only 0.018% of the cross product was labelled — the
sparsest of the five — and only **10.8%** of the near misses are confirmed non-matches.

**What to tell the matcher.** Compare `modelno` first and treat equality as decisive; treat
inequality as decisive too, but only after checking that the Amazon value is a real code and
not descriptive text. Require `brand` agreement as a gate and nothing more. Ignore
`category` entirely. Treat Walmart `price = 0.0` as missing. When only titles are available,
extract the code from the title and compare it character by character, because a single
changed digit reliably means a different capacity, colour or revision.

---

## Cross-dataset summary for prompting

Five things hold across all of them.

**The name is never enough on its own, and never in the same way twice.** Mean name-token
Jaccard on true matches ranges from 0.43 (abt-buy) to 0.98 (dblp-acm). A single similarity
threshold cannot serve both.

**The most useful attribute is usually the one with a controlled format, not the one with
the most text.** `modelno` on walmart-amazon separates matches from near misses by
0.6784 against 0.0026. Extracted model codes on abt-buy separate them by 0.8195 against
0.0246. Neither is the longest field.

**Exact string equality understates every attribute.** Venue on dblp-acm agrees on 0% of
matches as a string and 100% through a five-row crosswalk. Year on dblp-scholar agrees on 0%
as a string and 99.96% as a number. Model codes on abt-buy agree on 47% exactly and 82% by
containment. Always normalise, then compare.

**Near misses differ from matches on exactly one attribute, and it is usually short.** A
colour suffix, a version number, a capacity digit, a publication year. Whatever field carries
that distinction has to be compared exactly even when everything else is compared loosely.

**Two of the five cannot be scored fairly at face value.** Amazon-Google and Walmart-Amazon
label 0.26% and 0.018% of their cross products, so a correct match outside the label set
counts against precision. When SERF's precision on those two looks worse than on abt-buy, a
share of the gap is the benchmark, not the matcher.
