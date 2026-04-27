# Data Curation Workflow

Working spec for the full 80B-token data pipeline. This is the living reference across sessions — update as decisions are made and work completes.

Last updated: 2026-04-10

---

## 0. Scope and Orientation

**Goal:** 80B tokens across three curriculum phases for luxia-base 3B.
**Tokenizer:** SmolLM2 (49,152 vocab, `HuggingFaceTB/SmolLM2-135M`).
**Final format:** Packed uint16 binary with document-boundary attention masks.
**Final size:** ~160GB tokenized (plus metadata).
**Philosophy:** Research model. No commercial deployment. Data choices optimize for representational richness and conversational quality, not benchmark scores or legal defensibility.

### What "data curation" means here

It's not one task — it's six distinct jobs that happen to feed into each other:

1. **Sourcing** — finding and downloading raw data
2. **Cleaning** — removing garbage, boilerplate, PII
3. **Filtering** — scoring quality and selecting what stays
4. **Deduplication** — removing redundant content within and across sources
5. **Assembly** — mixing sources into phase-specific token budgets
6. **Packing** — tokenizing, shuffling, packing sequences with doc boundaries

---

## 1. Source Inventory

### Phase 1: Structural Scaffolding (15B tokens, 0–15B)

Data with strong statistical symmetries (syntactic, temporal, spatial, algebraic) to build routing frames.

| Category | Source | Target tokens | Acquisition | Status |
|----------|--------|--------------|-------------|--------|
| Code | Stack v2 (permissive, multi-paradigm) | 5B | HF `bigcode/the-stack-v2-dedup` | NOT STARTED |
| Math/Science | peS2o + OpenWebMath | 3.5B | HF `allenai/peS2o`, `open-web-math/open-web-math` | NOT STARTED |
| Temporal | CC-News (Dolma) + Hansard (early periods) | 2.5B | HF `allenai/dolma` CC-News subset; Hansard API/parlparse | NOT STARTED |
| Encyclopedic | Wikipedia + Wikidata (CC0 portion) | 2B | HF `wikimedia/wikipedia` | NOT STARTED |
| Procedural/Spatial | FineWeb procedural + Wikipedia geo-filtered | 2B | HF `HuggingFaceFW/fineweb` | NOT STARTED |

### Phase 2: Diversity and Conversation (40B tokens, 15B–55B)

Varied registers, genuine dialogue, reasoning-heavy text for geometric expansion.

| Category | Source | Target tokens | Acquisition | Status |
|----------|--------|--------------|-------------|--------|
| Parliamentary debate | Hansard + US Congress + Europarl | 4B | Hansard API; govinfo.gov bulk XML; HF europarl | NOT STARTED |
| Human conversation | WikiConv + WildChat (filtered) + OASST + SODA | 7B | HF for all four | NOT STARTED |
| Register-filtered web | FineWeb via propella-1 (conv/narrative/argument) | 8B | HF `HuggingFaceFW/fineweb` + TurkuNLP classifier | NOT STARTED |
| Legal reasoning | Pile of Law (court opinions + hearings) | 5B | HF `pile-of-law/pile-of-law` | NOT STARTED |
| Literature | PG-19 (plays, novels, essays, letters) | 4B | HF `deepmind/pg19` | NOT STARTED |
| StackExchange | Threads with comments, edits, critique | 4B | Dolma SE subset or direct dump | NOT STARTED |
| Academic | peS2o (humanities) + PMC OA | 3B | HF | NOT STARTED |
| Correspondence | Enron emails + PG letters | 1B | HF `talby/enron-emails` + PG | NOT STARTED |
| Code (maintained) | Stack v2 | 2.5B | Same as Phase 1 | NOT STARTED |
| Math (maintained) | OpenWebMath | 1.5B | Same as Phase 1 | NOT STARTED |

### Phase 3: Conversational Annealing (25B tokens, 55B–80B)

Highest-quality subset for consolidation. **OPEN QUESTION: how much synthetic/SFT-shaped data belongs here?**

| Category | Source | Target tokens | Acquisition | Status |
|----------|--------|--------------|-------------|--------|
| Best conversation (upsampled) | Hansard best + WikiConv best + WildChat long + OASST top | 8B | Quality-filtered from Phase 2 sources | NOT STARTED |
| Synthetic multi-turn | SmolTalk 2 + Dolci | 5B | HF `HuggingFaceTB/smoltalk` | NOT STARTED |
| Premium writing | PG-19 literary + peS2o humanities + speeches | 5B | Quality-filtered from Phase 1/2 | NOT STARTED |
| Code reasoning | OpenCodeReasoning + Stack-Edu | 3.5B | HF `nvidia/OpenCodeReasoning` | NOT STARTED |
| Math reasoning | FineMath 4+ | 3.5B | HF `HuggingFaceTB/finemath` | NOT STARTED |

### Sources requiring non-trivial acquisition work

These can't be solved with a single `load_dataset` call:

- **UK Hansard**: TheyWorkForYou API or `mysociety/parlparse` XML. Needs scraper/parser.
- **US Congressional Record**: govinfo.gov bulk XML. Needs downloader + text extraction.
- **Propella-1 FineWeb filtering**: Batch GPU inference over FineWeb sample. Longest single processing task.
- **WikiConv**: Older dataset, may need markup cleanup.
- **Federal Register / SEC EDGAR / USPTO**: Government procedural text (potential Phase 1 supplement). Available via bulk APIs but need custom downloaders.

---

## 2. Processing Pipeline

Each source flows through these stages. Not every source needs every stage (e.g., FineWeb already ran heuristic filters).

```
Download/Stream
  → Language ID (fastText lid.176.bin)
  → Source-specific cleaning (see §3)
  → Heuristic quality filters (line/word stats, boilerplate removal)
  → PII scrubbing (regex: emails, phones, SSNs, IPs)
  → Quality scoring (classifier scores as metadata, threshold later)
  → Evaluation decontamination (13-gram overlap vs eval suite)
  → Tokenization (SmolLM2, uint16)
  → Cross-source MinHash LSH dedup
  → Phase assembly (mix by token budget)
  → Sequence packing with doc-boundary attention masks
  → Shuffle
  → Ship to training nodes / push to aethera-gp on HF
```

### Tooling

| Tool | Purpose |
|------|---------|
| **datatrove** (HuggingFace) | Primary pipeline framework — extraction, filtering, dedup, tokenization, packing |
| **text-dedup** (Google) | MinHash LSH if more control needed than datatrove provides |
| **fastText lid.176.bin** | Language identification |
| **KenLM** | Perplexity-based quality scoring (optional, for sources without existing scores) |
| **lm-eval-harness** | Decontamination tooling (13-gram matching) |
| **TurkuNLP register classifier** | Propella-1 FineWeb register filtering |

---

## 3. Source-Specific Cleaning Notes

**Web crawl (FineWeb/Dolma):** Already filtered by upstream. If pulling from base FineWeb for propella-1, inherit upstream filters + add register classifier. Score distributions before thresholding.

**Code (Stack v2):** Filter by permissive license allowlist (MIT, Apache-2.0, BSD-2/3, ISC, Unlicense, CC0). Drop `no_license`. Remove auto-generated files, minified JS, config/data dumps, lockfiles. Scan for secrets (API keys, SSH keys, credentials). Size filters.

**Conversation (WildChat):** Heavy filtering needed. Remove: bot-generated turns, single-turn "hi/thanks", prompt injection attempts, non-English conversations, NSFW where undesirable. This is the dirtiest source.

**Parliamentary (Hansard/Congress):** Extract speech text from XML/JSON. Handle speaker attribution. Remove procedural boilerplate (roll calls, division metadata, session headers).

**Books (PG-19):** Strip Project Gutenberg headers/footers/license blocks. OCR artifact cleanup. Careful dedup — PG has many republications.

**Legal (Pile of Law):** Filter by subset (court opinions and congressional hearings are high quality; regulations and contracts vary). Already reasonably clean.

**Academic (peS2o):** Already well-processed by AI2. Filter for humanities/social science for Phase 2 diversity vs. STEM for Phase 1.

**Email (Enron):** Remove forwarded chain duplicates. Strip email headers. Small dataset (~500K emails).

**StackExchange:** Extract question + answer + comment threads. Preserve edit history structure if feasible. Score by vote count for Phase 3 upsampling.

**Synthetic/SFT (SmolTalk, OpenCodeReasoning):** These are assistant-shaped. Conscious decision needed: does this belong in base pretraining or post-training? Risk of making the base model too instruction-shaped, which may conflict with geometry-first goals. **DECIDE BEFORE PHASE 3.**

---

## 4. Metadata Schema

Every processed document carries:

```
doc_id:           str   # unique identifier
source:           str   # dataset name (e.g., "fineweb", "hansard", "pg19")
subset:           str   # source-specific subset (e.g., "court_opinions", "CC-News")
upstream_url:     str   # original URL or identifier where available
timestamp:        str   # document date if known
raw_hash:         str   # hash of raw text before cleaning
normalized_hash:  str   # hash after normalization
language:         str   # fastText language ID
token_count:      int   # after tokenization
quality_scores:   dict  # classifier scores (fineweb-edu score, propella-1 register, etc.)
pii_flags:        list  # types of PII detected and scrubbed
dedup_cluster_id: str   # MinHash cluster assignment
phase_assignment: str   # which training phase this doc is allocated to
```

Purpose: **debugging, not legal defense.** When the model produces weird outputs at step 40B, trace back to what was in the data.

---

## 5. Key Decisions (Open)

### Must decide before starting

- [ ] **Storage location for raw/intermediate data.** Options: local NVMe on processing machine, external drive, object storage (Backblaze B2 / Cloudflare R2), HF Hub. Need ~1-2TB for intermediates, ~160GB for final tokenized.
- [ ] **Processing machine.** Can pipeline run on the B200 training nodes, or do we use a separate CPU machine? datatrove is CPU-bound for everything except propella-1.
- [ ] **Pilot scale.** Run the full pipeline on a small subset first (1-5B tokens?) to validate before committing to 80B.

### Must decide before Phase 3

- [ ] **Synthetic/SFT data in base pretraining.** How much SmolTalk/OpenCodeReasoning/assistant-shaped data belongs in the base model vs. post-training? This is a philosophical question about what the base model should be.
- [ ] **Propella-1 thresholds.** Score FineWeb, inspect distributions, then decide thresholds. Don't threshold blindly.

### Revisit during assembly

- [ ] **Token budget rebalancing.** The spec's 15/40/25B split and per-source allocations are estimates. Actual available tokens after filtering/dedup may shift these.
- [ ] **Context scheduling interaction.** Phase 3 coincides with 2048→4096 context length change. Packing strategy needs to account for this.
- [ ] **Government procedural sources.** Federal Register / SEC EDGAR / USPTO as Phase 1 supplements — worth the scraping effort?

---

## 6. Work Order

Rough sequencing. Steps within a tier can be parallelized.

### Tier 0: Setup
- [ ] Set up datatrove environment (venv, dependencies)
- [ ] Set up storage (decide location, create directory structure)
- [ ] Build tokenization harness (SmolLM2 tokenizer → uint16 binary, matching proxy format)
- [ ] Build metadata tracking (schema above, probably JSONL sidecar files)

### Tier 1: Easy sources + pipeline validation
- [ ] Download + process one small source end-to-end (e.g., OASST or Enron) to validate the full pipeline
- [ ] Process trivial HF sources: Wikipedia, peS2o, OpenWebMath, FineMath, OASST, SODA, PG-19, WildChat
- [ ] Process Dolma subsets: CC-News, StackExchange

### Tier 2: Non-trivial sources
- [ ] Build Hansard scraper/parser
- [ ] Build Congressional Record downloader + text extractor
- [ ] Process Stack v2 (license allowlist, secret scanning, code-specific filters)
- [ ] Process Pile of Law (subset filtering)
- [ ] WikiConv cleanup

### Tier 3: Compute-heavy filtering
- [ ] Run propella-1 register classifier on FineWeb sample → inspect score distributions → threshold → produce FineWeb-Conv subset
- [ ] Run FineWeb-Edu score filtering for Phase 3 premium writing
- [ ] Quality-score all sources for Phase 3 upsampling decisions

### Tier 4: Assembly
- [ ] Cross-source MinHash LSH dedup (all tokenized sources combined)
- [ ] Evaluation decontamination pass
- [ ] Phase assembly: mix tokenized shards by token budget
- [ ] Sequence packing with document boundary masks
- [ ] Validation: spot-check samples from each phase, run basic stats
- [ ] Ship to training nodes

### Tier 5: Pilot validation (interleave with above)
- [ ] After Tier 1, train a small proxy on the partial mix to sanity-check
- [ ] After Tier 4, train a ~1B token pilot before committing to the full 80B

---

## 7. Lessons / Cautions

Things to remember across sessions:

- **Propella-1 is one signal, not gospel.** Score → inspect → ablate → threshold. Same for FineWeb-Edu scores and any other classifier.
- **Cross-source dedup matters.** FineWeb, Dolma, peS2o, and StackExchange overlap heavily. Dedup within each source is not sufficient.
- **Evaluation decontamination.** Strip benchmark text before finalizing. 13-gram overlap matching via lm-eval-harness.
- **Phase 3 synthetic risk.** Assistant-shaped data in base pretraining can conflict with geometry-first goals. Decide consciously.
- **Token budgets are aspirational.** Actual yields after filtering/dedup will differ. Some sources will be smaller than planned. Have fallback plans (more FineWeb, more Dolma).
- **PII scrubbing is an ethical obligation**, not a legal one. Do it properly for code, legal, email, and conversation sources especially.
- **Stream, don't hoard.** Download → process → save processed → delete raw. Don't try to store all raw data simultaneously.
- **Metadata for debugging.** When training goes weird, you need to trace back to data. The schema in §4 is the minimum.
