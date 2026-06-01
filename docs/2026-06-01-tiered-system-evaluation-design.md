# Tiered System Evaluation Design

**Date:** 2026-06-01  
**Status:** Draft for review  
**Scope:** 1–2 day diagnostic on 8–12 papers, mixed research + engineering goals, no human raters (proxy/automatic metrics only)

## 1. Goals and Non-Goals

### Goals

- Decide whether the system is **reliable enough for practical use** today.
- Produce **actionable engineering fixes** (top 3) from failure patterns.
- Collect **publishable-adjacent evidence** (tables, pass rates, cost/latency) without a full benchmark study.
- Answer: *Should we evaluate every subsystem, or mainly outputs?* → **Mainly outputs + system KPIs on all runs; internals only on failures/outliers.**

### Non-Goals (this cycle)

- Human expert rating of review quality (no raters available).
- Full component-by-component certification of MinerU, PASA, Neo4j KG, every tool implementation.
- Longitudinal regression dashboard (ongoing eval).
- Comparing against external baselines (optional stretch only if time remains).

## 2. Evaluation Architecture (Approach 3 — Tiered)

```text
                    ┌─────────────────────────────────────┐
                    │  Tier 1: All runs (8–12 papers)      │
                    │  Outputs + system KPIs (automatic) │
                    └─────────────────┬───────────────────┘
                                      │
                    pass thresholds?  │  fail / outlier
                                      ▼
                    ┌─────────────────────────────────────┐
                    │  Tier 2: Triggered deep-dive         │
                    │  events.jsonl, state, tool traces    │
                    │  Root-cause bucket + fix hypothesis    │
                    └─────────────────────────────────────┘
```

### Tier 1 — Run on every job

Collect from `data/jobs/<job_id>/` without opening source code:

| Dimension | Primary artifacts |
|-----------|-------------------|
| Reliability | `state.json` (status, error), `events.jsonl` |
| Cost / usage | `state.json` → `usage.token`, `usage.tool`, `usage.paper_search` |
| Latency | `events.jsonl` timestamps (`created` → `completed` / `failed`) |
| Output presence | `final_report.md`, `final_report.pdf` |
| Evidence footprint | `annotations.json`, annotation count in state |
| Output structure (proxy) | `final_report.md` section headers, word counts |

### Tier 2 — Only when Tier 1 flags failure or outlier

Inspect in order:

1. `events.jsonl` — last successful phase before failure; repeated events.
2. `state.json` — `metadata` (MinerU provider, paper_search runtime, fast mode, KG resolution).
3. `prompt_snapshot` (if present) — config actually used.
4. Tool progression — counts from state/events: `pdf_search`, `pdf_read_lines`, `pdf_annotate`, `paper_search`, `review_final_markdown_write`.
5. Optional: spot-check 3–5 `pdf_annotate` spans against `mineru` markdown page index (grounding sample).

Assign one **root-cause bucket** per failed/outlier run (see Section 7).

## 3. Test Corpus (8–12 Papers)

### Composition (target mix)

| Bucket | Count | Purpose |
|--------|-------|---------|
| Short / clean PDF (≤15 pages) | 2–3 | Baseline success rate |
| Long PDF (30+ pages) | 2 | Stress parse + context limits |
| Dense math / tables | 1–2 | MinerU + annotation alignment stress |
| Non-English or mixed language | 1 | i18n / font / prompt behavior |
| Venue in Neo4j KG (if configured) | 2–3 | KG-grounded report sections |
| Venue **not** in KG | 2 | Fallback pipeline behavior |
| Optional: retrieval-heavy topic | 1–2 | `paper_search` path (if enabled) |

### Fixed configuration matrix

Run **one primary config** on all papers, then **optional** second pass on 2 papers only if time allows:

| Profile | Purpose |
|---------|---------|
| **Primary:** `REVIEW_FAST_MODE=true`, retrieval as in your `.env` | Matches quick diagnostic intent |
| **Stretch:** `REVIEW_FAST_MODE=false` on 2 hard papers | Compare quality vs cost delta |

Record per run: `job_id`, paper label, page count, profile, git commit hash, `.env` snapshot (redact secrets).

## 4. Metrics, Thresholds, and Scoring Rubric

### 4.1 System reliability (automatic)

| Metric | How to measure | Pass (per run) | Aggregate pass |
|--------|----------------|----------------|----------------|
| Job completion | `status == completed` | Yes | ≥ 85% (7/8 min) |
| Final markdown | `final_report.md` exists, size > 2 KB | Yes | 100% of completed |
| Final PDF | `final_report.pdf` exists | Yes | ≥ 90% of completed |
| No fatal error | `error` null or PDF-only warning | Yes | ≥ 85% |
| Final write gate | Event `review_final_markdown_write` success or `final_report_ready` | Yes | 100% of completed |

**Outlier triggers (Tier 2):** any failed job; completion time > 2× median; token total > 2× median.

### 4.2 Cost and efficiency (automatic)

| Metric | Source | Note threshold (flag, not fail) |
|--------|--------|----------------------------------|
| Wall-clock minutes | events | > 45 min (fast mode) or > 120 min (full) |
| Total tokens | `usage.token.total_tokens` | > 2× corpus median |
| Tool calls | `usage.tool.total_calls` | > 150 or < 5 (stuck / noop) |
| `paper_search` calls | `usage.paper_search` | 0 when enabled + gates on |
| Resume attempts | events `agent_run_incomplete` | ≥ 2 on same job |

### 4.3 Research / output quality (proxy — no human raters)

Score each completed run **0–2** per dimension (0 = fail, 1 = partial, 2 = pass). Sum → **Output Quality Index (OQI)** max 10.

| ID | Dimension | Proxy check (automatic + quick manual spot-check) |
|----|-----------|---------------------------------------------------|
| Q1 | **Structure** | Required sections present in `final_report.md`: summary, strengths, weaknesses, suggestions (or section-mode equivalents); ≥ 800 words |
| Q2 | **Coverage** | Mentions method + experiments + limitations (keyword/heuristic or section headers) |
| Q3 | **Specificity** | ≥ 8 annotations; ≥ 3 distinct pages annotated; comments avg length > 40 chars |
| Q4 | **Grounding (sample)** | Random 3 annotations: quoted span appears in page lines from mineru markdown (exact or high overlap) |
| Q5 | **Actionability** | ≥ 3 distinct suggestion bullets; no empty/generic-only blocks (heuristic: < 30% lines are placeholders like "TBD", "N/A") |

**Tier 1 pass for quality:** OQI ≥ 7/10 on a run; corpus median OQI ≥ 7.

**Tier 2 trigger:** OQI ≤ 5 or Q4 grounding fails on 2+ of 3 samples.

### 4.4 Coding / implementation health (lightweight, corpus-level)

Not per-paper — run once on the repo + test suite:

| Check | Command / action | Pass |
|-------|------------------|------|
| Unit tests | `pytest tests/` | All pass |
| Fast mode contract | `test_review_fast_mode.py` | Pass |
| Retrieval disabled | `test_retrieval_disabled_mode.py` | Pass |
| CLI smoke | `python main.py submit --pdf <tiny.pdf> --wait-seconds 0` | Returns job_id, status queued/running |

**Do not** deep-audit every file in 1–2 days; only expand if Tier 2 points to a specific module (e.g., `review_tools.py` gate logic).

## 5. Two-Day Run Protocol

### Day 1 — Execute and collect Tier 1

| Block | Time | Actions |
|-------|------|---------|
| Setup | 30 min | Fix `.env`, record commit, prepare paper list + labels, ensure `data/` writable |
| Batch submit | 1–2 hr | Submit 8–12 PDFs; `watch` or poll until terminal state |
| Harvest | 1 hr | Script or spreadsheet: scrape `state.json` + events for metrics table |
| Spot-check | 1 hr | Q4 grounding sample on 3 annotations × 3 random completed jobs |
| Triage | 30 min | Mark Tier 2 candidates (failed + outliers + OQI ≤ 5) |

### Day 2 — Tier 2 + synthesis

| Block | Time | Actions |
|-------|------|---------|
| Deep-dive | 2–3 hr | Per Tier 2 job: timeline + root-cause bucket |
| Coding checks | 30 min | `pytest`, CLI smoke |
| Report | 1–2 hr | Fill summary template (Section 8); rank top 3 fixes |
| Optional stretch | 1 hr | 2 papers full mode OR 1 baseline comparison |

### Per-job run sheet (copy per row)

```text
job_id:
paper_id / label:
pages (approx):
config: fast|full, retrieval: on|off, provider:
status:
wall_clock_min:
tokens_in / out / total:
tool_calls_total:
pdf_annotate_count:
paper_search_calls:
OQI (0-10):
tier2_needed: Y/N
root_cause_bucket:
notes:
```

## 6. Tier 2 Deep-Dive Playbook

### Event timeline checklist

Look for this sequence; note where it breaks:

```text
created → llm_api_mode_selected → (pdf_uploading_to_mineru) → (pdf_parsing)
→ paper_search_runtime_state_resolved → review_criteria_resolved (optional)
→ agent_running → annotation_created* → (review_final_markdown_write)
→ completed | failed | completed_recovered
```

### Root-cause buckets

| Bucket | Signals | Typical fix area |
|--------|---------|------------------|
| R1 Parse / MinerU | Stuck in `pdf_parsing`, empty markdown, `markdown_parse_warning` | `adapters/mineru.py`, token, fallback |
| R2 Agent loop / gates | Many turns, no `final_report_ready`, `agent_run_incomplete` | `runner.py` resume, prompt, max_turns |
| R3 Final write | Annotations OK but no final md | `review_final_markdown_write` gates |
| R4 Retrieval | `paper_search_not_started`, health failed | `.env` provider, PASA/DeepXiv |
| R5 Grounding | Q4 fails — span not in page text | `pdf_search` before `pdf_annotate` prompt/tool |
| R6 PDF export only | `completed_recovered`, pdf_error set | `report/review_report_pdf.py` |
| R7 KG / criteria | `review_criteria_resolved` errors | Neo4j / JSON fallback |
| R8 Cost blowout | High tokens, low annotation yield | fast mode, markdown trim, model choice |

## 7. What to Evaluate: Outputs vs Parts

| Layer | Evaluate all 8–12 runs? | Method |
|-------|-------------------------|--------|
| Final report (md/pdf) | **Yes** | Tier 1 + OQI proxy |
| Job reliability / cost | **Yes** | Tier 1 automatic |
| Annotation grounding | **Sample** | 3 annotations × subset of runs |
| `events.jsonl` / tool trace | **Failures/outliers only** | Tier 2 |
| MinerU adapter | Only if R1 | Tier 2 |
| PASA / DeepXiv | Only if R4 | Tier 2 + provider doc |
| Neo4j KG | Only if R7 or KG papers fail criteria sections | Tier 2 |
| Full `review_tools.py` audit | **No** (unless repeated R3/R5) | Engineering backlog |

## 8. Reporting Template

### Executive summary (≤ 1 page)

- **Completion rate:** X/Y (Z%)
- **Median latency / tokens:** …
- **Median OQI:** …/10
- **Verdict:** Go / Go with fixes / No-go for production
- **Top 3 fixes:** (bucket → action)

### Tables to include

1. Per-job run sheet (Section 5)
2. Aggregate: pass/fail counts per Tier 1 metric
3. OQI distribution (histogram or min/median/max)
4. Tier 2 root-cause counts (bar chart optional)

### Research angle (lightweight claims you can defend without human raters)

- Report **process validity** proxies: annotation count, grounding sample pass rate, structured output rate.
- Avoid claiming "review quality equals human expert" — frame as **feasibility + reliability study**.
- Optional sentence: "Human evaluation deferred to follow-up study."

## 9. Go / No-Go Decision Rules

| Verdict | Conditions |
|---------|------------|
| **Go** | Completion ≥ 85%, median OQI ≥ 7, grounding sample ≥ 80% pass, no single bucket > 40% of failures |
| **Go with fixes** | Completion ≥ 70%, fixable buckets (R1/R3/R6), OQI ≥ 6 |
| **No-go** | Completion < 70%, median OQI < 6, or grounding sample < 50% |

## 10. Optional Stretch (if time remains)

- Run 2 papers in full (non-fast) mode and compare OQI vs tokens.
- Compare `PAPER_SEARCH_PROVIDER=deepxiv` vs disabled on same paper.
- Add one external baseline (e.g., raw LLM prompt without tools) on 1 paper only — qualitative diff, not full benchmark.

## 11. Implementation Artifacts (next step after spec approval)

After this spec is approved, use **writing-plans** skill to produce:

1. A small harvest script (or notebook) to fill the run sheet from `data/jobs/*/`.
2. A checklist markdown for Day 1 / Day 2 operators.
3. Optional: pytest marker for "evaluation smoke" if repeated runs are needed.

---

## Appendix A: Key paths

```text
data/jobs/<job_id>/state.json
data/jobs/<job_id>/events.jsonl
data/jobs/<job_id>/final_report.md
data/jobs/<job_id>/final_report.pdf
data/jobs/<job_id>/annotations.json
data/jobs/<job_id>/mineru_markdown.md   # if present
```

## Appendix B: CLI reference

```bash
python main.py submit --pdf /path/to/paper.pdf --wait-seconds 0
python main.py watch --job-id <job_id> --interval 2 --timeout 1800
python main.py status --job-id <job_id>
python main.py result --job-id <job_id> --format all
pytest tests/
```
