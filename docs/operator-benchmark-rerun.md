# Operator Runbook — KG Benchmark Rerun (Eval v2)

Use this **after** the eval v2 implementation is complete (`docs/superpowers/specs/2026-06-08-benchmark-eval-fix-design.md`).

## Prerequisites

```powershell
cd D:\code\Agentic_AI\AgenticAI-Group5\.worktrees\gemma-ragas-rerun
pip install -e ".[benchmark,benchmark-v2]"
```

**.env (reviews — OpenAI):**

- `OPENAI_API_KEY`, `AGENT_MODEL=gpt-5-mini`
- Neo4j + MinerU tokens as usual

**.env (eval v2 — split providers):**

```env
BENCHMARK_JUDGE_PROVIDER=openai
BENCHMARK_FAITHFULNESS_PROVIDER=google
BENCHMARK_JUDGE_MODEL=gpt-5-mini
BENCHMARK_FAITHFULNESS_MODEL=gemma-4-31b-it
GOOGLE_API_KEY=<from Google AI Studio>
BENCHMARK_JUDGE_MODE=per_metric
BENCHMARK_EVAL_PAUSE_SECONDS=5
BENCHMARK_JUDGE_MAX_MANUSCRIPT_CHARS=25000
BENCHMARK_JUDGE_MAX_REPORT_CHARS=8000
BENCHMARK_JUDGE_MAX_CRITERIA_CHARS=4000
BENCHMARK_RAGAS_ANNOTATION_MATCH_MIN_JACCARD=0.15
```

Reviews are unchanged (OpenAI agent). Eval v2 uses **OpenAI** for per-metric judge + pairwise comparison and **Gemma** for RAGAS faithfulness only.

## Pre-flight (v2 re-score)

1. Archive current eval outputs (does not touch `runs.jsonl` or review jobs):

   ```powershell
   python -m benchmark archive-results --tag pre-v2-2026-06-08
   ```

   Copies active files to `benchmark/results/archive/pre-v2-2026-06-08/`:

   - `review_quality_scores.csv`
   - `faithfulness_run_scores.csv`
   - `claim_scores.jsonl`
   - `paired_comparison.csv`
   - `benchmark_summary.md`
   - `pairwise_judge_scores.csv` (if present)

2. Delete or truncate active CSVs/JSONL above so skip-id logic does not block re-score
3. Set split-provider `.env` (see above)
4. Confirm 18 completed runs in `benchmark/results/runs.jsonl`
5. Kill orphan workers: no other `python -m benchmark run` or `main.py watch`

## Sanity gate (mandatory before full corpus)

Run on the ACL short pair before full v2 eval:

```powershell
python -m benchmark judge --job-id d9f100a7-874d-47b6-9c9b-b80d3f177af5
python -m benchmark judge --job-id 330c5928-a8d1-4ec1-8efe-ed2081001477
python -m benchmark pairwise-judge --paper-id acl_2024.acl-short.8
python -m benchmark faithfulness --job-id d9f100a7-874d-47b6-9c9b-b80d3f177af5
```

**Pass criteria:**

| Check | Criterion |
|-------|-----------|
| Judge | Not all four metrics = 5.0 on both ACL runs (spread or differential expected) |
| Faithfulness | Sanity job mean > 0.1 with ≥ 3 claims; `context_source` not `citation_label` in `claim_scores.jsonl` |
| Pairwise | Row has `winner` + non-empty `reason` |

Do **not** proceed to full corpus eval until all three pass.

## Run with terminal progress

### Phase 1 — Reviews only (OpenAI, gap-fill if needed)

```powershell
python -m benchmark pipeline --phase reviews
```

Shows a progress bar per step: `build-manifest` → `run`.

**Do not** use `benchmark all` for production reruns.

### Wait

Pause **≥60 minutes** after reviews finish if OpenAI rate limits were hit (TPM recovery).

### Phase 2 — Eval v2 + report

```powershell
python -m benchmark pipeline --phase eval
```

Steps: `collect` → `check` → `judge` → `pairwise-judge` → `faithfulness` → `compare` → `report`.

Or step-by-step:

```powershell
python -m benchmark judge
python -m benchmark pairwise-judge
python -m benchmark faithfulness
python -m benchmark compare
python -m benchmark report
```

### Dry-run (no API calls for reviews)

```powershell
python -m benchmark pipeline --phase reviews --dry-run
```

## Manual step-by-step (full pipeline)

```powershell
python -m benchmark build-manifest --local-only
python -m benchmark run
python -m benchmark collect
python -m benchmark check
python -m benchmark judge
python -m benchmark pairwise-judge
python -m benchmark faithfulness
python -m benchmark compare
python -m benchmark report
```

## Outputs

| File | Content |
|------|---------|
| `benchmark/results/benchmark_summary.md` | Research headline (pairwise KG win rate + per-metric deltas) |
| `benchmark/results/review_quality_scores.csv` | OpenAI per-metric judge (4 metrics) |
| `benchmark/results/pairwise_judge_scores.csv` | KG_ON vs KG_OFF preference per paper |
| `benchmark/results/faithfulness_run_scores.csv` | Gemma RAGAS per run |
| `benchmark/results/claim_scores.jsonl` | Per-claim faithfulness + `context_source` |
| `benchmark/results/paired_comparison.csv` | Paired deltas + pairwise columns |

## Token expectations (v2 eval, 18 runs)

| Step | Provider | Est. cost | Est. time |
|------|----------|-----------|-----------|
| Judge (4×/run) | OpenAI | ~$0.18 | ~10 min |
| Pairwise (9 pairs) | OpenAI | ~$0.04 | ~3 min |
| Faithfulness (~120 claims) | Gemma (free tier) | $0 | ~35 min |
| **Total v2 eval** | | **~$0.22** | **~50 min** |

## Troubleshooting

| Symptom | Action |
|---------|--------|
| OpenAI 429 (judge/pairwise) | Stop workers; wait; retry — 3× backoff built in |
| Gemma 429 / RPM | Increase `BENCHMARK_EVAL_PAUSE_SECONDS` to 8–10 |
| Faithfulness all zeros | Check `context_source` in `claim_scores.jsonl`; ensure v2 `.env` and annotation resolution |
| Judge scores all 5.0 | Confirm `BENCHMARK_JUDGE_MODE=per_metric` and truncation vars |
| `benchmark/.run.lock` stuck | Delete lock if owning PID is dead |
