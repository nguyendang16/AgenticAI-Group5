# Evaluation Summary

## Demo checklist (corpus)

- Jobs evaluated: **2**
- Success rate: **1/2** (50%)
- Demo pass rate: **1/2** (50%)
- Avg runtime: **30.0 min**
- Avg annotations: **10.0**
- Grounding pass rate (≥2/3): **1/1** (100%)
- Demo verdict: **FAIL**

## MINT–AgentBoard (system performance)

- Framework: **MINT–AgentBoard System Performance Evaluation Framework**
- Deliverable rate (M3): **1/2** (50%)
- Median wall-clock (M5): **30.0 min** (target ≤ 45 min)
- Median tokens (M6): **200,000**
- Median tool calls (M2 k proxy): **15**
- System pass rate: **1/2** (50%)
- Corpus verdict: **NO GO**

## Paper sources

- **MINT** (Wang et al., ICLR 2024): Table 2 — SR, interaction depth k
- **AgentBoard** (Ma et al., NeurIPS 2024 D&B): Table 1 — multi-round, latency, cost, trace

## Top root-cause buckets

- R2: 1 job(s)

## Per-job results

- `a` · completed · demo=pass · grounding=3/3 · ann=10 · wall=30 min
- `b` · failed · demo=fail · grounding=0/0 · ann=2 · wall=5 min
