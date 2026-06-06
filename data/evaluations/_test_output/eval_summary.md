# MINT–AgentBoard System Performance Summary

Framework: **MINT–AgentBoard System Performance Evaluation Framework**
- Jobs evaluated: **2**
- Success rate (M1): **1/2** (50%)
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

- `a` · completed · system=pass · wall=30 min · tokens=200000 · tools=15 · tier2=False · bucket=-
- `b` · failed · system=fail · wall=5 min · tokens=10000 · tools=2 · tier2=True · bucket=R2
