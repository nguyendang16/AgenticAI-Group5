<div align="center">

# Evidence-Verified Agentic Peer Review System

**Automated paper review with tool-grounded reasoning and evidence verification.**  
`PDF -> MinerU Markdown -> Review Agent Tool Loop -> Final Markdown -> Final PDF`

[Quick Start](#quick-start) • [Web UI](#web-ui) • [Configuration](#configuration) • [CLI Usage](#cli-usage)

</div>

---

## Features

| Feature | Description |
| :--- | :--- |
| End-to-End Review | Runs full asynchronous review from uploaded PDF to final Markdown and PDF report. |
| Tool-Grounded Reasoning | Agent uses review tools (`pdf_search`, `pdf_read_lines`, `pdf_annotate`, etc.) to produce traceable output. |
| Evidence Verification | Agent verifies text positions before annotation for accurate highlighting. |
| Usage Accounting | Tracks token usage, per-tool call counts, and paper-search statistics for each job. |
| Publication-Style Export | Produces `final_report.pdf` with branding, usage summary, source-paper appendix, and annotation overlays. |

---

## How It Works

Each review job is persisted under:

```text
data/jobs/<job_id>/
```

Pipeline:

1. Submit source PDF.
2. Parse with MinerU v4 into markdown and layout metadata.
3. Build review runtime context and run the review agent.
4. Agent iterates with tools (`pdf_search`, `pdf_read_lines`, `pdf_annotate`, ...).
5. Persist final markdown with `review_final_markdown_write`.
6. Export final PDF report with source appendix and overlay callouts.

---

## Quick Start

### 1) Install

```bash
cd <repo_root>
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### 2) Configure

```bash
cp .env.example .env
```

Minimal practical setup:

```bash
# LLM (OpenAI-compatible)
BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_openai_api_key
AGENT_MODEL=gpt-5-mini
AGENT_REASONING_EFFORT=low
OPENAI_USE_RESPONSES_API=false
OPENAI_AGENTS_DISABLE_TRACING=1

# MinerU
MINERU_API_TOKEN=your_mineru_token

# Fast review mode (recommended for quick reviews)
REVIEW_FAST_MODE=true
```

### 3) Submit and track a job

```bash
python main.py submit --pdf /path/to/paper.pdf --wait-seconds 0
python main.py status --job-id <job_id>
python main.py watch --job-id <job_id> --interval 2 --timeout 1800
```

### 4) Fetch result

```bash
python main.py result --job-id <job_id> --format all
python main.py result --job-id <job_id> --format md
python main.py result --job-id <job_id> --format pdf
```

---

## Web UI

Start the local server (same pipeline as the CLI — jobs run via `main.py _run-job`):

```bash
chmod +x scripts/serve_ui.sh
./scripts/serve_ui.sh
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080), upload a PDF, and watch **tool progress** (`pdf_search`, `pdf_annotate`, etc.) in the timeline. When the job completes, view the markdown report and PDF in the browser.

Environment variables are read from `.env` in the project root (same as CLI).

---

## Review Criteria Knowledge Graph (Neo4j)

The system uses a **Knowledge Graph** to ground reviews in venue-specific criteria extracted from official reviewer guidelines.

### What's in the KG?

| Node Type | Description |
| :--- | :--- |
| `Venue` / `Journal` | Conference or journal (e.g., ICLR, NeurIPS, TWELF) |
| `ReviewCriterion` | Specific review criterion with semantic ID (e.g., `TWELF_C01_CLARITY_PRESENTATION`) |
| `EvidenceRequirement` | Required evidence for each criterion |
| `SourceDocument` | Original guideline document provenance |

**Currently loaded venues:** ICLR, NeurIPS, ICML, ACL, CHI, AAAI, TWELF, Computers & Education, ETRD, ETS.

### How It Works

1. **Extraction**: LLM extracts criteria from Word templates → structured JSON
2. **Loading**: JSON loaded into Neo4j with semantic criterion IDs
3. **Runtime**: Review agent queries KG by venue → criteria injected into prompt
4. **Report**: Criterion Legend + Claim-Level Audit table trace critiques to criteria

### Setup

```bash
# 1) Extract criteria from templates
python -m src.main_extract --template_dir templates --output_dir outputs/extracted_json

# 2) Load into Neo4j (configure .env first)
python -m src.main_load_graph --json_dir outputs/extracted_json

# 3) Test retrieval
python -m src.retriever --venue "TWELF"
```

### Configuration

| Variable | Description | Default |
| :--- | :--- | :--- |
| `NEO4J_URI` | Neo4j connection URI (e.g., `neo4j+s://xxx.databases.neo4j.io`) | - |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | Required |
| `NEO4J_DATABASE` | Database name | `neo4j` |
| `REVIEW_CRITERIA_ENABLED` | Enable KG-grounded reviews | `true` |
| `REVIEW_CRITERIA_JSON_DIR` | Fallback JSON directory | `outputs/extracted_json` |
| `REVIEW_VENUE` | Override venue detection | Auto-detect |

### Fallback Behavior

If the manuscript's venue is **not in KG**:
- System uses general review pipeline (no venue-specific criteria)
- Agent still produces full review with all sections
- No Criterion Legend or Claim-Level Audit in report

### Running on Another Machine

If using **Neo4j Aura (cloud)**, just configure `.env` with the same credentials — no rebuild needed.

For local Neo4j:
```bash
# Export from source machine
neo4j-admin database dump neo4j --to-path=/backup/

# Import on target machine
neo4j-admin database load neo4j --from-path=/backup/

# Or simply reload from JSON
python -m src.main_load_graph --json_dir outputs/extracted_json --clear
```

---

## Configuration

### LLM Settings

| Variable | Description | Default |
| :--- | :--- | :--- |
| `BASE_URL` | OpenAI-compatible base URL | - |
| `OPENAI_API_KEY` | API key | Required |
| `AGENT_MODEL` | Review model name | `gpt-5-mini` |
| `AGENT_REASONING_EFFORT` | Reasoning effort: `low`, `medium`, `high`, `xhigh` | `low` |
| `OPENAI_USE_RESPONSES_API` | Use Responses API when provider supports it | `false` |
| `AGENT_RESUME_ATTEMPTS` | Resume attempts (hard cap) | `2` |

### Fast Review Mode

Set `REVIEW_FAST_MODE=true` for a lightweight profile aimed at short wall-clock time:

| Variable | Default | Effect |
| :--- | :--- | :--- |
| `REVIEW_FAST_MODE` | `false` | Enables compact prompt + tight turn budget |
| `REVIEW_FAST_MAX_TURNS` | `40` | Caps agent tool/LLM turns |
| `REVIEW_FAST_MAX_MARKDOWN_CHARS` | `100000` | Paper context size in prompt |
| `REVIEW_FAST_MIN_ANNOTATIONS` | `8` | Minimum PDF annotations required |

Fast mode also disables `paper_search`, lowers annotation gates, and enables MinerU local fallback when cloud download fails.

### PDF Branding

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PDF_BRAND_NAME` | `DILab` | Brand name in PDF header/footer |
| `PDF_PRODUCER_NAME` | `DILab` | Producer name in PDF metadata |
| `PDF_LOGO_PATH` | `assets/branding/logo.png` | Path to logo file |

### MinerU Settings

| Variable | Description |
| :--- | :--- |
| `MINERU_BASE_URL` | MinerU API base URL |
| `MINERU_API_TOKEN` | MinerU API token |
| `MINERU_MODEL_VERSION` | Model version (`vlm` recommended) |
| `MINERU_ALLOW_LOCAL_FALLBACK` | Allow local PDF parsing if cloud fails |

### Paper Search (Optional)

| Variable | Description |
| :--- | :--- |
| `PAPER_SEARCH_ENABLED` | Enable external paper search |
| `PAPER_SEARCH_PROVIDER` | `deepxiv` or `pasa` |
| `DEEPXIV_API_TOKEN` | DeepXiv API token |

---

## CLI Usage

| Command | Purpose |
| :--- | :--- |
| `python main.py submit --pdf /path/to/paper.pdf` | Submit a new review job |
| `python main.py status --job-id <job_id>` | Get one-shot status snapshot |
| `python main.py watch --job-id <job_id> --interval 2 --timeout 1800` | Poll progress until timeout/completion |
| `python main.py result --job-id <job_id> --format all` | Fetch markdown + pdf outputs |

### Output Artifacts

- `data/jobs/<job_id>/final_report.md`
- `data/jobs/<job_id>/final_report.pdf`
- `data/jobs/<job_id>/events.jsonl`

`final_report.pdf` includes:

- Final markdown content
- Token usage summary (input/output/total/requests)
- Original paper appendix pages
- Auto-rendered review overlays (when MinerU line bboxes are available)

---

## External Services

### MinerU (Required)

1. Register at [https://mineru.net/](https://mineru.net/)
2. Create API token in dashboard
3. Set `MINERU_API_TOKEN` in `.env`

### DeepXiv (Optional - for paper search)

1. Register for a DeepXiv API token.
2. Set `PAPER_SEARCH_PROVIDER=deepxiv`.
3. Set `DEEPXIV_API_TOKEN` in `.env`.

---

## Troubleshooting

- **`RuntimeError: Agent finished without successful review_final_markdown_write`**
  - Model ended before final-write gate completion.
  - Check `events.jsonl` for phase progression and tool usage.

- **MinerU timeout/failure**
  - Verify token validity and endpoint reachability.
  - Try `MINERU_ALLOW_LOCAL_FALLBACK=true` for local PDF parsing.

- **Annotations misaligned**
  - Ensure agent uses `pdf_search` before `pdf_annotate` for accurate positioning.

---

