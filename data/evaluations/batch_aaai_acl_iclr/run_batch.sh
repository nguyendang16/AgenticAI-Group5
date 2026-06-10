#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

LOG_DIR="$ROOT/data/evaluations/batch_aaai_acl_iclr"
LOG_FILE="$LOG_DIR/run_log.jsonl"
mkdir -p "$LOG_DIR"

PAPERS=(
  "/Users/xuxiangjie/Downloads/aaai_2404.01847v3.pdf"
  "/Users/xuxiangjie/Downloads/aaai_ra02aayytw.pdf"
  "/Users/xuxiangjie/Downloads/aaai_t1yqsvzeoo.pdf"
  "/Users/xuxiangjie/Downloads/acl_2024.acl-short.8.pdf"
  "/Users/xuxiangjie/Downloads/acl_2024.findings-acl.438.pdf"
  "/Users/xuxiangjie/Downloads/computers_and_education_1-s2.0-s0360131524002380-main.pdf"
  "/Users/xuxiangjie/Downloads/ets_liu-usingaibasedobject-2023.pdf"
  "/Users/xuxiangjie/Downloads/iclr_1412.6980v9.pdf"
  "/Users/xuxiangjie/Downloads/iclr_9447_tabular_insights_visual_i.pdf"
  "/Users/xuxiangjie/Downloads/icml_2311.10263v2.pdf"
  "/Users/xuxiangjie/Downloads/icml_2402.01869v2.pdf"
  "/Users/xuxiangjie/Downloads/neurips_1706.03762v7.pdf"
  "/Users/xuxiangjie/Downloads/neurips_2402.05602v2.pdf"
)

total=${#PAPERS[@]}
start_idx="${START_IDX:-1}"

if [ "$start_idx" -eq 1 ] && [ ! -f "$LOG_FILE" ]; then
  : > "$LOG_FILE"
fi

idx=0
for pdf in "${PAPERS[@]}"; do
  idx=$((idx + 1))
  if [ "$idx" -lt "$start_idx" ]; then
    continue
  fi

  paper_name="$(basename "$pdf")"
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[$idx/$total] Submitting: $paper_name"

  submit_json="$(python main.py submit --pdf "$pdf" --wait-seconds 0)"
  job_id="$(printf '%s' "$submit_json" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")"
  echo "  job_id=$job_id"

  watch_exit=0
  python main.py watch --job-id "$job_id" --interval 2 --timeout 1800 >/dev/null || watch_exit=$?

  status_json="$(python main.py status --job-id "$job_id")"
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  STATUS_TMP="$(mktemp)"
  printf '%s' "$status_json" > "$STATUS_TMP"
  python3 - "$paper_name" "$job_id" "$started_at" "$finished_at" "$watch_exit" "$LOG_FILE" "$STATUS_TMP" <<'PY'
import json, sys
paper_name, job_id, started_at, finished_at, watch_exit, log_file, status_path = sys.argv[1:8]
status = json.loads(open(status_path, encoding='utf-8').read())
row = {
    'paper_name': paper_name,
    'job_id': job_id,
    'started_at': started_at,
    'finished_at': finished_at,
    'watch_exit': int(watch_exit),
    'status': status.get('status'),
    'annotation_count': status.get('annotation_count'),
    'total_tokens': (status.get('usage') or {}).get('token', {}).get('total_tokens'),
    'error': status.get('error'),
}
with open(log_file, 'a', encoding='utf-8') as f:
    f.write(json.dumps(row, ensure_ascii=False) + '\n')
print(f"  done: {row['status']} ann={row.get('annotation_count')} tokens={row.get('total_tokens')}")
PY
  rm -f "$STATUS_TMP"

  echo
done

echo "Batch complete. Log: $LOG_FILE"
python scripts/harvest_eval.py --output-dir "$LOG_DIR" --save-per-job
