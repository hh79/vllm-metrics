# AGENTS.md — vLLM Metrics Collector

This file describes conventions, architecture, and constraints for AI agents
working on this project.

## Project Identity

A lightweight, zero-dependency (except PyYAML) daemon that scrapes Prometheus
`/metrics` endpoints from multiple vLLM servers, tracks token generation
statistics per model, and stores them in SQLite for long-term reporting.

## Key Design Decisions

### Delta-at-Ingest (Never store raw cumulative counters)

vLLM's `/metrics` returns **cumulative counters** that only go up. Storing them
raw and computing `MAX-MIN` at query time breaks when vLLM restarts (counters
reset to 0).

**Rule:** On every scrape, compute `current - last_baseline` and store only the
delta. The baseline is the last-seen cumulative value, stored in the
`last_values` table.

**Reset detection:** If `current < baseline` for a counter, the server restarted.
In that case, delta = current (the counter started fresh from 0).

Located in: `vllm_metrics/scraper.py` → `compute_deltas()`

### Try-Import pattern for PyYAML

PyYAML is the only external dependency. Import it with a try/except so the error
message is user-friendly. The rest of the codebase must use only Python stdlib
(urllib, sqlite3, json, time, datetime).

### Dual tables for long-term storage

Raw per-scrape deltas are stored in `raw_snapshots` with 90-day retention.
A daily rollup aggregates them into `daily_stats` (SUM of deltas, AVG of gauges)
for year-scale queries. Rollup happens once per day after midnight.

Located in: `vllm_metrics/db.py` → `rollup_and_prune()`, `compute_daily_rollup()`

## Module Responsibilities

| Module | Owns |
|--------|------|
| `vllm-metrics` (CLI) | argparse, subcommand dispatch, config path resolution |
| `vllm_metrics/scraper.py` | HTTP scraping, Prometheus text parsing, label splitting, delta math |
| `vllm_metrics/db.py` | SQLite schema, CRUD for servers/models/snapshots, rollup queries |
| `vllm_metrics/report.py` | Query formatting, terminal output, number formatting |
| `vllm_metrics/daemon.py` | Config loading, main loop, failure tracking |

## CLI Conventions

- Entry point: `vllm-metrics` (executable Python script, no `.py` extension)
- Subcommands: `daemon`, `scrape`, `report`, `servers`, `models`
- Config via `--config` flag or auto-detect from `./config.yaml` / `~/.vllm-metrics/config.yaml`
- All output goes to stdout. Errors to stderr.
- Daemon suppresses repeated `[FAIL]` messages for persistently down servers
  (tracks via a `set[str]` passed to `run_once()`).

## Database Naming

- Column names in `raw_snapshots` use `_total` suffix even though they store
  deltas, because the column names were chosen to mirror Prometheus metric names
  and changing them would break all existing databases. Do not rename.
- `model_id = -1` means "unlabeled" (metrics without a model_name label).
  Avoid NULL in UNIQUE constraints — use sentinel values instead.

## Prometheus Parsing

- Use `parse_prometheus_text()` → `group_by_model()` → `extract_model_stats()` pipeline.
- `extract_model_stats()` sums over all `engine=N` labels for each metric name.
- Histogram metrics get their `_count` and `_sum` variants extracted separately.
- The `compute_deltas()` function accepts `(current, previous)` dicts and returns
  only the safe increment for counters, pass-through for gauges.

## Failure Modes Handled

1. **vLLM server down:** `[FAIL]` printed once, baseline preserved, retries next loop.
2. **vLLM restart (counter reset):** Detected by `compute_deltas()`, delta = new value.
3. **First scrape (no baseline):** Stores cumulative as baseline, no delta written.
4. **Daemon crash + restart:** Baselines persist in `last_values` table.
5. **Database file deleted:** Schema auto-creates on next connect. Old data lost.
