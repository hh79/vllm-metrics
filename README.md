# vLLM Metrics Collector

[![GitHub](https://img.shields.io/badge/github-hh79%2Fvllm--metrics-blue?logo=github)](https://github.com/hh79/vllm-metrics)

Track token generation across all your vLLM servers — aggregated by model, stored for years,
safe against restarts and downtime.

```
vllm-metrics daemon   →  runs forever, scrapes every 60s
vllm-metrics scrape   →  one-shot collect
vllm-metrics report   →  usage by server + model
```

## Quick Start

```bash
pip install pyyaml

# Configure your vLLM server(s)
vim config.yaml

# One-shot test
./vllm-metrics scrape

# Background daemon (see "Run as a systemd Service" below for production)
systemctl --user start vllm-metrics

# View report (last 7 days)
./vllm-metrics report
```

## Architecture

```
                  ┌──────────────────┐
                  │  config.yaml     │
                  │  - server list   │
                  │  - interval      │
                  └────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              │  vllm-metrics daemon    │
              │  (or scrape, one-shot) │
              │                         │
              │  1. Scrape /metrics     │
              │  2. Split by model_name │
              │  3. Compute safe delta  │
              │  4. Store in SQLite     │
              │  5. Log to stdout       │
              └────────────┬────────────┘
                           │
              ┌────────────┴────────────┐
              │  ~/.vllm-metrics.db     │
              │  - raw_snapshots (90d)  │
              │  - daily_stats (years)  │
              └─────────────────────────┘
```

Each vLLM server exposes `/metrics` (Prometheus format) with counters and histograms
labelled by `model_name` and `engine`. The scraper reads these, groups them per model,
computes **incremental deltas** (never stores raw cumulative values), and writes to a
local SQLite database. Raw data is kept indefinitely; run `vllm-metrics prune` to
aggregate completed days into daily summaries.

## Commands

### `daemon`

```bash
./vllm-metrics daemon
./vllm-metrics daemon --config /path/to/config.yaml
```

Runs continuously, scraping every `interval` seconds (default: 60).
- Failed servers are reported once then silenced until they recover.
- Server restarts (counter resets) are detected and handled transparently.

### `scrape`

```bash
./vllm-metrics scrape
```

One-shot: scrape all servers, store results, exit.
- On first run, each server+model pair is recorded as a **baseline** (no delta yet).
- On second run, deltas are computed from the baseline.

### `prune`

```bash
./vllm-metrics prune
```

Manually aggregate raw snapshots into `daily_stats` for completed dates (yesterday
and earlier). Also prunes raw snapshots older than `raw_retention_days` if set (default:
0 = keep forever).

Run this periodically (e.g., weekly cron) if you want daily summaries without keeping
all raw data. Otherwise raw data is kept indefinitely.

### `report`

```bash
# Last 7 days (default)
./vllm-metrics report

# Custom range
./vllm-metrics report --since 2026-01-01 --until 2026-06-01

# Last N days
./vllm-metrics report --days 30

# Per-model or per-server
./vllm-metrics report --model "Mistral-Small-4-119B-2603-NVFP4"
./vllm-metrics report --server spark-1-mistral
```

### `servers` / `models`

```bash
./vllm-metrics servers     # list all tracked vLLM instances
./vllm-metrics models      # list all models seen across all servers
```

## Run as a systemd Service

For automatic start on boot + auto-restart on crash, install as a **systemd user service**:

```bash
# 1. Create the service unit
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/vllm-metrics.service << 'EOF'
[Unit]
Description=vLLM Metrics Collector Daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=%h/ai/vllm-metrics/vllm-metrics daemon
WorkingDirectory=%h/ai/vllm-metrics
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF

# 2. Start now + enable on boot
systemctl --user daemon-reload
systemctl --user enable --now vllm-metrics
```

> **Note:** Adjust `%h/ai/vllm-metrics/vllm-metrics` to match your install path.

**Management commands:**

```bash
systemctl --user status vllm-metrics     # status + recent logs
systemctl --user stop vllm-metrics       # stop
systemctl --user start vllm-metrics      # start
systemctl --user restart vllm-metrics    # restart
journalctl --user -u vllm-metrics -f    # live log tail
```

**Uninstall:**

```bash
systemctl --user stop vllm-metrics
systemctl --user disable vllm-metrics
rm ~/.config/systemd/user/vllm-metrics.service
systemctl --user daemon-reload
```

> **Boot without login:** User services start when you log in. For headless boot, run `sudo loginctl enable-linger $(whoami)` first.

**List all user services:**

```bash
systemctl --user list-units --type=service
```

## Config

Edit `config.yaml`:

```yaml
servers:
  - name: spark-1-mistral
    url: http://192.168.1.10:8000
    notes: Mistral-Small-4 on Spark #1

  - name: cluster-nemotron
    url: http://192.168.1.10:8000
    notes: Nemotron-3-Super on 2-node cluster

interval: 60                 # scrape every 60 seconds
database: ~/.vllm-metrics.db
raw_retention_days: 0       # set > 0 to prune raw data after prune run
```

## Database Schema

File: `~/.vllm-metrics.db`

| Table | Contents |
|-------|----------|
| `servers` | Tracked vLLM instances (name, url, added_at, last_seen) |
| `models` | Models discovered per server (model_name, first/last seen) |
|| `raw_snapshots` | Per-scrape **deltas** (incremental counters + gauge snapshots). Kept indefinitely when `raw_retention_days: 0`. |
| `daily_stats` | Pre-aggregated daily rows (SUM of deltas, AVG of gauges/histograms). Kept forever for year-scale queries. |
| `last_values` | Last-seen cumulative counter values (the baseline for next delta computation) |

Rollup happens manually via `vllm-metrics prune`. It:
1. SELECTs all raw_snapshots for each completed date
2. SUMs the deltas into a daily_stats row
3. If `raw_retention_days > 0`, DELETEs raw_snapshots older than that threshold
4. When `raw_retention_days: 0`, raw data is never deleted (recommended for per-hour graphing)

## Collected Metrics

All metrics come from vLLM's Prometheus `/metrics` endpoint.

**Token counters** (stored as incremental deltas):
| Metric | Tracks |
|--------|--------|
| prompt_tokens_total | Prompt tokens processed (by model) |
| generation_tokens_total | Output tokens generated |
| prompt_tokens_cached_total | Prefix cache hits |
| request_success_total | Completed requests |
| num_preemptions_total | Preempted requests |
| prefix_cache_hits / queries | Prefix cache efficiency |

**Live gauges** (stored as-is each scrape):
| Metric | Tracks |
|--------|--------|
| num_requests_running | Currently active requests |
| kv_cache_usage_perc | GPU KV cache utilization |

**Performance histograms** (stored as _count + _sum for averaging):
| Metric | What it measures |
|--------|------------------|
| time_to_first_token_seconds | TTFT (latency to first output token) |
| inter_token_latency_seconds | ITL / TPOT (time between output tokens) |
| e2e_request_latency_seconds | End-to-end request duration |
| request_queue_time_seconds | Time spent waiting in queue |
| request_prefill_time_seconds | Time in prefill phase |
| request_decode_time_seconds | Time in decode phase |

## Sample Report

```
======================================================================
  vLLM USAGE STATISTICS REPORT
======================================================================
  Period:              2026-06-01  to  2026-06-07
  Models tracked:      3

  GLOBAL TOTALS
  ----------------------------------------------------------------
  Total tokens processed (prompt + generation)        14.2M
    Prompt tokens                                      3.1M
    Generation tokens                                 11.1M
    From prefix cache                                  2.4M
    Prefix cache hit rate                              77.4%
  Completed requests                                   8,520
  Preemptions                                             12
  Active days                                             7
  Avg prompt tokens per request                          364
  Avg generation tokens per request                     1303

  PER-MODEL BREAKDOWN
  ----------------------------------------------------------------
  [spark-1]  Mistral-Small-4-119B-2603-NVFP4
    Total tokens                                      14.2M
    Prompt tokens                                      3.1M
    Generation tokens                                 11.1M
    Requests                                          8,520
    Avg TTFT                                         342.5ms
    Avg ITL/TPOT                                       18.3ms

  DAILY TREND
  ----------------------------------------------------------------
  Date             Prompt tok     Gen tok    Requests
  ----------------------------------------------------------------
  2026-06-07          450,000    1,650,000      1,250
  2026-06-06          420,000    1,580,000      1,190
  ...
======================================================================
```

## Requirements

- Python 3.10+
- PyYAML (`pip install pyyaml`)
- Everything else is Python standard library

## Project Structure

```
~/ai/vllm-metrics/
├── vllm-metrics            CLI entry point (chmod +x)
├── config.yaml             Edit for your servers
├── README.md               This file
├── AGENT.md                Agent behavior guide
└── vllm_metrics/
    ├── __init__.py
    ├── scraper.py           Prometheus parser + delta computation
    ├── db.py                SQLite schema + rollup queries
    ├── report.py            Report formatter
    └── daemon.py            Main loop
```
