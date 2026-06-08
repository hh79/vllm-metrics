"""
Daemon loop for vLLM metrics collection.

Scrapes all configured vLLM servers at a regular interval,
stores snapshots per-model, and runs daily rollup maintenance.
"""

import time
import sys
import os
from datetime import datetime, timezone

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


from .scraper import scrape_server, extract_model_stats, compute_deltas
from .db import (
    connect, get_db_path,
    upsert_server, update_last_seen, get_servers, upsert_model,
    store_snapshot, get_last_values, save_last_values,
)


def load_config(path: str) -> dict:
    """Load config.yaml."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        # Check a few default locations
        alt_paths = [
            os.path.expanduser('~/.vllm-metrics/config.yaml'),
            os.path.join(os.path.dirname(__file__), '..', 'config.yaml'),
        ]
        for ap in alt_paths:
            if os.path.exists(ap):
                path = ap
                break
        else:
            raise FileNotFoundError(
                f"Config not found at {path} or any default location. "
                f"Create it from the template."
            )

    if yaml is None:
        raise ImportError("PyYAML is required. Install it: pip install pyyaml")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    return cfg


def run_once(conn, config: dict, failed_servers: set | None = None):
    """Scrape all servers once and store results.

    Args:
        failed_servers: Optional set to track which servers are persistently
                       failing, to avoid repeated log spam.
    """
    servers_cfg = config.get('servers', [])
    if not servers_cfg:
        print("  [WARN] No servers configured in config.yaml")
        return

    for sv in servers_cfg:
        name = sv['name']
        url = sv['url']
        notes = sv.get('notes', '')

        # Ensure server is registered in DB
        server_id = upsert_server(conn, name, url, notes)

        # Scrape
        result = scrape_server(name, url)
        if not result.success:
            if failed_servers is not None:
                if name not in failed_servers:
                    print(f"  [FAIL] {name} ({url}): {result.error}")
                    failed_servers.add(name)
            else:
                print(f"  [FAIL] {name} ({url}): {result.error}")
            continue

        # Mark as online
        update_last_seen(conn, server_id)

        # Server came back after failures
        if failed_servers is not None and name in failed_servers:
            print(f"  [RECOVERED] {name} ({url}): back online")
            failed_servers.discard(name)

        print(f"  [OK]   {name} ({url}): {len(result.models)} model(s)", end='')

        # Process each model
        for model_name, samples in result.models.items():
            model_id = upsert_model(conn, server_id, model_name)

            # Extract current cumulative scalar stats
            current_stats = extract_model_stats(samples, result.raw_timestamp)

            # Get last known cumulative values for delta computation
            last_vals = get_last_values(conn, server_id, model_id)

            # Compute deltas (handles restarts: counter decreased -> delta = new value)
            # On first scrape (last_vals empty), deltas are empty -> stores gauges/histos only
            deltas = compute_deltas(current_stats, last_vals) if last_vals else {}

            # Save the raw cumulative values as the new baseline
            save_last_values(conn, server_id, model_id, current_stats)

            if deltas:
                # Store the deltas (incremental counters)
                store_snapshot(conn, server_id, model_id, result.raw_timestamp, deltas)
            elif not last_vals:
                print(f" [BASELINE] {model_name}", end='')
            else:
                print(f" [NO DELTA] {model_name}", end='')

        # Also store unlabeled samples
        if result.unlabeled:
            unlabeled_stats = extract_model_stats(result.unlabeled, result.raw_timestamp)
            last_vals = get_last_values(conn, server_id, None)
            deltas = compute_deltas(unlabeled_stats, last_vals) if last_vals else {}
            save_last_values(conn, server_id, None, unlabeled_stats)
            if deltas:
                store_snapshot(conn, server_id, None, result.raw_timestamp, deltas)
            print(" [+unlabeled]", end='')

        print()


def daemon_loop(config: dict):
    """Main daemon loop."""
    db_path = get_db_path(config.get('database', '~/.vllm-metrics.db'))
    interval = config.get('interval', 60)
    raw_retention = config.get('raw_retention_days', 90)

    conn = connect(db_path)
    failed_servers: set[str] = set()
    print(f"vLLM Metrics Collector Daemon")
    print(f"  Database: {db_path}")
    print(f"  Interval: {interval}s")
    print(f"  Servers:  {len(config.get('servers', []))} configured")
    print(f"  Raw retention: {raw_retention}d")
    print(f"  Press Ctrl+C to stop")
    print()

    while True:
        try:
            # Scrape all servers
            now_str = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{now_str}] Scraping...")
            run_once(conn, config, failed_servers)

            time.sleep(interval)

        except KeyboardInterrupt:
            print("\nDaemon stopped.")
            break
        except Exception as e:
            print(f"  [ERROR] {e}", file=sys.stderr)
            time.sleep(interval)
