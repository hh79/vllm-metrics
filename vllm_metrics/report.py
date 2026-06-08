"""
Report generator for vLLM metrics.

Produces human-readable usage reports from the database,
supporting per-model, per-server, and global aggregation
over arbitrary time ranges (day, week, month, year, custom).
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
from collections import defaultdict
import time as _time
import os
import subprocess


def _detect_timezone(config_tz: str | None = None) -> datetime.tzinfo:
    """Return a tzinfo from config, auto-detect, or fallback to UTC."""
    if config_tz and config_tz.lower() not in ('auto', 'utc', ''):
        from zoneinfo import ZoneInfo
        try:
            return ZoneInfo(config_tz)
        except (KeyError, TypeError):
            pass

    if config_tz and config_tz.lower() != 'auto':
        # Explicitly set to UTC (or invalid — fall through to auto)
        return timezone.utc

    # Auto-detect from system
    for method in [
        lambda: subprocess.run(
            ['timedatectl', 'show', '--property=Timezone', '--value'],
            capture_output=True, text=True, timeout=5
        ).stdout.strip(),
        lambda: open('/etc/timezone').read().strip(),
    ]:
        try:
            tz_name = method()
            if tz_name:
                from zoneinfo import ZoneInfo
                return ZoneInfo(tz_name)
        except Exception:
            continue

    return timezone.utc


def _fmt_s(seconds: float | None) -> str:
    if seconds is None:
        return '--'
    if seconds < 1:
        return f'{seconds*1000:.0f}ms'
    return f'{seconds:.1f}s'


def _fmt_decimal(n: float) -> str:
    if n >= 100:
        return f'{n:.0f}'
    if n >= 10:
        return f'{n:.1f}'
    if n >= 1:
        return f'{n:.2f}'
    if n >= 0.01:
        return f'{n:.3f}'
    if n >= 0.001:
        return f'{n:.4f}'
    return f'~{n:.2g}'


def _fmt_number(n: float | None) -> str:
    if n is None:
        return '--'
    if abs(n) >= 1_000_000_000:
        return f'{n/1_000_000_000:.2f}B'
    if abs(n) >= 1_000_000:
        return f'{n/1_000_000:.2f}M'
    if abs(n) >= 1_000:
        return f'{n/1_000:.1f}K'
    if n == int(n):
        return f'{int(n)}'
    return f'{n:.1f}'


def _fmt_ms(s: float | None) -> str:
    if s is None or s == 0:
        return '--'
    return f'{s*1000:.1f}ms'


def _fmt_s(s: float | None) -> str:
    if s is None or s == 0:
        return '--'
    if s < 60:
        return f'{s:.1f}s'
    if s < 3600:
        return f'{s/60:.1f}m'
    return f'{s/3600:.1f}h'


def _fmt_pct(f: float | None) -> str:
    if f is None:
        return '--'
    return f'{f*100:.1f}%'


def _build_time_clause(conn, since, until, table_alias='d', date_col='date'):
    """Build WHERE clause and params for time filtering on daily_stats or raw_snapshots."""
    conditions = []
    params = []

    if since:
        if isinstance(since, str):
            since = datetime.fromisoformat(since).date()
        conditions.append(f'{table_alias}.{date_col} >= ?')
        params.append(since.isoformat() if hasattr(since, 'isoformat') else since)
    if until:
        if isinstance(until, str):
            until = datetime.fromisoformat(until).date()
        conditions.append(f'{table_alias}.{date_col} <= ?')
        params.append(until.isoformat() if hasattr(until, 'isoformat') else until)

    return conditions, params


def _run_summary_query(conn, since=None, until=None, model_name=None, server_name=None):
    """
    Run the main aggregate query over daily_stats.
    Returns list of dicts.
    """
    conditions = []
    params = []

    tc, tp = _build_time_clause(conn, since, until, 'd', 'date')
    conditions.extend(tc)
    params.extend(tp)

    if model_name:
        conditions.append('m.model_name = ?')
        params.append(model_name)
    if server_name:
        conditions.append('s.name = ?')
        params.append(server_name)

    where = ' AND '.join(conditions) if conditions else '1'

    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT
            s.name       AS server_name,
            m.model_name AS model_name,
            COUNT(DISTINCT d.date)                      AS active_days,
            SUM(d.prompt_tokens)                        AS total_prompt_tokens,
            SUM(d.generation_tokens)                    AS total_gen_tokens,
            SUM(d.prompt_tokens_cached)                 AS total_cached_tokens,
            SUM(d.completed_requests)                   AS total_requests,
            SUM(d.preemptions)                          AS total_preemptions,
            AVG(d.avg_ttft_ms)                          AS avg_ttft_ms,
            AVG(d.avg_itl_ms)                           AS avg_itl_ms,
            AVG(d.avg_e2e_s)                            AS avg_e2e_s,
            AVG(d.avg_queue_s)                          AS avg_queue_s,
            AVG(d.avg_kv_cache_pct)                     AS avg_kv_cache_pct,
            AVG(d.avg_running)                          AS avg_running,
            MAX(d.max_running)                          AS max_running,
            AVG(d.avg_waiting)                          AS avg_waiting,
            SUM(d.num_snapshots)                        AS total_snapshots
        FROM daily_stats d
        JOIN servers s ON d.server_id = s.id
        LEFT JOIN models m ON d.model_id = m.id
        WHERE {where}
        GROUP BY s.name, m.model_name
        HAVING m.model_name IS NOT NULL
        ORDER BY s.name, m.model_name
    """, params)
    return [dict(row) for row in cursor.fetchall()]


def _run_raw_summary(conn, since=None, until=None, model_name=None, server_name=None):
    """
    Query raw_snapshots directly with time-weighted averages.
    """
    conditions = []
    params = []

    if since:
        conditions.append("DATE(r.timestring) >= ?")
        params.append(since.isoformat() if hasattr(since, 'isoformat') else since)
    if until:
        conditions.append("DATE(r.timestring) <= ?")
        params.append(until.isoformat() if hasattr(until, 'isoformat') else until)
    if model_name:
        conditions.append('m.model_name = ?')
        params.append(model_name)
    if server_name:
        conditions.append('s.name = ?')
        params.append(server_name)

    where = ' AND '.join(conditions) if conditions else '1'

    cursor = conn.cursor()

    # Fetch the raw snapshot timeseries for time-weighted avg computation
    # Grouped by (server_id, model_id) for later Python-side processing
    cursor.execute(f"""
        SELECT
            r.server_id,
            r.model_id,
            r.timestamp,
            r.num_requests_running,
            r.num_requests_waiting,
            r.generation_tokens_total
        FROM raw_snapshots r
        JOIN servers s ON r.server_id = s.id
        LEFT JOIN models m ON r.model_id = m.id
        WHERE {where} AND m.model_name IS NOT NULL
        ORDER BY r.server_id, r.model_id, r.timestamp
    """, params)
    rows = cursor.fetchall()

    # Group rows by (server_id, model_id) for per-group processing
    groups = {}
    for row in rows:
        key = (row['server_id'], row['model_id'])
        groups.setdefault(key, []).append(row)

    # Compute time-weighted averages per group (active-only: exclude idle gaps)
    tw_avg = {}  # key -> {'avg_running': ..., 'avg_waiting': ...}
    for key, snapshots in groups.items():
        n = len(snapshots)
        if n == 0:
            tw_avg[key] = {'avg_running': None, 'avg_waiting': None}
            continue

        # Find active snapshots (those with running > 0)
        active_indices = [i for i in range(n)
                          if snapshots[i]['num_requests_running'] is not None
                          and snapshots[i]['num_requests_running'] > 0]
        if not active_indices:
            tw_avg[key] = {'avg_running': 0.0, 'avg_waiting': 0.0}
            continue

        # Build "active clusters" — merge consecutive snapshot durations
        # where running > 0, treating intervening idle gaps as breaks between
        # separate active periods (excluded from the average).
        total_weight = 0.0
        weighted_running = 0.0
        weighted_waiting = 0.0

        # Generation throughput: average rate between consecutive gen-producing snapshots
        gen_rates = []

        # Walk through each active snapshot and its duration (until next snapshot)
        for i in range(n):
            running = snapshots[i]['num_requests_running']
            if running is None or running <= 0:
                continue

            waiting = snapshots[i]['num_requests_waiting']
            ts = snapshots[i]['timestamp']
            # Duration: until next snapshot (any running state)
            if i + 1 < n:
                duration = snapshots[i + 1]['timestamp'] - ts
            else:
                duration = 60.0  # assume one interval for last snapshot

            if duration <= 0:
                continue
            total_weight += duration
            weighted_running += running * duration
            if waiting is not None:
                weighted_waiting += waiting * duration

        # Compute gen rate from consecutive snapshots with gen tokens
        gen_rows = [(i, snapshots[i]) for i in range(n)
                     if snapshots[i]['generation_tokens_total'] is not None
                     and snapshots[i]['generation_tokens_total'] > 0]
        for j in range(1, len(gen_rows)):
            prev = gen_rows[j - 1][1]
            cur = gen_rows[j][1]
            dt = cur['timestamp'] - prev['timestamp']
            gen_delta = cur['generation_tokens_total']
            if dt > 0 and gen_delta > 0:
                rate = gen_delta / dt
                if 0.1 <= rate <= 10000:
                    gen_rates.append(rate)

        tw_avg[key] = {
            'avg_running': weighted_running / total_weight if total_weight > 0 else 0.0,
            'avg_waiting': weighted_waiting / total_weight if total_weight > 0 else 0.0,
            'avg_gen_rate': sum(gen_rates) / len(gen_rates) if gen_rates else None,
        }

    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT
            s.name       AS server_name,
            m.model_name AS model_name,
            COUNT(DISTINCT DATE(r.timestring))          AS active_days,
            SUM(r.prompt_tokens_total)                  AS total_prompt_tokens,
            SUM(r.generation_tokens_total)              AS total_gen_tokens,
            SUM(r.prompt_tokens_cached_total)           AS total_cached_tokens,
            SUM(r.request_success_total)                AS total_requests,
            SUM(r.num_preemptions_total)                AS total_preemptions,
            AVG(r.ttft_sum / NULLIF(r.ttft_count, 0)) * 1000 AS avg_ttft_ms,
            AVG(r.itl_sum / NULLIF(r.itl_count, 0)) * 1000   AS avg_itl_ms,
            MAX(r.num_requests_running)                 AS max_running,
            MIN(r.timestamp)                            AS first_ts,
            MAX(r.timestamp)                            AS last_ts,
            COUNT(*)                                    AS total_snapshots,
            SUM(CASE WHEN r.generation_tokens_total > 0 THEN 1 ELSE 0 END) AS active_snapshots
        FROM raw_snapshots r
        JOIN servers s ON r.server_id = s.id
        LEFT JOIN models m ON r.model_id = m.id
        WHERE {where}
        GROUP BY s.name, m.model_name
        HAVING m.model_name IS NOT NULL
        ORDER BY s.name, m.model_name
    """, params)
    results = [dict(row) for row in cursor.fetchall()]

    # Merge Python-computed time-weighted averages into results
    for row in results:
        cursor2 = conn.cursor()
        cursor2.execute("SELECT id FROM servers WHERE name = ?", (row['server_name'],))
        srow = cursor2.fetchone()
        cursor2.execute("SELECT id FROM models WHERE model_name = ? AND server_id = ?",
                        (row['model_name'], srow['id']))
        mrow = cursor2.fetchone()
        if srow and mrow:
            key = (srow['id'], mrow['id'])
            tw = tw_avg.get(key, {})
            row['avg_running'] = tw.get('avg_running')
            row['avg_waiting'] = tw.get('avg_waiting')
            row['avg_gen_rate'] = tw.get('avg_gen_rate')

    return results


def _print_separator(char='='):
    print(char * 78)


def _print_row(left, right, width=55):
    print(f"  {left:<{width}} {right}")


def generate_report(conn, since=None, until=None, model_name=None, server_name=None, tz=None):
    """
    Generate a full usage report.

    Args:
        tz: Optional timezone (datetime.tzinfo or string IANA name).
            If None, auto-detects from system or falls back to UTC.
    """
    # Resolve timezone
    if isinstance(tz, str):
        tz = _detect_timezone(tz)
    elif tz is None:
        tz = _detect_timezone()
    elif not isinstance(tz, (timezone, type(None))):
        # Assume ZoneInfo or similar tzinfo
        pass

    local_now = datetime.now(tz)

    if since is None:
        # Default: last 7 days
        since = (local_now - timedelta(days=7)).date()
    if isinstance(since, str):
        since = datetime.fromisoformat(since).date()
    if isinstance(until, str):
        until = datetime.fromisoformat(until).date()

    period = f"{since}"
    if until:
        period += f"  to  {until}"
    else:
        period += f"  to  {local_now.date()}"

    # Query raw snapshots first (live data, time-weighted averages)
    rows = _run_raw_summary(conn, since=since, until=until, model_name=model_name, server_name=server_name)
    if not rows:
        # Fall back to daily_stats (rolled up data for longer ranges)
        rows = _run_summary_query(conn, since=since, until=until, model_name=model_name, server_name=server_name)

    # Calculate totals across all rows
    totals = {
        'total_prompt_tokens': sum(r['total_prompt_tokens'] or 0 for r in rows),
        'total_gen_tokens': sum(r['total_gen_tokens'] or 0 for r in rows),
        'total_cached_tokens': sum(r['total_cached_tokens'] or 0 for r in rows),
        'total_requests': sum(r['total_requests'] or 0 for r in rows),
        'total_preemptions': sum(r['total_preemptions'] or 0 for r in rows),
        'active_days': max((r['active_days'] or 0) for r in rows) if rows else 0,
    }

    # =====================================================
    # HEADER
    # =====================================================
    _print_separator()
    print("  vLLM USAGE STATISTICS REPORT")
    _print_separator()
    print(f"  Period:              {period}")
    print(f"  Models tracked:      {len(rows)}")

    filter_parts = []
    if model_name:
        filter_parts.append(f"model={model_name}")
    if server_name:
        filter_parts.append(f"server={server_name}")
    if filter_parts:
        print(f"  Filter:              {', '.join(filter_parts)}")

    print()

    # =====================================================
    # GLOBAL TOTALS
    # =====================================================
    print("  GLOBAL TOTALS")
    _print_separator('-')

    total_tokens = totals['total_prompt_tokens'] + totals['total_gen_tokens']
    _print_row("Total tokens processed (prompt + generation)", _fmt_number(total_tokens))
    _print_row("  Prompt tokens", _fmt_number(totals['total_prompt_tokens']))
    _print_row("  Generation tokens", _fmt_number(totals['total_gen_tokens']))
    _print_row("  From prefix cache", _fmt_number(totals['total_cached_tokens']))

    if totals['total_prompt_tokens'] and totals['total_prompt_tokens'] > 0:
        cache_pct = (totals['total_cached_tokens'] / totals['total_prompt_tokens']) * 100
        _print_row("  Prefix cache hit rate", f"{cache_pct:.1f}%")

    _print_row("Completed requests", _fmt_number(totals['total_requests']))
    _print_row("Preemptions", _fmt_number(totals['total_preemptions']))
    _print_row("Active days", str(totals['active_days']))

    print()

    # =====================================================
    # PER-MODEL BREAKDOWN
    # =====================================================
    if rows:
        print("  PER-MODEL BREAKDOWN")
        _print_separator('-')

        for i, row in enumerate(rows):
            server = row['server_name'] or '?'
            model = row['model_name'] or '?'
            if i > 0:
                print()

            total_t = (row['total_prompt_tokens'] or 0) + (row['total_gen_tokens'] or 0)

            print(f"  [{server}]  {model}")
            _print_row("    Total tokens", _fmt_number(total_t))
            _print_row("    Prompt tokens", _fmt_number(row['total_prompt_tokens']))
            _print_row("    Generation tokens", _fmt_number(row['total_gen_tokens']))
            _print_row("    Requests", _fmt_number(row['total_requests']))
            _print_row("    Preemptions", _fmt_number(row['total_preemptions']))
            _print_row("    Active days", str(row['active_days']))

            # Generation throughput (per-snapshot rate, gap-tolerant)
            gen_rate = row.get('avg_gen_rate')
            gen_tokens = row['total_gen_tokens'] or 0
            if gen_rate and gen_rate > 0:
                _print_row("    Gen throughput", f"{gen_rate:.1f} tok/s")
            elif gen_tokens:
                _print_row("    Gen throughput", f"{gen_tokens / 3600:.1f} tok/s (approx)")

            # Concurrent / waiting requests
            avg_running = row.get('avg_running')
            max_running = row.get('max_running')
            avg_waiting = row.get('avg_waiting')
            if avg_running is not None:
                _print_row("    Avg concurrent", _fmt_decimal(avg_running))
            if max_running is not None:
                _print_row("    Peak concurrent", f"{max_running:.0f}")
            if avg_waiting is not None and avg_waiting > 0:
                _print_row("    Avg waiting", f"{avg_waiting:.1f}")

            cached = row.get('total_cached_tokens') or 0
            if cached:
                _print_row("    From prefix cache", _fmt_number(cached))
                prompt_t = row['total_prompt_tokens'] or 1
                _print_row("    Prefix cache hit rate", f"{cached / prompt_t * 100:.1f}%")

            if row.get('avg_ttft_ms'):
                _print_row("    Avg TTFT", _fmt_ms(row['avg_ttft_ms'] / 1000))
            if row.get('avg_itl_ms'):
                _print_row("    Avg ITL/TPOT", _fmt_ms(row['avg_itl_ms'] / 1000))
            if row.get('avg_e2e_s'):
                _print_row("    Avg E2E latency", _fmt_s(row['avg_e2e_s']))
            if row.get('avg_queue_s'):
                _print_row("    Avg queue time", _fmt_s(row['avg_queue_s']))

    print()

    # =====================================================
    # SERVER STATS (from latest unlabeled snapshot)
    # =====================================================
    _print_separator()
    print("  SERVER STATS")
    _print_separator('-')

    stale_threshold = 300  # 5 min without a successful scrape = offline

    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            s.name,
            s.last_seen,
            r.server_uptime_seconds,
            r.process_resident_memory_bytes,
            r.process_virtual_memory_bytes,
            r.process_cpu_seconds_total,
            r.process_open_fds,
            r.timestring
        FROM servers s
        LEFT JOIN raw_snapshots r ON r.server_id = s.id
            AND r.model_id IS NULL
            AND r.timestamp = (
                SELECT MAX(r2.timestamp)
                FROM raw_snapshots r2
                WHERE r2.server_id = r.server_id AND r2.model_id IS NULL
            )
        ORDER BY s.name
    """)

    server_rows = cursor.fetchall()
    if server_rows:
        now_ts = _time.time()
        for row in server_rows:
            print(f"  [{row['name']}]")
            last_seen = row['last_seen']
            is_offline = last_seen is None or (now_ts - last_seen) > stale_threshold
            if is_offline:
                print("    Server unavailable")
                print()
                continue
            uptime = row['server_uptime_seconds']
            if uptime:
                days = int(uptime // 86400)
                hours = int((uptime % 86400) // 3600)
                mins = int((uptime % 3600) // 60)
                if days:
                    _print_row("    Uptime", f"{days}d {hours}h {mins}m")
                else:
                    _print_row("    Uptime", f"{hours}h {mins}m")
            rss = row['process_resident_memory_bytes']
            if rss:
                _print_row("    RSS memory", _fmt_number(rss))
            virt = row['process_virtual_memory_bytes']
            if virt:
                _print_row("    Virtual mem", _fmt_number(virt))
            cpu = row['process_cpu_seconds_total']
            if cpu:
                _print_row("    CPU time", _fmt_s(cpu))
            fds = row['process_open_fds']
            if fds:
                _print_row("    Open FDs", f"{int(fds)}")
            print()
    else:
        print("  (no servers configured yet)")
    print()

    # =====================================================
    # DAILY TREND (summary table)
    # =====================================================
    _print_separator()
    print("  DAILY TREND")
    _print_separator('-')

    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT
            d.date,
            SUM(d.generation_tokens) AS gen_tokens,
            SUM(d.prompt_tokens) AS prompt_tokens,
            SUM(d.completed_requests) AS requests
        FROM daily_stats d
        JOIN servers s ON d.server_id = s.id
        WHERE d.date >= ? AND d.date <= ?
        GROUP BY d.date
        ORDER BY d.date DESC
        LIMIT 31
    """, (
        since.isoformat(),
        (until or datetime.now(timezone.utc).date()).isoformat(),
    ))
    daily_rows = cursor.fetchall()

    if daily_rows:
        print(f"  {'Date':<14} {'Prompt tok':>12} {'Gen tok':>12} {'Requests':>10}")
        _print_separator('-')
        for row in daily_rows:
            print(f"  {row['date']:<14} {_fmt_number(row['prompt_tokens']):>12} "
                  f"{_fmt_number(row['gen_tokens']):>12} {_fmt_number(row['requests']):>10}")
    else:
        print("  (No daily rollup data yet -- run the collector for at least a day)")

    print()
    _print_separator()
