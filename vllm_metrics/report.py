"""
Report generator for vLLM metrics.

Produces human-readable usage reports from the database,
supporting per-model, per-server, and global aggregation
over arbitrary time ranges (day, week, month, year, custom).
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
from collections import defaultdict


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
    Fallback: query raw_snapshots directly (for data that hasn't been rolled up yet).
    """
    conditions = []
    params = []

    tc, tp = _build_time_clause(conn, since, until, 'r', 'timestring')
    # For raw, the date column is 'timestring' (ISO text)
    conditions = [f"DATE(r.timestring) >= ?" if 'timestring' in str(c) else c for c in tc]
    # Actually let me just rebuild properly
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
            AVG(r.num_requests_running)                 AS avg_running,
            MAX(r.num_requests_running)                 AS max_running,
            AVG(r.num_requests_waiting)                 AS avg_waiting,
            MIN(r.timestamp)                            AS first_ts,
            MAX(r.timestamp)                            AS last_ts,
            COUNT(*)                                    AS total_snapshots,
            SUM(CASE WHEN r.generation_tokens_total > 0 THEN 1 ELSE 0 END) AS active_snapshots,
            (
                SELECT AVG(sub.rate) FROM (
                    SELECT r2.generation_tokens_total /
                        MAX(NULLIF(r2.timestamp - LAG(r2.timestamp)
                            OVER (PARTITION BY r2.server_id, r2.model_id ORDER BY r2.timestamp), 0), 1)
                        AS rate
                    FROM raw_snapshots r2
                    WHERE r2.server_id = r.server_id AND r2.model_id = r.model_id
                      AND r2.generation_tokens_total > 0
                ) sub
                WHERE sub.rate BETWEEN 0.1 AND 500
            ) AS avg_gen_rate
        FROM raw_snapshots r
        JOIN servers s ON r.server_id = s.id
        LEFT JOIN models m ON r.model_id = m.id
        WHERE {where}
        GROUP BY s.name, m.model_name
        HAVING m.model_name IS NOT NULL
        ORDER BY s.name, m.model_name
    """, params)
    return [dict(row) for row in cursor.fetchall()]


def _print_separator(char='='):
    print(char * 78)


def _print_row(left, right, width=55):
    print(f"  {left:<{width}} {right}")


def generate_report(conn, since=None, until=None, model_name=None, server_name=None):
    """
    Generate a full usage report.
    """
    if since is None:
        # Default: last 7 days
        since = (datetime.now(timezone.utc) - timedelta(days=7)).date()
    if isinstance(since, str):
        since = datetime.fromisoformat(since).date()
    if isinstance(until, str):
        until = datetime.fromisoformat(until).date()

    period = f"{since}"
    if until:
        period += f"  to  {until}"
    else:
        period += f"  to  {datetime.now(timezone.utc).date()}"

    # Query daily stats first
    rows = _run_summary_query(conn, since=since, until=until, model_name=model_name, server_name=server_name)
    if not rows:
        # Try raw snapshots
        rows = _run_raw_summary(conn, since=since, until=until, model_name=model_name, server_name=server_name)

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
                _print_row("    Avg concurrent", f"{avg_running:.1f}")
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

    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            s.name,
            r.server_uptime_seconds,
            r.process_resident_memory_bytes,
            r.process_virtual_memory_bytes,
            r.process_cpu_seconds_total,
            r.process_open_fds,
            r.timestring
        FROM raw_snapshots r
        JOIN servers s ON r.server_id = s.id
        WHERE r.model_id IS NULL
          AND r.timestamp = (
              SELECT MAX(r2.timestamp)
              FROM raw_snapshots r2
              WHERE r2.server_id = r.server_id AND r2.model_id IS NULL
          )
        ORDER BY s.name
    """)

    server_rows = cursor.fetchall()
    if server_rows:
        for row in server_rows:
            print(f"  [{row['name']}]")
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
        print("  (no server stats collected yet)")
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
