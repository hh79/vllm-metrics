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
            SUM(d.num_snapshots)                        AS total_snapshots
        FROM daily_stats d
        JOIN servers s ON d.server_id = s.id
        LEFT JOIN models m ON d.model_id = m.id
        WHERE {where}
        GROUP BY s.name, m.model_name
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
            COUNT(*)                                        AS total_snapshots
        FROM raw_snapshots r
        JOIN servers s ON r.server_id = s.id
        LEFT JOIN models m ON r.model_id = m.id
        WHERE {where}
        GROUP BY s.name, m.model_name
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

    if totals['total_requests'] and totals['total_requests'] > 0:
        avg_prompt = totals['total_prompt_tokens'] / totals['total_requests']
        avg_gen = totals['total_gen_tokens'] / totals['total_requests']
        _print_row("Avg prompt tokens per request", f"{avg_prompt:.0f}")
        _print_row("Avg generation tokens per request", f"{avg_gen:.0f}")

    print()

    # =====================================================
    # PER-MODEL BREAKDOWN
    # =====================================================
    if rows:
        print("  PER-MODEL BREAKDOWN")
        _print_separator('-')

        for i, row in enumerate(rows):
            server = row['server_name'] or '?'
            model = row['model_name'] or '(aggregate)'
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

            if row['avg_ttft_ms']:
                _print_row("    Avg TTFT", _fmt_ms(row['avg_ttft_ms'] / 1000))
            if row['avg_itl_ms']:
                _print_row("    Avg ITL/TPOT", _fmt_ms(row['avg_itl_ms'] / 1000))
            if row['avg_e2e_s']:
                _print_row("    Avg E2E latency", _fmt_s(row['avg_e2e_s']))
            if row['avg_queue_s']:
                _print_row("    Avg queue time", _fmt_s(row['avg_queue_s']))

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
