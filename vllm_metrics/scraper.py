"""
Prometheus /metrics scraper and parser for vLLM.

Parses Prometheus text-format output from vLLM's /metrics endpoint
and extracts counter, gauge, and histogram values keyed by
(model_name, engine) label pairs.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError


@dataclass
class MetricSample:
    """A single metric sample with its labels and value."""
    name: str
    labels: dict
    value: float


@dataclass
class ScrapeResult:
    """Result from scraping one vLLM server."""
    server_name: str
    server_url: str
    success: bool
    error: Optional[str] = None
    # model_name -> list of samples
    models: dict[str, list[MetricSample]] = field(default_factory=dict)
    # samples with no model_name label
    unlabeled: list[MetricSample] = field(default_factory=list)
    raw_timestamp: float = 0.0


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def _parse_metric_line(line: str):
    """Parse a single Prometheus metric line.

    Returns (name, labels_dict, value) or None.
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    try:
        if '{' in line:
            name_part, rest = line.split('{', 1)
            labels_part, value_part = rest.rsplit('}', 1)
            name = name_part.strip()
            labels = dict(LABEL_RE.findall(labels_part))
            value = float(value_part.strip())
        else:
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                return None
            name, value_str = parts
            labels = {}
            value = float(value_str)
        return name, labels, value
    except (ValueError, IndexError):
        return None


def parse_prometheus_text(text: str) -> list[MetricSample]:
    """Parse full Prometheus text format into a list of MetricSamples."""
    samples = []
    for line in text.splitlines():
        parsed = _parse_metric_line(line)
        if parsed:
            name, labels, value = parsed
            samples.append(MetricSample(name=name, labels=labels, value=value))
    return samples


# ---------------------------------------------------------------------------
# Grouping by model
# ---------------------------------------------------------------------------

def group_by_model(samples: list[MetricSample]) -> tuple[dict[str, list[MetricSample]], list[MetricSample]]:
    """Split samples into per-model groups and unlabeled remainder.

    vLLM metrics carry labels: model_name, engine.
    Returns (model_dict, unlabeled_list).
    """
    models: dict[str, list[MetricSample]] = defaultdict(list)
    unlabeled: list[MetricSample] = []

    for s in samples:
        model = s.labels.get('model_name', '')
        if model:
            models[model].append(s)
        else:
            unlabeled.append(s)

    return dict(models), unlabeled


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def scrape_server(server_name: str, server_url: str, timeout: int = 15) -> ScrapeResult:
    """Scrape /metrics from a vLLM server and return grouped result."""
    url = server_url.rstrip('/') + '/metrics'
    import time
    ts = time.time()
    try:
        req = Request(url, headers={'Accept': 'text/plain'})
        with urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode('utf-8')
    except URLError as e:
        return ScrapeResult(
            server_name=server_name, server_url=server_url,
            success=False, error=str(e),
            raw_timestamp=ts,
        )
    except Exception as e:
        return ScrapeResult(
            server_name=server_name, server_url=server_url,
            success=False, error=str(e),
            raw_timestamp=ts,
        )

    samples = parse_prometheus_text(text)
    models, unlabeled = group_by_model(samples)

    return ScrapeResult(
        server_name=server_name,
        server_url=server_url,
        success=True,
        models=models,
        unlabeled=unlabeled,
        raw_timestamp=ts,
    )


# ---------------------------------------------------------------------------
# Key metric extraction (flatten into scalar dicts)
# ---------------------------------------------------------------------------

# Metrics we care about, grouped by type
COUNTER_METRICS = [
    'vllm:prompt_tokens_total',
    'vllm:generation_tokens_total',
    'vllm:prompt_tokens_cached_total',
    'vllm:request_success_total',
    'vllm:request_prompt_tokens_total',
    'vllm:request_generation_tokens_total',
    'vllm:num_preemptions_total',
    'vllm:prefix_cache_queries_total',
    'vllm:prefix_cache_hits_total',
    'vllm:mm_cache_queries_total',
    'vllm:mm_cache_hits_total',
    'vllm:prompt_tokens_total_by_source_local_compute',  # synthesized below
]

GAUGE_METRICS = [
    'vllm:num_requests_running',
    'vllm:num_requests_waiting',
    'vllm:kv_cache_usage_perc',
]

HISTOGRAM_METRICS = [
    'vllm:time_to_first_token_seconds',
    'vllm:inter_token_latency_seconds',
    'vllm:e2e_request_latency_seconds',
    'vllm:request_queue_time_seconds',
    'vllm:request_prefill_time_seconds',
    'vllm:request_decode_time_seconds',
    'vllm:request_inference_time_seconds',
    'vllm:request_time_per_output_token_seconds',
]


def extract_model_stats(samples: list[MetricSample]) -> dict[str, float]:
    """Convert a list of MetricSamples for one model into a flat value dict.

    Returns dict like {'vllm:prompt_tokens_total': 12345, ...}
    """
    result = {}

    # Build name->samples lookup
    by_name: dict[str, list[MetricSample]] = defaultdict(list)
    for s in samples:
        by_name[s.name].append(s)

    # Counters: sum over all label combos (e.g. engine=0, engine=1)
    for name in COUNTER_METRICS:
        samples = by_name.get(name, [])
        if samples:
            result[name] = sum(s.value for s in samples)

    # Gauges: sum over label combos
    for name in GAUGE_METRICS:
        samples = by_name.get(name, [])
        if samples:
            result[name] = sum(s.value for s in samples)

    # Histograms: extract _count and _sum
    for name in HISTOGRAM_METRICS:
        count_name = name + '_count'
        sum_name = name + '_sum'
        count_samples = by_name.get(count_name, [])
        sum_samples = by_name.get(sum_name, [])
        if count_samples:
            result[count_name] = sum(s.value for s in count_samples)
        if sum_samples:
            result[sum_name] = sum(s.value for s in sum_samples)

    # Synthesize prompt_tokens_by_source
    for source in ('local_compute', 'local_cache_hit', 'external_kv_transfer'):
        source_metric = f'vllm:prompt_tokens_total_by_source'
        source_samples = by_name.get(source_metric, [])
        matching = [s for s in source_samples if s.labels.get('source') == source]
        if matching:
            result[f'vllm:prompt_tokens_by_source_{source}'] = sum(s.value for s in matching)

    return result


def compute_deltas(current: dict[str, float], previous: dict[str, float] | None) -> dict[str, float]:
    """Compute delta between two snapshots for counter-type metrics.

    Handles vLLM server restarts: if a counter decreased (reset to 0),
    the delta is just the new value (the counter started fresh).

    Non-counter metrics (gauges) are passed through as-is from current.
    """
    if previous is None:
        return {}

    deltas = {}

    for key in current:
        cur = current[key]
        prev = previous.get(key)

        is_counter = (
            key.endswith('_total')
            or key in COUNTER_METRICS
            or (key.endswith('_count') and any(key.startswith(h) for h in HISTOGRAM_METRICS))
            or (key.endswith('_sum') and any(key.startswith(h) for h in HISTOGRAM_METRICS))
        )

        if is_counter:
            if prev is not None:
                delta = cur - prev
                if delta >= 0:
                    # Normal case: counter increased
                    deltas[key] = delta
                else:
                    # Counter reset! Server restarted. Delta is the new value
                    # since the counter started from 0 after restart.
                    deltas[key] = cur
            # else: no previous value, can't compute delta, skip
        elif key in current:
            # Gauge / histogram _count / _sum - store as-is
            deltas[key] = cur

    return deltas
