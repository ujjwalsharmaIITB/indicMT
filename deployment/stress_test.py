"""Load test the live API with an asyncio+httpx load generator. Run against an already-running API (scripts/slurm/deployment_stress_test.sh
starts uvicorn, waits for /api/health, then runs this).

Reports p50/p95/p99 latency and RPS, broken out by:
  - direction class: en->xx, xx->en, xx->xx (pivot)
  - payload size: short (~10 words), medium (~40 words), long (~100+ words)
  - concurrency level: 1, 4, 8, 16, 32, 64, 128
"""

import argparse
import asyncio
import json
import statistics
import time

import httpx

LANGS = ["hi", "ta", "bn", "ur", "sa"]  # mixed scripts/families - representative, not exhaustive

PAYLOADS = {
    "short": "The weather today is quite pleasant and sunny.",
    "medium": (
        "The government announced a new policy yesterday that aims to improve "
        "access to education in rural areas over the next five years, with a "
        "particular focus on digital literacy and vocational training programs "
        "for young people."
    ),
    "long": (
        "In recent years, the field of machine translation has undergone a "
        "remarkable transformation, driven largely by advances in neural network "
        "architectures and the availability of large-scale parallel corpora. "
        "Early statistical approaches, which relied heavily on phrase-based "
        "alignment and hand-crafted features, have been almost entirely "
        "superseded by end-to-end neural models capable of learning rich "
        "contextual representations directly from data. This shift has brought "
        "substantial improvements in fluency and adequacy across a wide range "
        "of language pairs, though challenges remain for low-resource languages "
        "where parallel data is scarce and linguistic structures can differ "
        "substantially from those of high-resource languages used to pretrain "
        "these systems."
    ),
}


def direction_classes():
    for lang in LANGS:
        yield ("en->xx", {"text": PAYLOADS["short"], "source_lang": "en", "target_lang": lang})
    for lang in LANGS:
        yield ("xx->en", {"text": PAYLOADS["short"], "source_lang": lang, "target_lang": "en"})
    for i in range(len(LANGS)):
        a, b = LANGS[i], LANGS[(i + 1) % len(LANGS)]
        yield ("xx->xx", {"text": PAYLOADS["short"], "source_lang": a, "target_lang": b})


async def one_request(client: httpx.AsyncClient, base_url: str, payload: dict) -> tuple:
    t0 = time.time()
    try:
        r = await client.post(f"{base_url}/api/translate", json=payload, timeout=60.0)
        ok = r.status_code == 200
    except Exception:
        ok = False
    return time.time() - t0, ok


async def run_level(base_url: str, concurrency: int, payload_size: str, direction_class: str,
                     sample_payload: dict, total_requests: int) -> dict:
    payload = dict(sample_payload)
    payload["text"] = PAYLOADS[payload_size]

    latencies = []
    errors = 0
    async with httpx.AsyncClient() as client:
        t_start = time.time()
        sem = asyncio.Semaphore(concurrency)

        async def bound_request():
            nonlocal errors
            async with sem:
                lat, ok = await one_request(client, base_url, payload)
                latencies.append(lat)
                if not ok:
                    errors += 1

        await asyncio.gather(*[bound_request() for _ in range(total_requests)])
        wall = time.time() - t_start

    latencies.sort()
    n = len(latencies)
    return {
        "concurrency": concurrency, "payload_size": payload_size, "direction_class": direction_class,
        "n_requests": n, "errors": errors,
        "p50_ms": statistics.median(latencies) * 1000 if n else None,
        "p95_ms": latencies[int(0.95 * n) - 1] * 1000 if n else None,
        "p99_ms": latencies[int(0.99 * n) - 1] * 1000 if n else None,
        "rps": n / wall if wall > 0 else None,
    }


async def main_async(base_url: str, concurrencies: list, requests_per_level: int, output: str):
    results = []
    directions = list(direction_classes())
    for concurrency in concurrencies:
        for payload_size in PAYLOADS:
            for direction_class, sample_payload in directions:
                r = await run_level(base_url, concurrency, payload_size, direction_class,
                                     sample_payload, requests_per_level)
                results.append(r)
                print(f"[stress_test] c={concurrency:>3} {payload_size:>6} {direction_class:>6} "
                      f"-> p50={r['p50_ms']:.0f}ms p95={r['p95_ms']:.0f}ms "
                      f"rps={r['rps']:.1f} errors={r['errors']}", flush=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[stress_test] wrote {output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrencies", nargs="+", type=int, default=[1, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--requests-per-level", type=int, default=None,
                     help="default: 3x the concurrency level, min 10")
    ap.add_argument("--output", default="logs/stress_test_results.json")
    args = ap.parse_args()

    requests_per_level = args.requests_per_level or max(30, max(args.concurrencies) // 2)
    asyncio.run(main_async(args.base_url, args.concurrencies, requests_per_level, args.output))


if __name__ == "__main__":
    main()
