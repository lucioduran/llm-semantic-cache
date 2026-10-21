# LLM Semantic Cache

[![CI](https://github.com/lucioduran/llm-semantic-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/lucioduran/llm-semantic-cache/actions/workflows/ci.yml)

A small SQLite answer cache with explicit authorization boundaries, transactional invalidation, and a real local neural evaluation.

## Problem

Similar questions can require different answers when the user, permissions, date, entity, or source version changes. A high embedding similarity is insufficient authorization to reuse an answer. Embedding overhead can also erase the savings from additional hits.

## Scope

This repository implements exact and cosine semantic lookup, expiration, updates, user and version deletion, persistence, and concurrent access. A 48-request original fictional dataset challenges dates, permissions, users, entities, versions, paraphrases, changed facts, and different intents.

The cache library uses only Python's standard library. The evaluation uses pinned MiniLM embeddings and FLAN-T5-small generation on a local CPU. It makes no hosted inference calls. The unsafe diagnostic exists only in the experiment runner.

## Quick start

Python 3.11 or newer:

```sh
git clone https://github.com/lucioduran/llm-semantic-cache.git
cd llm-semantic-cache
python -m unittest discover -v
python -m scripts.verify results/reference/report.json
```

These commands use the saved neural evidence and do not install or download models.

## Usage

```python
from semantic_cache import Cache, Scope

cache = Cache("answers.db", model="your-embedding-model@exact-revision", dimensions=3)
scope = Scope("alice", ("read",), "catalog-v7", "Atlas", "2026-01-01")
ticket = cache.ticket()  # Capture before reading source data or generating an answer.
cache.put(scope, "What is the quota?", "40 GB", [1.0, 0.0, 0.0], ttl=60, ticket=ticket)
hit = cache.lookup(scope, "What is the quota?")
assert hit is not None and hit.answer == "40 GB"
cache.invalidate(user="alice")
cache.close()
```

The example vector is a test double, not a neural embedding. For semantic lookup, pass a real embedding as the third `lookup` argument. Try exact lookup first to avoid unnecessary embedding work. Pass `None` as the stored vector for exact-only entries. `put` returns `False` if invalidation occurred after the ticket was captured; discard that stale answer and retry against fresh authorized source data.

Callers must derive scope from trusted authentication, authorization, and source metadata on every request. Never let an untrusted user supply their own permissions. Use a version covering the full prompt, answer model, source content, and policy. Entity and date must come from authoritative request routing or a validated parser. This library does not authenticate users or infer those fields from prose.

`invalidate(user=...)` removes one user's entries. `invalidate(version=...)` removes that version across users. Supplying both selects their intersection. `prune()` removes expired entries. Updating an existing query replaces its answer atomically. Capture the generation ticket before source access, and invalidate on relevant source updates or revocations.

## Architecture

```text
Trusted request scope + query
            |
Exact scope and embedding-model namespace filter
            |
Exact text match -> optional embedding + cosine search
            |
Unexpired hit OR fresh generation + fenced transactional write
```

SQLite transactions serialize writes across cache instances. A per-instance lock protects its connection across threads. All scope boundaries are matched before candidates reach the similarity calculation. Invalidation deletes matching rows and advances a global generation counter in one transaction. The counter conservatively rejects even unrelated in-flight writes.

See [design and concurrency semantics](docs/design.md) and [evaluation methodology](docs/methodology.md).

## Reproduction

The reference run used Python 3.11.13 and four CPU threads on Intel macOS. The model dependency lock describes that environment; other platforms may require a different compatible PyTorch wheel. CI needs no model packages.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-models.lock
python -m scripts.evaluate download
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -m scripts.evaluate run --output results/my-run
python -m scripts.verify results/my-run/report.json
```

Use a new output directory for each run. Downloads are explicit and free; inference loads local safetensors with remote code disabled. The recorded reference run used the series' shared exclusive benchmark lock, with no other series benchmark running simultaneously. For the shared author environment, wrap the run with `python3 /Users/luchoduran/Sites/open-source-projects/.coordination/benchmark_lock.py COMMAND ARGUMENTS`.

[Configuration](data/config.json) freezes model revisions, token budgets, seed, greedy decoding, four threads, and a 0.90 similarity threshold before measurement. [Raw report](results/reference/report.json) records model file SHA-256 hashes, package versions, hardware, prompts, vectors, answers, decisions, timings, and the implementation commit. [Trace](results/reference/trace.jsonl) is flushed after each request. The verifier reconstructs decisions and metrics without neural inference; it cannot authenticate measurements.

## Results

One local sequential run, 48 requests per policy. Correct and incorrect hit rates use **all requests** as the denominator. Net savings include request-time embeddings, lookup, generation, writes, and invalidation. Model loading and warmup are excluded.

| Policy | Correct hits | Incorrect hits | Total time | Net time saved |
| --- | ---: | ---: | ---: | ---: |
| No cache | 0/48 (0%) | 0/48 (0%) | 3.893 s | 0% |
| Scoped exact | 12/48 (25.0%) | 0/48 (0%) | 2.915 s | 25.1% |
| Scoped semantic | 16/48 (33.3%) | 0/48 (0%) | 3.310 s | 15.0% |
| Unsafe scope-blind diagnostic | 8/48 (16.7%) | 32/48 (66.7%) | 0.717 s | Invalid benefit |

Semantic caching spent 0.308 s embedding queries and 0.012 s on lookup. Exact caching spent 0.002 s on lookup and no time embedding. Semantic caching reused four additional answers, but its measured net saving was smaller. Generation-time variation also contributes to the difference; this single fixed-order run does not establish a stable performance ranking.

A correct hit means the source request has identical scope, intended fact, and reference value. Answer accuracy is separately scored using the fixed numeric answer with an optional ` GB` suffix. No cache, scoped exact, and scoped semantic achieved 100% answer exact match on these simple extraction questions. The unsafe diagnostic achieved 33.3%; 80% of its hits were incorrect reuse.

See [machine-readable summary](results/reference/summary.json) and [limitations and interpretation](docs/methodology.md). Zero observed incorrect hits is a fixture result, not a general safety guarantee.

## Testing

```sh
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m mypy semantic_cache --check-untyped-defs
python -m coverage run -m unittest discover -v
python -m coverage report
python -m scripts.verify results/reference/report.json
```

CI runs on Linux, macOS, and Windows using Python 3.11.9. Tests exercise each scope boundary, permission canonicalization, model namespaces, malformed vectors and inputs, deterministic ties, expiration, replacement, pruning, persistent concurrent connections, invalidation fencing, transaction rollback, and evidence tampering. Cache tests use explicit two-dimensional test doubles. Coverage measures the library, with a 90% combined statement/branch floor. The actual neural adapter is exercised by the saved local experiment, not cross-platform CI.

## Limitations

- Scope metadata is a trusted caller contract. Missing a dependency in `version` or mislabeling an entity can cause stale or unauthorized reuse.
- Different intents within the same scope can still exceed the cosine threshold. This small author-written fixture is not a security audit or a broad semantic benchmark.
- Reads linearize at the SQLite query snapshot. An invalidation cannot retract an answer already read by an overlapping request. There is no distributed revocation barrier.
- No single-flight generation, size cap, encryption, distributed cache, vector index, or automatic expiry worker is provided. Call `prune` and manage database storage externally. Deletion is logical, not secure disk erasure.
- Similarity lookup is a linear scan within the authorized scope. This run is not a load test and does not establish scaling performance.
- One fixed-order run on one CPU has no timing confidence interval. Model loading, warmup, downloads, electricity, hardware cost, and network latency are excluded from net savings.
- The small extraction model and fictional numeric facts are intentionally tractable. Results do not establish quality for larger models, real access policies, or production traffic.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). Keep changes focused, preserve raw evidence, add regression cases, and version experiments before changing thresholds or fixtures.

## License

Code, original fixtures, documentation, and evidence are [MIT licensed](LICENSE). Model weights are not redistributed. Both pinned models use Apache-2.0; dependencies retain their upstream licenses. See [model notices](docs/model-notices.md).
