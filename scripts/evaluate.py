"""Real offline CPU inference; each policy runs its own cache and miss generation."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from semantic_cache import Cache, Scope

from .local_models import LocalModels, download_models, model_path


def timed(call):
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scope_of(row, policy):
    if policy == "unsafe_diagnostic":
        return Scope("shared", (), "shared", "shared", "shared")
    return Scope(**(row["scope"] | {"permissions": tuple(row["scope"]["permissions"])}))


def summarize(rows):
    result = {}
    baseline = sum(r["seconds"] for r in rows if r["policy"] == "none")
    for policy in ("none", "exact", "semantic", "unsafe_diagnostic"):
        selected = [r for r in rows if r["policy"] == policy]
        hits = [r for r in selected if r["hit"] is not None]
        correct = sum(r["correct_hit"] for r in hits)
        total = sum(r["seconds"] for r in selected)
        result[policy] = dict(
            requests=len(selected),
            hits=len(hits),
            correct_hits=correct,
            incorrect_hits=len(hits) - correct,
            correct_hit_rate=correct / len(selected),
            incorrect_hit_rate=(len(hits) - correct) / len(selected),
            incorrect_fraction_of_hits=(len(hits) - correct) / len(hits) if hits else 0,
            seconds=total,
            net_saved_seconds=baseline - total,
            net_saved_fraction=(baseline - total) / baseline,
            embedding_seconds=sum(r["embedding_seconds"] for r in selected),
            lookup_seconds=sum(r["lookup_seconds"] for r in selected),
            generation_seconds=sum(r["generation_seconds"] for r in selected),
            answer_exact_match=sum(r["answer_correct"] for r in selected) / len(selected),
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "run"])
    parser.add_argument("--cache", type=Path, default=Path(".cache/models"))
    parser.add_argument("--output", type=Path, default=Path("results/run"))
    args = parser.parse_args()
    config = json.loads(Path("data/config.json").read_text())
    if args.command == "download":
        download_models(config, args.cache)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    models, load_seconds = timed(lambda: LocalModels(config, args.cache))
    models.embed(["Warmup"])
    models.generate("Answer briefly: What is two plus two?")
    requests = json.loads(Path("data/requests.json").read_text())
    rows = []
    for policy in ("none", "exact", "semantic", "unsafe_diagnostic"):
        cache = Cache(
            str(args.output / f"{policy}.db"),
            model=config["embedding_model"]["revision"],
            dimensions=384,
            threshold=config["threshold"],
        )
        source = {}
        for request in requests:
            start = time.perf_counter()
            if request["event"]:
                cache.invalidate(**request["event"])
            scope = scope_of(request, policy)
            ticket = cache.ticket()
            query = request["query"]
            hit, lookup_seconds = timed(lambda: cache.lookup(scope, query))
            if policy == "none":
                hit = None
            vector, embedding_seconds, generation_seconds = None, 0.0, 0.0
            if hit is None and policy in ("semantic", "unsafe_diagnostic"):
                vectors, embedding_seconds = timed(lambda: models.embed([query]))
                vector = vectors[0]
                hit, duration = timed(lambda: cache.lookup(scope, query, vector))
                lookup_seconds += duration
            prompt = config["prompt"].format(context=request["context"], question=query)
            if hit is None:
                generated, generation_seconds = timed(lambda: models.generate(prompt))
                answer = generated["answer"]
                if policy != "none":
                    cache.put(scope, query, answer, vector, ttl=3600, ticket=ticket)
                    source[(scope.key(), query)] = request
                correct_hit, origin = None, None
            else:
                generated = None
                answer = hit.answer
                origin = source[(scope.key(), hit.query)]
                correct_hit = all(origin[k] == request[k] for k in ("scope", "intent", "expected"))
            elapsed = time.perf_counter() - start
            row = dict(
                id=request["id"],
                policy=policy,
                hit=asdict(hit) if hit else None,
                source_id=origin["id"] if origin else None,
                correct_hit=correct_hit,
                answer=answer,
                answer_correct=answer.strip().removesuffix(" GB") == request["expected"],
                generated=generated,
                prompt=prompt,
                vector=vector,
                seconds=elapsed,
                embedding_seconds=embedding_seconds,
                lookup_seconds=lookup_seconds,
                generation_seconds=generation_seconds,
            )
            rows.append(row)
            with (args.output / "trace.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        cache.close()
    hashes = {}
    for field in ("embedding_model", "answer_model"):
        model = config[field]
        path = model_path(args.cache, model["id"], model["revision"])
        hashes[field] = {p.name: digest(p) for p in sorted(path.iterdir()) if p.is_file()}
    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    report = dict(
        schema_version=1,
        config=config,
        requests=requests,
        rows=rows,
        summary=summarize(rows),
        model_hashes=hashes,
        environment=dict(
            platform=platform.platform(),
            cpu=cpu,
            python=platform.python_version(),
            threads=config["threads"],
            packages={
                p: importlib.metadata.version(p)
                for p in ["torch", "transformers", "numpy", "tokenizers"]
            },
        ),
        load_seconds=load_seconds,
        implementation_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        input_hashes={p: digest(Path(p)) for p in ["data/config.json", "data/requests.json"]},
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(report["summary"], indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
