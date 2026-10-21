"""Replay cached decisions without importing model dependencies."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from semantic_cache import Cache

from .evaluate import scope_of, summarize


def verify(report):
    config = report["config"]
    requests = report["requests"]
    ids = [r["id"] for r in requests]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("Request identifiers must be unique and nonempty")
    if len(report["rows"]) != 4 * len(requests):
        raise ValueError("Incomplete policy trace")
    for policy in ("none", "exact", "semantic", "unsafe_diagnostic"):
        cache = Cache(
            model=config["embedding_model"]["revision"],
            dimensions=384,
            threshold=config["threshold"],
            clock=lambda: 0,
        )
        source = {}
        try:
            rows = [r for r in report["rows"] if r["policy"] == policy]
            if [r["id"] for r in rows] != ids:
                raise ValueError("Missing or reordered requests")
            for request, row in zip(requests, rows, strict=True):
                if request["event"] and policy != "none":
                    cache.invalidate(**request["event"])
                scope = scope_of(request, policy)
                hit = None if policy == "none" else cache.lookup(scope, request["query"])
                if hit is None and policy in ("semantic", "unsafe_diagnostic"):
                    if row["vector"] is None:
                        raise ValueError("Missing measured embedding")
                    hit = cache.lookup(scope, request["query"], row["vector"])
                elif row["vector"] is not None:
                    raise ValueError("Unexpected embedding")
                expected_hit = asdict(hit) if hit else None
                if row["hit"] != expected_hit:
                    raise ValueError("Cache decision mismatch")
                prompt = config["prompt"].format(
                    context=request["context"], question=request["query"]
                )
                if row["prompt"] != prompt:
                    raise ValueError("Prompt mismatch")
                if hit:
                    origin = source[(scope.key(), hit.query)]
                    correct = all(origin[k] == request[k] for k in ("scope", "intent", "expected"))
                    if row["source_id"] != origin["id"] or row["correct_hit"] != correct:
                        raise ValueError("Hit label mismatch")
                    if row["answer"] != hit.answer or row["generated"] is not None:
                        raise ValueError("Hit answer mismatch")
                else:
                    if row["source_id"] is not None or row["correct_hit"] is not None:
                        raise ValueError("Miss has source label")
                    if row["generated"]["answer"] != row["answer"]:
                        raise ValueError("Generated answer mismatch")
                    if policy != "none":
                        cache.put(
                            scope,
                            request["query"],
                            row["answer"],
                            row["vector"],
                            ttl=3600,
                            ticket=cache.ticket(),
                        )
                        source[(scope.key(), request["query"])] = request
                answer_correct = row["answer"].strip().removesuffix(" GB") == request["expected"]
                if row["answer_correct"] != answer_correct:
                    raise ValueError("Answer quality mismatch")
                fields = ["seconds", "lookup_seconds", "embedding_seconds", "generation_seconds"]
                if any(not math.isfinite(row[k]) or row[k] < 0 for k in fields):
                    raise ValueError("Invalid duration")
                if sum(row[k] for k in fields[1:]) > row["seconds"] + 1e-9:
                    raise ValueError("Duration accounting mismatch")
        finally:
            cache.close()
    if report["summary"] != summarize(report["rows"]):
        raise ValueError("Summary mismatch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    verify(report)
    for name, expected in report["input_hashes"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError("Input file digest mismatch")
    trace = [
        json.loads(line) for line in args.report.with_name("trace.jsonl").read_text().splitlines()
    ]
    if trace != report["rows"]:
        raise ValueError("Trace mismatch")
    if json.loads(args.report.with_name("summary.json").read_text()) != report["summary"]:
        raise ValueError("Summary file mismatch")
    print("Verified decisions, isolation labels, prompts, answer scores, timings and input hashes.")


if __name__ == "__main__":
    main()
