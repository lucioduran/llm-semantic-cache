import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from semantic_cache import Cache, Scope


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.cache = Cache(model="test-double-v1", dimensions=2, clock=lambda: self.now)
        self.addCleanup(self.cache.close)
        self.scope = Scope("alice", ("read",), "v1", "Atlas", "2026-01-01")

    def put(self, **kwargs):
        args = dict(
            scope=self.scope,
            query="quota",
            answer="10",
            vector=[1, 0],
            ttl=10,
            ticket=self.cache.ticket(),
        )
        args.update(kwargs)
        return self.cache.put(**args)

    def test_exact_semantic_and_miss(self):
        self.put()
        self.assertEqual(self.cache.lookup(self.scope, "quota").kind, "exact")
        self.assertEqual(self.cache.lookup(self.scope, "limit", [1, 0]).answer, "10")
        self.assertIsNone(self.cache.lookup(self.scope, "limit", [0, 1]))
        self.assertIsNone(self.cache.lookup(self.scope, "limit"))

    def test_each_boundary_blocks_identical_query_and_embedding(self):
        self.put()
        for field, value in [
            ("user", "bob"),
            ("permissions", ()),
            ("version", "v2"),
            ("entity", "Orion"),
            ("date", "2026-01-02"),
        ]:
            other = replace(self.scope, **{field: value})
            self.assertIsNone(self.cache.lookup(other, "quota", [1, 0]))

    def test_permissions_order_and_duplicates_are_canonical(self):
        scope = replace(self.scope, permissions=("read", "write", "read"))
        self.put(scope=scope)
        self.assertIsNotNone(
            self.cache.lookup(replace(scope, permissions=("write", "read")), "quota")
        )

    def test_update_expiration_and_prune(self):
        self.put()
        self.put(answer="20")
        self.assertEqual(self.cache.lookup(self.scope, "quota").answer, "20")
        self.now = 110
        self.assertIsNone(self.cache.lookup(self.scope, "quota"))
        self.assertEqual(self.cache.prune(), 1)
        self.assertEqual(self.cache.prune(), 0)

    def test_invalidation_fences_inflight_generation(self):
        ticket = self.cache.ticket()
        self.put()
        self.put(scope=replace(self.scope, user="bob"))
        self.assertEqual(self.cache.invalidate(user="alice"), 1)
        self.assertFalse(self.put(ticket=ticket))
        self.assertIsNone(self.cache.lookup(self.scope, "quota"))
        self.assertEqual(self.cache.invalidate(version="v1"), 1)
        self.assertTrue(self.put())
        self.assertEqual(self.cache.invalidate(user="alice", version="v2"), 0)
        self.assertEqual(self.cache.invalidate(user="alice", version="v1"), 1)

    def test_best_match_and_tie_are_deterministic(self):
        self.put(query="b", vector=[1, 0.2])
        self.put(query="c", vector=[1, 0])
        self.put(query="a", vector=[1, 0])
        self.assertEqual(self.cache.lookup(self.scope, "new", [1, 0]).query, "a")

    def test_invalid_input(self):
        for vector in ([0, 0], [1], [float("nan"), 1], [float("inf"), 1], [1e308, 1]):
            with self.assertRaises(ValueError):
                self.put(vector=vector)
        for kwargs in ({"ttl": 0}, {"ttl": float("inf")}, {"query": " "}, {"answer": ""}):
            with self.assertRaises(ValueError):
                self.put(**kwargs)
        for scope in (
            replace(self.scope, user=""),
            replace(self.scope, permissions=("",)),
            replace(self.scope, permissions="read"),
        ):
            with self.assertRaises(ValueError):
                self.cache.lookup(scope, "quota")
        for kwargs in ({}, {"user": ""}, {"version": 3}):
            with self.assertRaises(ValueError):
                self.cache.invalidate(**kwargs)
        for kwargs in (
            {"model": ""},
            {"dimensions": 0},
            {"threshold": -1},
            {"threshold": float("nan")},
        ):
            args = dict(model="test", dimensions=2)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                Cache(**args)

    def test_persistence_models_and_concurrent_connections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/cache.db"
            caches = [Cache(path, model="a", dimensions=2) for _ in range(4)]
            try:

                def write(i):
                    cache = caches[i % 4]
                    return cache.put(
                        self.scope, str(i), str(i), [1, 0], ttl=30, ticket=cache.ticket()
                    )

                with ThreadPoolExecutor(max_workers=4) as pool:
                    self.assertTrue(all(pool.map(write, range(40))))
                for i in range(40):
                    self.assertEqual(caches[0].lookup(self.scope, str(i)).answer, str(i))
                old = caches[1].ticket()
                self.assertEqual(caches[0].invalidate(user="alice"), 40)
                self.assertFalse(
                    caches[1].put(self.scope, "old", "old", [1, 0], ttl=30, ticket=old)
                )
                write(0)
            finally:
                for cache in caches:
                    cache.close()
            reopened = Cache(path, model="a", dimensions=2)
            self.assertIsNotNone(reopened.lookup(self.scope, "0"))
            reopened.close()
            for model, dimensions in [("b", 2), ("a", 3)]:
                other = Cache(path, model=model, dimensions=dimensions)
                self.assertIsNone(other.lookup(self.scope, "0"))
                other.close()

    def test_transaction_rollbacks(self):
        self.cache.db.execute(
            "CREATE TRIGGER fail BEFORE INSERT ON entries BEGIN SELECT RAISE(ABORT, 'failure'); END"
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.put()
        self.cache.db.execute("DROP TRIGGER fail")
        self.put()
        self.cache.db.execute(
            "CREATE TRIGGER fail BEFORE DELETE ON entries BEGIN SELECT RAISE(ABORT, 'failure'); END"
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.cache.invalidate(user="alice")
        self.assertIsNotNone(self.cache.lookup(self.scope, "quota"))

    def test_exact_only_entries(self):
        self.put(vector=None)
        self.assertIsNotNone(self.cache.lookup(self.scope, "quota"))
        self.assertIsNone(self.cache.lookup(self.scope, "limit", [1, 0]))
