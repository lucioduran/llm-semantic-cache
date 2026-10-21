# Cache contract

A scope contains a user, a canonical permission set, a version, an entity, and a date. All fields except the permission set must be nonempty strings. An empty permission set is a valid isolated scope. Scope serialization uses JSON rather than a delimiter that could collide. Exact query keys preserve whitespace and case. Model revision and dimension are additional namespaces.

Embeddings must be finite, nonzero vectors with the configured dimension. The cache normalizes them before cosine comparison. Ties resolve by query lexicographic order. Exact hits bypass cosine scoring. Entries without a vector participate only in exact matching.

SQLite uses parameterized values. A scope's primary-key prefix bounds candidate retrieval; each candidate is checked for expiration. The database is trusted local storage, not a hostile file format. One connection is guarded by an RLock; independent connections coordinate through SQLite locking and a 30-second busy timeout. This is not a network-filesystem or distributed locking guarantee.

Writes use BEGIN IMMEDIATE. Invalidation atomically deletes selected entries and increments a global epoch. Callers capture a ticket before source retrieval and generation. A stale ticket rejects the write even if its key was absent when invalidation ran. This deliberately sacrifices unrelated in-flight writes for a simpler correctness contract. Rejected writes must not be treated as fresh authorized answers by the application.

A read linearizes when its SELECT establishes a snapshot. If an invalidation overlaps a read, the read may return the prior snapshot, even after the invalidation finishes. Applications needing revocation acknowledgement across in-flight responses need an additional admission and response barrier. Source changes and cache invalidation are not a single transaction across external databases.

TTL uses wall-clock Unix time for persistent entries. A backwards clock adjustment can prolong lifetime; use a disciplined host clock and source versions for correctness-critical freshness. Pruning is explicit. Invalidating rows does not securely erase SQLite pages or backups.
