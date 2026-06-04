"""
GlidingCache — Core data structures for the Gliding Horse memory architecture.

L2: In-memory knowledge graph (pure Python dict-based adjacency)
L3: SQLite-backed persistent store (stdlib sqlite3)
MESI: Modified / Exclusive / Shared / Invalid coherence tracking

All pure stdlib — no external dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DB_NAME = "gliding_cache.db"
_GRAPH_NAME = "gliding_graph.db"  # same SQLite file, separate tables
_MAX_PREFETCH_HOPS = 2
_MAX_PREFETCH_RESULTS = 5
_MAX_TURNS_PER_SESSION = 200
_MIN_ENTITY_LABEL_LEN = 3

# MESI states
M_MODIFIED = "M"
E_EXCLUSIVE = "E"
S_SHARED = "S"
I_INVALID = "I"

# ---------------------------------------------------------------------------
# MESI Coherence
# ---------------------------------------------------------------------------


class CoherenceTracker:
    """Tracks MESI state per entity across sessions.

    States:
        M (Modified) — entity changed locally, not synced
        E (Exclusive) — entity in this session only, clean
        S (Shared) — entity known to exist in >=2 sessions, clean
        I (Invalid) — entity was invalidated by a write in another session
    """

    def __init__(self):
        self._lock = threading.Lock()
        # entity_id -> state
        self._states: Dict[str, str] = {}
        # entity_id -> set of session_ids that hold it
        self._holders: Dict[str, set] = defaultdict(set)

    def acquire(self, entity_id: str, session_id: str) -> str:
        """Record that a session is using this entity. Returns the state."""
        with self._lock:
            holders = self._holders[entity_id]
            holders.add(session_id)
            if entity_id not in self._states:
                self._states[entity_id] = E_EXCLUSIVE
            elif len(holders) > 1 and self._states[entity_id] != M_MODIFIED:
                self._states[entity_id] = S_SHARED
            return self._states[entity_id]

    def release(self, entity_id: str, session_id: str) -> None:
        """Release a session's hold on an entity."""
        with self._lock:
            holders = self._holders.get(entity_id, set())
            holders.discard(session_id)
            if not holders:
                self._states.pop(entity_id, None)
                self._holders.pop(entity_id, None)

    def mark_modified(self, entity_id: str) -> None:
        """Mark entity as modified (dirty, needs sync on next read)."""
        with self._lock:
            self._states[entity_id] = M_MODIFIED

    def invalidate(self, entity_id: str) -> None:
        """Invalidate entity so next read re-fetches."""
        with self._lock:
            self._states[entity_id] = I_INVALID

    def is_valid(self, entity_id: str) -> bool:
        """Check if entity data is safe to use without re-fetch."""
        with self._lock:
            return self._states.get(entity_id, E_EXCLUSIVE) != I_INVALID

    def state(self, entity_id: str) -> str:
        with self._lock:
            return self._states.get(entity_id, E_EXCLUSIVE)

    def release_session(self, session_id: str) -> None:
        """Release all entities held by a session."""
        with self._lock:
            to_release = [
                eid for eid, holders in self._holders.items()
                if session_id in holders
            ]
            for eid in to_release:
                self.release(eid, session_id)

    def snapshot(self) -> Dict[str, str]:
        """Return a copy of current states for debugging."""
        with self._lock:
            return dict(self._states)


# ---------------------------------------------------------------------------
# L2 — In-Memory Knowledge Graph
# ---------------------------------------------------------------------------


@dataclass
class GraphEntity:
    """A node in the knowledge graph."""
    id: str
    label: str
    type: str = "concept"
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Weight for diffusion activation (0.0-1.0)
    activation: float = 0.0


class KnowledgeGraph:
    """L2 in-memory RDF-like knowledge graph.

    Uses adjacency dict for O(1) edge lookups and pure Python traversal
    for diffusion activation prefetch.
    """

    def __init__(self):
        self._lock = threading.RLock()
        # entity_id -> GraphEntity
        self._entities: Dict[str, GraphEntity] = {}
        # label -> entity_id (for fast lookup)
        self._label_index: Dict[str, str] = {}
        # entity_id -> {predicate -> [target_id]}
        self._edges: Dict[str, Dict[str, List[str]]] = {}
        # Reverse edges: target_id -> {predicate -> [source_id]}
        self._reverse_edges: Dict[str, Dict[str, List[str]]] = {}

    # ---- Entity CRUD ----

    def add_entity(self, label: str, type: str = "concept",
                   metadata: Dict[str, Any] = None) -> str:
        """Add or retrieve an entity by label. Returns entity id."""
        with self._lock:
            lower = label.strip().lower()
            if lower in self._label_index:
                eid = self._label_index[lower]
                # Merge metadata
                if metadata:
                    self._entities[eid].metadata.update(metadata)
                return eid
            if len(label.strip()) < _MIN_ENTITY_LABEL_LEN:
                return ""
            eid = str(uuid.uuid4())[:12]
            self._entities[eid] = GraphEntity(
                id=eid, label=label.strip(), type=type,
                metadata=metadata or {},
            )
            self._label_index[lower] = eid
            return eid

    def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        with self._lock:
            return self._entities.get(entity_id)

    def find_entity(self, label: str) -> Optional[GraphEntity]:
        with self._lock:
            eid = self._label_index.get(label.strip().lower())
            if eid:
                return self._entities.get(eid)
            return None

    def find_entity_by_id(self, entity_id: str) -> Optional[GraphEntity]:
        """Look up an entity by its unique ID."""
        with self._lock:
            return self._entities.get(entity_id)

    def remove_entity(self, entity_id: str) -> None:
        with self._lock:
            ent = self._entities.pop(entity_id, None)
            if ent:
                self._label_index.pop(ent.label.lower(), None)
            self._edges.pop(entity_id, None)
            self._reverse_edges.pop(entity_id, None)
            # Clean up edges referencing this entity
            for src in list(self._edges):
                for pred in list(self._edges[src]):
                    self._edges[src][pred] = [
                        t for t in self._edges[src][pred] if t != entity_id
                    ]
            for tgt in list(self._reverse_edges):
                for pred in list(self._reverse_edges[tgt]):
                    self._reverse_edges[tgt][pred] = [
                        s for s in self._reverse_edges[tgt][pred]
                        if s != entity_id
                    ]

    def entity_count(self) -> int:
        with self._lock:
            return len(self._entities)

    def match_by_type(self, type_name: str) -> List[GraphEntity]:
        with self._lock:
            return [e for e in self._entities.values() if e.type == type_name]

    # ---- Edge CRUD ----

    def add_edge(self, subj_id: str, pred: str, obj_id: str) -> None:
        """Add triple (subj, pred, obj) to the graph."""
        with self._lock:
            if subj_id not in self._entities or obj_id not in self._entities:
                return
            if subj_id not in self._edges:
                self._edges[subj_id] = {}
            if pred not in self._edges[subj_id]:
                self._edges[subj_id][pred] = []
            if obj_id not in self._edges[subj_id][pred]:
                self._edges[subj_id][pred].append(obj_id)

            # Reverse edge
            if obj_id not in self._reverse_edges:
                self._reverse_edges[obj_id] = {}
            if pred not in self._reverse_edges[obj_id]:
                self._reverse_edges[obj_id][pred] = []
            if subj_id not in self._reverse_edges[obj_id][pred]:
                self._reverse_edges[obj_id][pred].append(subj_id)

    def get_neighbors(self, entity_id: str,
                      max_hops: int = 1) -> Dict[str, float]:
        """Get all entities reachable within max_hops.

        Returns dict of {entity_id: score} where score decays with distance.
        """
        with self._lock:
            if entity_id not in self._entities:
                return {}

            visited: Dict[str, float] = {}
            queue: List[Tuple[str, int]] = [(entity_id, 0)]
            visited[entity_id] = 1.0

            while queue:
                current, depth = queue.pop(0)
                if depth >= max_hops:
                    continue
                next_depth = depth + 1
                decay = 1.0 / (next_depth + 1)

                # Outgoing edges
                for pred, targets in self._edges.get(current, {}).items():
                    for tgt in targets:
                        if tgt not in visited or visited[tgt] < decay * 0.8:
                            visited[tgt] = decay * 0.8
                            queue.append((tgt, next_depth))

                # Incoming edges
                for pred, sources in self._reverse_edges.get(current, {}).items():
                    for src in sources:
                        if src not in visited or visited[src] < decay * 0.7:
                            visited[src] = decay * 0.7
                            queue.append((src, next_depth))

            # Remove self
            visited.pop(entity_id, None)
            return visited

    def query_entities(self, text: str) -> List[Tuple[str, float]]:
        """Basic keyword matching for entity labels.

        Scores by substring match, then partial word match.
        Used as a lightweight pre-filter before diffusion.
        """
        with self._lock:
            lower = text.lower()
            words = set(lower.split())
            scored: List[Tuple[str, float]] = []

            for eid, ent in self._entities.items():
                el = ent.label.lower()
                score = 0.0
                if lower in el:
                    score += 1.0
                if el in lower:
                    score += 0.8
                for w in words:
                    if w in el:
                        score += 0.3
                    if el in w:
                        score += 0.2
                if score > 0:
                    scored.append((eid, min(score, 2.0)))

            scored.sort(key=lambda x: -x[1])
            return scored

    def snapshot(self) -> Dict[str, Any]:
        """Serialize graph state for debugging/display."""
        with self._lock:
            return {
                "entity_count": len(self._entities),
                "edge_count": sum(
                    sum(len(tgts) for tgts in preds.values())
                    for preds in self._edges.values()
                ),
                "entities": [
                    {"id": e.id, "label": e.label, "type": e.type, "edges": len(self._edges.get(e.id, {}))}
                    for e in list(self._entities.values())[:20]
                ],
            }


# ---------------------------------------------------------------------------
# L3 — SQLite Persistent Store
# ---------------------------------------------------------------------------


class PersistentStore:
    """L3 long-term storage backed by SQLite.

    Stores entities, triples, and session turns.
    """

    def __init__(self, hermes_home: str):
        self._hermes_home = hermes_home
        self._db_path = str(Path(hermes_home) / _DB_NAME)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Create tables on first use."""
        with self._lock:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    type TEXT DEFAULT 'concept',
                    metadata TEXT DEFAULT '{}',
                    created_at REAL DEFAULT (julianday('now')),
                    updated_at REAL DEFAULT (julianday('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_entities_label ON entities(label);

                CREATE TABLE IF NOT EXISTS triples (
                    subj_id TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    obj_id TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    created_at REAL DEFAULT (julianday('now')),
                    PRIMARY KEY (subj_id, predicate, obj_id),
                    FOREIGN KEY (subj_id) REFERENCES entities(id),
                    FOREIGN KEY (obj_id) REFERENCES entities(id)
                );
                CREATE INDEX IF NOT EXISTS idx_triples_subj ON triples(subj_id);
                CREATE INDEX IF NOT EXISTS idx_triples_obj ON triples(obj_id);

                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_number INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    summary TEXT,
                    created_at REAL DEFAULT (julianday('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    label TEXT,
                    turn_count INTEGER DEFAULT 0,
                    summary TEXT,
                    created_at REAL DEFAULT (julianday('now')),
                    updated_at REAL DEFAULT (julianday('now'))
                );
            """)
            self._conn.commit()

    def persist_entity(self, eid: str, label: str, type_: str = "concept",
                       metadata: Dict[str, Any] = None) -> None:
        with self._lock:
            if not self._conn:
                return
            self._conn.execute(
                """INSERT OR REPLACE INTO entities (id, label, type, metadata, updated_at)
                   VALUES (?, ?, ?, ?, julianday('now'))""",
                (eid, label, type_, json.dumps(metadata or {}, ensure_ascii=False)),
            )
            self._conn.commit()

    def persist_triple(self, subj_id: str, pred: str, obj_id: str,
                       weight: float = 1.0) -> None:
        with self._lock:
            if not self._conn:
                return
            self._conn.execute(
                """INSERT OR REPLACE INTO triples (subj_id, predicate, obj_id, weight)
                   VALUES (?, ?, ?, ?)""",
                (subj_id, pred, obj_id, weight),
            )
            self._conn.commit()

    def persist_turn(self, session_id: str, turn_number: int,
                     role: str, content: str, summary: str = "") -> None:
        with self._lock:
            if not self._conn:
                return
            self._conn.execute(
                """INSERT INTO turns (session_id, turn_number, role, content, summary)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, turn_number, role, content, summary),
            )
            # Upsert session
            self._conn.execute(
                """INSERT INTO sessions (session_id, turn_count, updated_at)
                   VALUES (?, 1, julianday('now'))
                   ON CONFLICT(session_id) DO UPDATE SET
                       turn_count = turn_count + 1,
                       updated_at = julianday('now')""",
                (session_id,),
            )
            self._conn.commit()

    def load_all_entities(self) -> List[Dict[str, Any]]:
        """Load all entities from L3 into L2 on startup."""
        with self._lock:
            if not self._conn:
                return []
            rows = self._conn.execute(
                "SELECT id, label, type, metadata FROM entities"
            ).fetchall()
            return [
                {"id": r[0], "label": r[1], "type": r[2],
                 "metadata": json.loads(r[3] or "{}")}
                for r in rows
            ]

    def load_all_triples(self) -> List[Dict[str, Any]]:
        """Load all triples from L3 into L2 on startup."""
        with self._lock:
            if not self._conn:
                return []
            rows = self._conn.execute(
                "SELECT subj_id, predicate, obj_id, weight FROM triples"
            ).fetchall()
            return [
                {"subj": r[0], "pred": r[1], "obj": r[2], "weight": r[3]}
                for r in rows
            ]

    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """FTS-like search over entity labels and metadata."""
        with self._lock:
            if not self._conn:
                return []
            # Use LIKE for basic matching (no FTS5 dependency)
            pattern = f"%{query}%"
            rows = self._conn.execute(
                """SELECT id, label, type, metadata
                   FROM entities
                   WHERE label LIKE ? OR metadata LIKE ?
                   LIMIT ?""",
                (pattern, pattern, limit),
            ).fetchall()
            return [
                {"id": r[0], "label": r[1], "type": r[2],
                 "metadata": json.loads(r[3] or "{}")}
                for r in rows
            ]

    def get_session_summary(self, session_id: str) -> str:
        """Get accumulated summary for a session."""
        with self._lock:
            if not self._conn:
                return ""
            row = self._conn.execute(
                "SELECT summary FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row[0] if row and row[0] else ""

    def update_session_summary(self, session_id: str, summary: str) -> None:
        with self._lock:
            if not self._conn:
                return
            self._conn.execute(
                "UPDATE sessions SET summary = ?, updated_at = julianday('now') WHERE session_id = ?",
                (summary, session_id),
            )
            self._conn.commit()

    def shutdown(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    def db_stats(self) -> Dict[str, int]:
        with self._lock:
            if not self._conn:
                return {"entities": 0, "triples": 0, "turns": 0, "sessions": 0}
            ents = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            triples = self._conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
            turns = self._conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
            sessions = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            return {"entities": ents, "triples": triples, "turns": turns, "sessions": sessions}


# ---------------------------------------------------------------------------
# Prefetch Engine (Diffusion Activation)
# ---------------------------------------------------------------------------


class PrefetchEngine:
    """Diffusion activation prefetcher.

    Given a query, finds seed entities via keyword matching, then spreads
    activation through the graph for _MAX_PREFETCH_HOPS hops, collecting
    related entities to inject as context.
    """

    def __init__(self, graph: KnowledgeGraph):
        self._graph = graph
        self._cached_result: str = ""
        self._cache_lock = threading.Lock()

    def prefetch(self, query: str) -> str:
        """Run diffusion activation and return formatted context.

        Steps:
        1. Keyword match query against entity labels (L2)
        2. For matched entities, diffuse activation through graph edges
        3. Collect top-K by activation score
        4. Format as markdown context block
        """
        if not query or len(query.strip()) < _MIN_ENTITY_LABEL_LEN:
            return ""

        # Step 1: Find seed entities
        seeds = self._graph.query_entities(query)
        if not seeds:
            return ""

        # Step 2: Diffuse activation
        activated: Dict[str, float] = {}  # entity_id -> score
        seen: set = set()

        for eid, base_score in seeds[:3]:  # Use top 3 seeds
            neighbors = self._graph.get_neighbors(eid, max_hops=_MAX_PREFETCH_HOPS)
            for neighbor_id, n_score in neighbors.items():
                combined = base_score * n_score
                if neighbor_id not in activated or activated[neighbor_id] < combined:
                    activated[neighbor_id] = combined
            seen.add(eid)

        # Step 3: Sort and limit
        ranked = sorted(activated.items(), key=lambda x: -x[1])
        results: List[Tuple[str, float]] = []

        # Include the matched seeds themselves (they're obviously relevant)
        seen_list = list(seen)
        for eid in seen_list:
            results.append((eid, 2.0))

        for eid, score in ranked:
            if eid not in seen and len(results) < _MAX_PREFETCH_RESULTS:
                results.append((eid, score))
                seen.add(eid)

        if not results:
            return ""

        # Step 4: Format
        lines = ["📌 **Gliding Memory — Related Context**"]
        for eid, score in results:
            ent = self._graph.get_entity(eid)
            if ent:
                score_pct = int(score * 100)
                meta_str = ""
                if ent.metadata:
                    # Show 1-2 key metadata items
                    meta_items = []
                    for k, v in list(ent.metadata.items())[:2]:
                        if isinstance(v, str) and len(v) > 40:
                            v = v[:40] + "..."
                        meta_items.append(f"{k}={v}")
                    meta_str = f" ({', '.join(meta_items)})" if meta_items else ""
                lines.append(f"- **{ent.label}** [{ent.type}] (rel: {score_pct}%){meta_str}")

        with self._cache_lock:
            self._cached_result = "\n".join(lines)

        return self._cached_result

    def queue_prefetch(self, query: str) -> None:
        """Queue a background prefetch for the next turn."""
        # In the current implementation, we run synchronously
        # (threading can be added later for high-latency backends)
        pass

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cached_result = ""
