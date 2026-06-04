"""
GlidingCache — Gliding Horse Memory Provider for Hermes Agent.

Implements a tiered memory architecture (L0-L3) with diffusion-activation
prefetch and MESI-like coherence tracking, inspired by the Gliding Horse
agent memory system.

Architecture:
  L3 (persistent)  — SQLite-backed entity/triple/turn store
  L2 (working)     — In-memory RDF-like knowledge graph
  L1 (prompt)      — System prompt block with active entities + summary
  L0 (immediate)   — Current conversation context (handled by Hermes)

Key features:
  - Diffusion activation prefetch: keyword match → graph traverse → inject
  - MESI coherence: track entity states across sessions to prevent staleness
  - Auto-extraction: entity and triple extraction from user/assistant turns
  - LLM-driven NER with regex fallback for hybrid entity extraction
  - Auto-linking: co-occurrence edges + transitive inference across entities
  - Zero external dependencies beyond stdlib + pyyaml (config read)

Usage in config.yaml:
  memory:
    provider: glidingcache
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from agent.memory_provider import MemoryProvider
from plugins.memory.glidingcache.core import (
    CoherenceTracker,
    KnowledgeGraph,
    PersistentStore,
    PrefetchEngine,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_L1_CHARS = 6000  # Max chars for the system prompt block
_MAX_TURNS_BEFORE_SUMMARY = 10  # Summarize every N turns
_MIN_CONTENT_FOR_ENTITY = 15  # Min content length to attempt entity extraction

# Entity extraction patterns (regex fallback)
_ENTITY_PATTERN = re.compile(
    r'(?:关于|谈到|提到|讨论|涉及|我们(?:的)?(?:项目|系统|工具|代码|服务器|模型|框架|平台))?'
    r'[\u201c\u201d\u300c]([^\u201c\u201d\u300d]{2,40})[\u201c\u201d\u300d]'
)
_TOPIC_PATTERN = re.compile(
    r'(?:话题|主题|方向|领域|方面|模块|组件|功能|特性):\s*[\u201c\u201d]?([^,。，\n]{2,30})[\u201c\u201d]?'
)

# LLM NER configuration
_LLM_NER_PROMPT = (
    "你是一个命名实体识别（NER）工具。从以下对话内容中提取所有有意义的实体——"
    "包括项目名、工具名、技术术语、概念名、专有名词、缩写等。\n\n"
    "要求：\n"
    "- 每个实体应是一个独立的、有实际意义的名称或术语（2~40个字符）\n"
    "- 去除通用词（\"这个\"、\"那个\"、\"问题\"、\"东西\"等）\n"
    "- 如果内容中没有实体，返回空数组 []\n"
    "- 返回纯 JSON 数组，不要其他文字\n"
    '示例：["Gliding Horse", "Hermes Agent", "MESI", "SQLite", "知识图谱"]'
)
_LLM_NER_SYSTEM_TRUNCATED = _LLM_NER_PROMPT[:120] + "..."

_NER_CACHE_TTL = 300  # seconds to cache NER results for the same content
_LLM_RATE_LIMIT_SECS = 8  # minimum seconds between LLM calls
_MIN_LLM_NER_LENGTH = 60  # minimum content length to attempt LLM NER
_LLM_NER_TIMEOUT = 12  # seconds to wait for LLM response

# Entity quality filtering
_NER_STOPLIST = frozenset({
    "a", "an", "the", "it", "is", "be", "to", "of", "in", "for", "on",
    "and", "or", "but", "with", "from", "by", "at", "this", "that",
    "true", "false", "null", "none", "yes", "no",
    "name", "type", "value", "key", "id", "data", "info", "text",
    "description", "summary", "content", "result", "status", "error",
    "code", "config", "setup", "mode", "path", "file", "dir", "list",
    "parameters", "params", "args", "size", "limit", "count",
    "source", "target", "role", "label", "example", "option",
    "default", "custom", "basic", "core", "main", "user", "admin",
    "cold", "stale", "new", "old", "empty", "full", "local",
    # Programming/JSON noise
    "co_occur", "memory_entry", "memory_write", "system_prompt",
    "kwargs", "import", "return", "yield", "raise", "except",
    # Single French/German words that aren't entity names
    "la", "le", "les", "des", "das", "der", "die", "ein", "und",
})
_NER_MIN_LABEL_LEN = 3  # Minimum entity label length
_NER_MAX_LABEL_LEN = 50  # Maximum entity label length
_BASE64_PATTERN = re.compile(r'^[A-Za-z0-9+/=]{20,}$')  # base64-like strings
_NUMERIC_LIKE = re.compile(r'^\d[\d.%,/-]*$')  # pure numbers or percentages
_CODEPATH_LIKE = re.compile(r'^[a-z_][a-z0-9_]*\(\)?$')  # function names like foo() or foo_bar

# Auto-linking
_CO_OCCUR_PREDICATE = "co_occur"
_INFERRED_PREDICATE = "related"
_TRANSITIVE_HOPS = 1  # depth for transitive inference
_MAX_AUTO_LINK_ENTITIES = 30  # cap pairs to avoid O(n²) explosion

# Ollama embedding for semantic entity dedup (Phase 2.1)
_EMBEDDING_MODEL = "bge-small-zh"  # lightest: 23.69M, 512d
_EMBEDDING_URL = "http://localhost:11434/api/embeddings"
_EMBEDDING_DEDUP_THRESHOLD = 0.85
_EMBEDDING_CACHE_TTL = 600
_EMBED_ENABLED = True
_EMBED_LABEL_MIN_LEN = 5

# GBrain integration (Phase 2.1)
_GBRAIN_BIN = os.path.expanduser("~/.bun/bin/gbrain")
_GBRAIN_CAPTURE_TIMEOUT = 20
_GBRAIN_SLUG_PREFIX = "gliding/turn_"

# GBrain query fallback (Phase 2.2)
_GBRAIN_QUERY_TIMEOUT = 10  # seconds
_GBRAIN_QUERY_LIMIT = 5  # max results to pull
_GBRAIN_SEARCH_MIN_LEN = 8  # min query length to bother with GBrain
_WEAK_RESULT_ENTITY_THRESHOLD = 2  # fewer entities = weak, fall back to GBrain


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class GlidingCacheMemoryProvider(MemoryProvider):
    '''Gliding Horse tiered memory provider with diffusion prefetch.

    v0.2.0 additions:
      - LLM-driven NER extraction with regex fallback
      - Auto co-occurrence linking + transitive inference

    v0.3.0 additions (Phase 2.1):
      - GBrain capture push on each sync_turn
      - Ollama embedding-based entity semantic dedup
    '''

    def __init__(self):
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._platform: str = ""
        self._agent_context: str = ""
        self._initialized = False
        self._turn_counter = 0

        # Core subsystems
        self._graph: Optional[KnowledgeGraph] = None
        self._store: Optional[PersistentStore] = None
        self._coherence: Optional[CoherenceTracker] = None
        self._prefetch: Optional[PrefetchEngine] = None

        # LLM NER state
        self._llm_base_url: str = ""
        self._llm_api_key: str = ""
        self._llm_model: str = ""
        self._llm_available = False
        self._ner_cache: Dict[str, Tuple[float, List[str]]] = {}
        self._last_llm_call: float = 0.0

        # Embedding dedup state (Phase 2.1)
        self._embed_cache: Dict[str, Tuple[float, List[float]]] = {}  # label -> (timestamp, vec)
        self._embed_available = False

        # GBrain integration state (Phase 2.1)
        self._gbrain_available = False

    # -- Identity ----------------------------------------------------------

    @property
    def name(self) -> str:
        return "glidingcache"

    # -- Availability ------------------------------------------------------

    def is_available(self) -> bool:
        """Always available — no external services needed."""
        return True

    # -- Lifecycle ---------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        """Set up the L2 graph and L3 store for this session.

        On first init (per hermes_home), loads existing entities from
        the SQLite store into the graph. Subsequent inits reuse the
        shared graph/store singletons.
        """
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", "")
        self._platform = kwargs.get("platform", "cli")
        self._agent_context = kwargs.get("agent_context", "primary")
        self._turn_counter = 0

        # Initialize subsystems (singletons across sessions)
        if self._store is None:
            self._store = PersistentStore(
                self._hermes_home or str(__import__("pathlib").Path.home() / ".hermes")
            )
            self._store.initialize()
            self._graph = KnowledgeGraph()
            self._coherence = CoherenceTracker()
            self._graph = self._warm_graph()

        self._prefetch = PrefetchEngine(self._graph)

        # Load LLM config for NER
        self._load_llm_config()

        # Detect GBrain availability (Phase 2.1)
        self._detect_gbrain()

        # Warm embedding cache from existing entities (Phase 2.1)
        self._warm_embed_cache()

        self._initialized = True
        logger.info(
            "GlidingCache initialized: session=%s, platform=%s, ctx=%s, "
            "llm_ner=%s, embed=%s, gbrain=%s",
            session_id[:12], self._platform, self._agent_context,
            self._llm_available, self._embed_available, self._gbrain_available,
        )

    def _warm_graph(self) -> KnowledgeGraph:
        """Load existing entities and triples from L3 into L2."""
        g = KnowledgeGraph()
        if not self._store:
            return g

        # Track ID mapping: store_id -> graph_id (they should match after add_entity)
        id_map: Dict[str, str] = {}

        entities = self._store.load_all_entities()
        for ent in entities:
            eid = g.add_entity(ent["label"], ent["type"], ent.get("metadata", {}))
            if not eid and ent["label"]:
                # Entity too short for auto-index; try with original id
                eid = ent["id"]
            if eid:
                id_map[ent["id"]] = eid

        triples = self._store.load_all_triples()
        for t in triples:
            # Map using either original store ID or label (fallback to original key)
            subj_id = id_map.get(t["subj"]) or t["subj"]
            obj_id = id_map.get(t["obj"]) or t["obj"]
            g.add_edge(subj_id, t["pred"], obj_id)

        logger.info("L2 graph warmed: %d entities, %d triples",
                     g.entity_count(), len(triples))
        return g

    def system_prompt_block(self) -> str:
        """Generate the L1 prompt block — active entities plus session context.

        Stays under _MAX_L1_CHARS to avoid bloating the system prompt.
        """
        if not self._graph or not self._store:
            return ""

        blocks: List[str] = []
        blocks.append("# 🐎 Gliding Memory Context")

        # Session summary (L1 persistent summary)
        session_summary = self._store.get_session_summary(self._session_id)
        if session_summary:
            blocks.append(f"## Session Summary\n{session_summary[:500]}")

        # Active entities (top entities from L2 with recent modifications)
        if self._coherence:
            valid_entities = []
            if self._graph and self._coherence:
                # Sample recent/active entities
                snapshot = self._graph.snapshot()
                entity_list = snapshot.get("entities", [])[:8]
                if entity_list:
                    blocks.append("## Active Entities")
                    for e in entity_list:
                        state = self._coherence.state(e["id"])
                        state_mark = {"M": "📝", "E": "✅", "S": "🔄", "I": "⛔"}.get(state, "❓")
                        blocks.append(f"- {state_mark} **{e['label']}** [{e['type']}] (edges: {e['edges']})")

        # DB stats
        if self._store:
            stats = self._store.db_stats()
            blocks.append(f"\n*L3 store: {stats['entities']} entities, {stats['triples']} triples, {stats['sessions']} sessions*")

        result = "\n\n".join(blocks)

        # Trim to max chars
        if len(result) > _MAX_L1_CHARS:
            result = result[:_MAX_L1_CHARS] + "\n\n*(context trimmed)*"

        return result if result != "# 🐎 Gliding Memory Context" else ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Run diffusion activation prefetch — GBrain fallback if weak.

        Two-stage strategy (Phase 2.2):
          1. Graph-based diffusion activation (existing PrefetchEngine)
          2. If result is weak (< 2 entities), fall back to GBrain keyword search
          3. Merge: graph entities first, GBrain semantic pages second

        Returns formatted markdown context block, or empty string.
        """
        graph_result = ""
        if self._prefetch and query:
            graph_result = self._prefetch.prefetch(query)

        # If graph prefetch is already strong, return it as-is
        if not self._is_weak_prefetch_result(graph_result):
            return graph_result

        # Weak result — try GBrain semantic fallback
        if not self._gbrain_available or len(query.strip()) < _GBRAIN_SEARCH_MIN_LEN:
            return graph_result  # return whatever graph gave (maybe empty)

        gbrain_result = self._gbrain_search_prefetch(query)
        if not gbrain_result:
            return graph_result

        # Merge: graph entities first, GBrain pages second
        return self._merge_prefetch_results(graph_result, gbrain_result)

    def _is_weak_prefetch_result(self, result: str) -> bool:
        """Check if graph prefetch returned too few results to be useful.

        Counts lines like '- **EntityName** [...type...] (rel: N%)'.
        Fewer than _WEAK_RESULT_ENTITY_THRESHOLD = weak → trigger GBrain.
        """
        if not result:
            return True
        entity_count = sum(
            1 for line in result.split("\n")
            if line.startswith("- **")
        )
        return entity_count < _WEAK_RESULT_ENTITY_THRESHOLD

    def _gbrain_search_prefetch(self, query: str) -> str:
        """Run GBrain keyword search and return formatted results.

        Uses gbrain search (keyword-based, fast) rather than query
        (hybrid vector+keyword) because prefetch needs low latency.
        Returns empty string if query returns nothing or errors out.
        """
        if not query.strip():
            return ""
        try:
            result = subprocess.run(
                [_GBRAIN_BIN, "search", query,
                 "--limit", str(_GBRAIN_QUERY_LIMIT)],
                capture_output=True, text=True,
                timeout=_GBRAIN_QUERY_TIMEOUT,
                env=self._gbrain_env(),
            )
            if result.returncode != 0 or not result.stdout.strip():
                return ""

            # Parse output: "[0.3462] slug -- title\n\ncontent...\n\nnext..."
            # Each result block starts with "[score] slug -- title_line"
            # followed by optional content lines
            lines = result.stdout.strip().split("\n")
            parsed: List[Tuple[str, str, str]] = []  # (score, slug, title)

            for line in lines:
                line = line.strip()
                if line.startswith("[") and "]" in line and "--" in line:
                    score_end = line.index("]")
                    score_str = line[1:score_end].strip()
                    after_score = line[score_end + 1:].strip()
                    if "--" in after_score:
                        slug_part, title_part = after_score.split("--", 1)
                        slug = slug_part.strip()
                        # Clean title: strip markdown header markers, quotes, limit length
                        title = title_part.strip().lstrip("#").strip()
                        if title.startswith(("'", '"')):
                            title = title.strip("'\"")
                        # Collapse whitespace
                        title = " ".join(title.split())
                        if len(title) > 60:
                            title = title[:60] + "..."
                        parsed.append((score_str, slug, title))

            if not parsed:
                return ""

            # Build formatted output — fewer subprocess calls than gbrain get per item
            blocks = ["📄 **Semantically Related Pages (GBrain)**"]
            for score_str, slug, title in parsed:
                display = title if title else slug
                blocks.append(
                    f"- **{display}** [score: {score_str}] — `{slug}`"
                )

            return "\n".join(blocks)

        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("GBrain search prefetch failed: %s", e)
            return ""

    def _merge_prefetch_results(self, graph_result: str, gbrain_result: str) -> str:
        """Merge graph and GBrain results into a single context block.

        Graph entities (structural relevance) come first, GBrain pages
        (semantic relevance) come second. Dedup by checking entity labels
        don't repeat in the GBrain section.
        """
        if not graph_result and not gbrain_result:
            return ""
        if not graph_result:
            return gbrain_result
        if not gbrain_result:
            return graph_result

        # Graph result already has its header — just tack GBrain on
        return f"{graph_result}\n\n{gbrain_result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue prefetch for next turn (placeholder for async)."""
        if self._prefetch:
            self._prefetch.queue_prefetch(query)

    def sync_turn(self, user_content: str, assistant_content: str,
                  *, session_id: str = "", messages: Optional[List[Dict[str, Any]]] = None) -> None:
        """Process a completed turn — extract entities (LLM+regex), link them, persist.

        Only processes primary agent turns (not subagents/cron) to avoid
        polluting the knowledge graph with non-conversational content.
        """
        if self._agent_context not in ("primary", ""):
            return
        if not self._graph or not self._store:
            return

        self._turn_counter += 1

        # 1. Extract entities from both user and assistant content
        #    Returns entity IDs so we can link them
        user_ids = self._extract_and_add_entities(user_content)
        asst_ids = self._extract_and_add_entities(assistant_content)

        # 2. Auto-link: create co_occur edges between all entities in the same turn
        all_ids: Set[str] = set(user_ids + asst_ids)
        if len(all_ids) >= 2:
            self._auto_link_entities(list(all_ids))
            logger.info("sync_turn #%d: %d user + %d asst entities, %d unique → auto-linked",
                         self._turn_counter, len(user_ids), len(asst_ids), len(all_ids))
        elif all_ids:
            logger.info("sync_turn #%d: %d entities (single, nothing to link)",
                         self._turn_counter, len(all_ids))

        # 3. Collect entity labels for downstream use
        entity_labels: List[str] = []
        if self._graph:
            for eid in all_ids:
                ent = self._graph.get_entity(eid)
                if ent:
                    entity_labels.append(ent.label)
        # Remove duplicates while preserving order
        seen_labels: Set[str] = set()
        unique_labels: List[str] = []
        for lbl in entity_labels:
            if lbl.lower() not in seen_labels:
                seen_labels.add(lbl.lower())
                unique_labels.append(lbl)

        # 4. Persist turn to L3
        user_summary = self._summarize_content(user_content)
        asst_summary = self._summarize_content(assistant_content)
        self._store.persist_turn(self._session_id, self._turn_counter,
                                  "user", user_content, user_summary)
        self._store.persist_turn(self._session_id, self._turn_counter,
                                  "assistant", assistant_content, asst_summary)

        # 5. Push to GBrain (Phase 2.1 — fire-and-forget)
        self._push_turn_to_gbrain(
            turn_num=self._turn_counter,
            user_summary=user_summary,
            asst_summary=asst_summary,
            entity_labels=unique_labels,
        )

        # 6. Periodic session summary update
        if self._turn_counter % _MAX_TURNS_BEFORE_SUMMARY == 0:
            self._update_session_summary()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """No custom tools — context-only provider."""
        return []

    def shutdown(self) -> None:
        """Clean shutdown — persist L2 to L3, close store."""
        if self._store:
            self._store.shutdown()
        self._initialized = False
        logger.info("GlidingCache shut down")

    # -- Hooks -------------------------------------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Finalize session — update summary, release coherence."""
        self._update_session_summary()
        if self._coherence:
            self._coherence.release_session(self._session_id)

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mirror built-in memory writes to the Gliding graph.

        When the agent writes to built-in memory, we extract entities
        from the content and add them to our graph + store.
        """
        if not self._graph or not self._store:
            return
        if action == "add" and content:
            self._extract_and_add_entities(content)
            # Also persist the memory as an entity
            label = content.split("\n")[0][:60]
            eid = self._graph.add_entity(label, "memory_entry",
                                          {"source": "memory_write", "target": target})
            if eid:
                self._store.persist_entity(eid, label, "memory_entry",
                                            {"source": "memory_write", "target": target})

    # ---------------------------------------------------------------------------
    # LLM NER — New in v0.2.0
    # ---------------------------------------------------------------------------

    def _load_llm_config(self) -> None:
        """Read Hermes config.yaml to get the LLM endpoint for NER calls.

        Uses the same model/provider that the agent is configured with,
        so NER quality matches the model in use.
        """
        config_path = os.path.expanduser("~/.hermes/config.yaml")
        if not os.path.exists(config_path):
            logger.warning("NER config: config.yaml not found at %s", config_path)
            return

        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)

            model_cfg = cfg.get("model", {})
            base_url = model_cfg.get("base_url", "")
            api_key = model_cfg.get("api_key", "")
            model = model_cfg.get("default", "")

            if not base_url or not api_key or not model:
                logger.warning("NER config: incomplete model config (url=%s, model=%s)",
                               bool(base_url), bool(model))
                return

            self._llm_base_url = base_url.rstrip("/")
            self._llm_api_key = api_key
            self._llm_model = model
            self._llm_available = True
            logger.info("NER LLM configured: model=%s, url=%s",
                        model, self._llm_base_url)

        except Exception as e:
            logger.warning("NER config load failed: %s", e)

    def _extract_entities_llm(self, content: str) -> Optional[List[str]]:
        """Attempt LLM-based entity extraction with caching and rate limiting.

        Returns None to signal "fall back to regex". Returns a list
        (possibly empty) on success.
        """
        if not self._llm_available:
            return None
        if len(content) < _MIN_LLM_NER_LENGTH:
            return None

        # 1. Cache check — same content hash → return cached entities
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:16]
        cached = self._ner_cache.get(content_hash)
        if cached:
            elapsed = time.time() - cached[0]
            if elapsed < _NER_CACHE_TTL:
                return cached[1]

        # 2. Rate limit — at most one LLM call every _LLM_RATE_LIMIT_SECS
        now = time.time()
        if now - self._last_llm_call < _LLM_RATE_LIMIT_SECS:
            return None
        self._last_llm_call = now

        # 3. Make the API call
        return self._call_llm_for_entities(content, content_hash)

    def _call_llm_for_entities(self, content: str,
                                content_hash: str) -> Optional[List[str]]:
        """Raw LLM API call for entity extraction."""
        url = f"{self._llm_base_url}/chat/completions"

        # Truncate content to avoid token blow-up
        truncated = content[:2000]

        payload = json.dumps({
            "model": self._llm_model,
            "messages": [
                {"role": "system", "content": _LLM_NER_PROMPT},
                {"role": "user", "content": truncated},
            ],
            "max_tokens": 400,
            "temperature": 0.05,
        }).encode("utf-8")

        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._llm_api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=_LLM_NER_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                raw_text = body["choices"][0]["message"]["content"].strip()

            # Parse response — could be JSON array or JSON object with entities key
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                raw_entities: List[str] = [str(e).strip() for e in parsed if e]
            elif isinstance(parsed, dict):
                for key in ("entities", "names", "result", "data"):
                    val = parsed.get(key)
                    if isinstance(val, list):
                        raw_entities = [str(e).strip() for e in val if e]
                        break
                else:
                    raw_entities = []
            else:
                raw_entities = []

            # Apply quality filters
            entities = self._filter_llm_entities(raw_entities)

            # Cache result (even empty — avoid re-calling for content with no entities)
            self._ner_cache[content_hash] = (time.time(), entities)

            logger.debug("LLM NER: %d entities (filtered from %d) from %d chars",
                         len(entities), len(raw_entities), len(truncated))
            return entities

        except (urllib.error.URLError, urllib.error.HTTPError,
                json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
            logger.debug("LLM NER failed for %s...: %s", truncated[:80], e)
            return None

    @staticmethod
    def _filter_llm_entities(raw_entities: List[str]) -> List[str]:
        """Clean and deduplicate LLM-extracted entities.

        Rules:
        1. Remove stoplist words
        2. Remove base64-like strings
        3. Remove pure numbers / percentages
        4. Remove function names (foo(), foo_bar)
        5. Remove too-short (< 3 chars) or too-long (> 50 chars) labels
        6. Deduplicate case-insensitively, preferring the longer form
        """
        cleaned: Dict[str, str] = {}  # lower -> original

        for label in raw_entities:
            label = label.strip().strip('"').strip("'")

            # Length check
            if len(label) < _NER_MIN_LABEL_LEN or len(label) > _NER_MAX_LABEL_LEN:
                continue

            lower = label.lower()

            # Stoplist
            if lower in _NER_STOPLIST:
                continue

            # Base64-like pattern
            if _BASE64_PATTERN.match(label):
                continue

            # Pure numeric
            if _NUMERIC_LIKE.match(label):
                continue

            # Function-like names
            if _CODEPATH_LIKE.match(label):
                continue

            # Deduplicate: keep longest form for case-insensitive duplicates
            existing = cleaned.get(lower)
            if existing is None or len(label) > len(existing):
                cleaned[lower] = label

        return list(cleaned.values())

    # ---------------------------------------------------------------------------
    # Auto-linking — New in v0.2.0
    # ---------------------------------------------------------------------------

    def _auto_link_entities(self, entity_ids: List[str]) -> None:
        """Create edges between co-occurring entities and infer transitive links.

        1. Bidirectional co_occur edges between all pairs extracted in the same turn
        2. Simple transitive inference: if A→B (new) and B→C (existing), infer A→C

        Capped to avoid O(n²) explosion on large entity sets.
        """
        ids = entity_ids[:]
        n = len(ids)

        # 1. Direct co-occurrence edges (bidirectional)
        pairs_created = 0
        max_pairs = _MAX_AUTO_LINK_ENTITIES
        for i in range(n):
            for j in range(i + 1, n):
                if pairs_created >= max_pairs:
                    break
                a, b = ids[i], ids[j]
                self._graph.add_edge(a, _CO_OCCUR_PREDICATE, b)
                self._graph.add_edge(b, _CO_OCCUR_PREDICATE, a)
                self._store.persist_triple(a, _CO_OCCUR_PREDICATE, b)
                self._store.persist_triple(b, _CO_OCCUR_PREDICATE, a)
                pairs_created += 1
            if pairs_created >= max_pairs:
                break

        # 2. Simple transitive inference (1-hop):
        #    For each pair (A, C) in this batch, if A and C share
        #    a common neighbor B (existing), add A→C with _INFERRED_PREDICATE
        if pairs_created > 0:
            self._infer_transitive_links(ids)

        logger.info("auto_link: %d entities → %d direct edge pairs",
                     len(ids), pairs_created)

    def _infer_transitive_links(self, entity_ids: List[str]) -> None:
        """Check if any pair of current entities shares a graph neighbor.

        If A and C are both linked to B (via co_occur edges), infer A→C.
        Uses the existing graph's edge data.
        """
        ids_set = set(entity_ids)
        inferred = 0

        for i in range(len(entity_ids)):
            a = entity_ids[i]
            # Get A's one-hop neighbors from L2
            neighbors_a = self._graph.get_neighbors(a, max_hops=1)
            for c in entity_ids[i + 1:]:
                if c in ids_set and c in neighbors_a:
                    # Already directly linked — skip
                    continue
                # Check if A and C share a common neighbor
                neighbors_c = self._graph.get_neighbors(c, max_hops=1)
                common = set(neighbors_a.keys()) & set(neighbors_c.keys())
                if common:
                    # Transitive relation found — add inferred edge
                    weight = 0.5  # lower weight than direct co_occur
                    self._graph.add_edge(a, _INFERRED_PREDICATE, c)
                    self._graph.add_edge(c, _INFERRED_PREDICATE, a)
                    self._store.persist_triple(a, _INFERRED_PREDICATE, c, weight)
                    self._store.persist_triple(c, _INFERRED_PREDICATE, a, weight)
                    inferred += 1

        if inferred:
            logger.debug("Auto-linked: %d direct pairs, %d transitive links",
                         len(entity_ids) * (len(entity_ids) - 1) // 2, inferred)

    # -----------------------------------------------------------------------
    # GBrain integration (Phase 2.1)
    # -----------------------------------------------------------------------

    def _detect_gbrain(self) -> None:
        """Check if GBrain CLI is available."""
        if not os.path.isfile(_GBRAIN_BIN):
            logger.debug("GBrain: binary not found at %s", _GBRAIN_BIN)
            return
        try:
            result = subprocess.run(
                [_GBRAIN_BIN, "--version"],
                capture_output=True, text=True, timeout=5,
                env=self._gbrain_env(),
            )
            if result.returncode == 0:
                self._gbrain_available = True
                logger.info("GBrain integration ready: %s", result.stdout.strip())
            else:
                logger.debug("GBrain binary error: %s", result.stderr[:100])
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("GBrain detection failed: %s", e)

    @staticmethod
    def _gbrain_env() -> Dict[str, str]:
        """Build env with gbrain bin dir and bun in PATH."""
        gbrain_dir = os.path.dirname(_GBRAIN_BIN)
        bun_dir = "/usr/local/nodejs/bin"
        env = dict(os.environ)
        env["PATH"] = f"{gbrain_dir}:{bun_dir}:{env.get('PATH', '')}"
        return env

    def _push_turn_to_gbrain(self, turn_num: int, user_summary: str,
                              asst_summary: str, entity_labels: List[str]) -> None:
        """Push a structured turn summary to GBrain (fire-and-forget).

        Builds a compact markdown note and calls gbrain capture --stdin.
        Non-blocking: logs failures but never raises.
        """
        if not self._gbrain_available:
            return

        slug = f"{_GBRAIN_SLUG_PREFIX}{self._session_id[:8]}_{turn_num}"

        # Build structured content
        entities_str = ", ".join(entity_labels[:20])
        content_lines = [
            f"## Turn #{turn_num}",
            f"**User**: {user_summary or '(empty)'}",
            f"**Assistant**: {asst_summary or '(empty)'}",
        ]
        if entities_str:
            content_lines.append(f"**Entities**: {entities_str}")
        content = "\n\n".join(content_lines)

        try:
            result = subprocess.run(
                [_GBRAIN_BIN, "capture", "--stdin", "--slug", slug,
                 "--type", "note", "--quiet"],
                input=content.encode("utf-8"),
                capture_output=True, timeout=_GBRAIN_CAPTURE_TIMEOUT,
                env=self._gbrain_env(),
            )
            if result.returncode == 0:
                slug_out = result.stdout.decode("utf-8").strip()
                logger.info("GBrain pushed turn #%d → %s", turn_num, slug_out)
            else:
                logger.debug("GBrain push turn #%d failed: %s",
                             turn_num, result.stderr.decode("utf-8", errors="replace")[:200])
        except subprocess.TimeoutExpired:
            logger.debug("GBrain push turn #%d timed out", turn_num)
        except OSError as e:
            logger.debug("GBrain push turn #%d error: %s", turn_num, e)

    # -----------------------------------------------------------------------
    # Ollama embedding for semantic entity dedup (Phase 2.1)
    # -----------------------------------------------------------------------

    def _embed_label(self, label: str) -> Optional[List[float]]:
        """Get embedding vector from Ollama for a label.

        Returns None if Ollama is unavailable or returns an error.
        """
        if not _EMBED_ENABLED:
            return None

        # Check cache first
        now = time.time()
        cached = self._embed_cache.get(label)
        if cached and (now - cached[0]) < _EMBEDDING_CACHE_TTL:
            return cached[1]

        payload = json.dumps({
            "model": _EMBEDDING_MODEL,
            "prompt": label,
        }).encode("utf-8")

        req = urllib.request.Request(
            _EMBEDDING_URL, data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            vec: Optional[List[float]] = body.get("embedding")
            if vec:
                self._embed_cache[label] = (now, vec)
                self._embed_available = True
            return vec
        except (urllib.error.URLError, urllib.error.HTTPError,
                json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
            logger.debug("Embedding failed for '%s': %s", label[:20], e)
            return None

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        denom = na * nb
        return dot / denom if denom > 0 else 0.0

    def _warm_embed_cache(self) -> None:
        """Pre-warm embedding cache from existing graph entities."""
        if not self._graph or not self._store or not _EMBED_ENABLED:
            return
        # Load ALL entities from store (not just snapshot top 20)
        all_entities = self._store.load_all_entities()
        labels = [
            e["label"] for e in all_entities
            if len(e["label"]) >= _EMBED_LABEL_MIN_LEN
        ]
        if not labels:
            logger.debug("Embedding warm: no entities to warm (%d total)", len(all_entities))
            return

        # Embed in batches to avoid overloading Ollama
        warmed = 0
        for label in labels:
            if label not in self._embed_cache:
                vec = self._embed_label(label)
                if vec:
                    warmed += 1
                    if warmed >= 20:
                        break

        if warmed:
            logger.info("Embedding cache warmed: %d/%d entities cached (bge-small-zh)",
                         len(self._embed_cache), len(labels))

    def _is_duplicate_via_embedding(self, label: str) -> Optional[str]:
        """Check if label is semantically similar to an existing entity.

        Returns the existing label if a match is found, None otherwise.
        Only checks entities with labels >= _EMBED_LABEL_MIN_LEN.
        """
        if len(label) < _EMBED_LABEL_MIN_LEN or not _EMBED_ENABLED:
            return None

        vec = self._embed_label(label)
        if vec is None:
            return None

        # Check against all cached embeddings
        best_label: Optional[str] = None
        best_sim = 0.0
        for cached_label, (_, cached_vec) in self._embed_cache.items():
            if cached_label.lower() == label.lower():
                continue  # exact match — not a dedup concern
            sim = self._cosine_similarity(vec, cached_vec)
            if sim > best_sim and sim >= _EMBEDDING_DEDUP_THRESHOLD:
                best_sim = sim
                best_label = cached_label

        if best_label:
            logger.debug("Embedding dedup: '%s' ≈ '%s' (sim=%.3f)",
                         label, best_label, best_sim)
            return best_label
        return None

    # -- Internal helpers --------------------------------------------------

    def _extract_and_add_entities(self, content: str) -> List[str]:
        """Parse content for entity mentions and add them to L2 + L3.

        Strategy (try in order):
          1. LLM-driven NER (with rate limit + cache)
          2. Regex fallback (quoted / topic: / CamelCase patterns)

        Returns a list of entity IDs that were added or found.
        """
        if not content or len(content) < _MIN_CONTENT_FOR_ENTITY:
            return []

        entity_ids: List[str] = []

        # --- Phase 1: LLM NER (with embedding dedup) ---
        llm_labels = self._extract_entities_llm(content)
        if llm_labels is not None:
            # LLM succeeded — use its results
            for label in llm_labels:
                # Embedding-based dedup: skip if semantically similar entity exists
                dup = self._is_duplicate_via_embedding(label)
                if dup:
                    # Find the existing entity ID for the duplicate
                    existing = self._graph.find_entity(dup)
                    if existing:
                        entity_ids.append(existing.id)
                    continue
                eid = self._graph.add_entity(label, "topic")
                if eid:
                    entity_ids.append(eid)
                    self._store.persist_entity(eid, label, "topic")
            if entity_ids:
                logger.debug("Extracted %d entities (LLM + embed dedup)", len(entity_ids))
                return entity_ids

        # --- Phase 2: Regex fallback ---
        for match in _ENTITY_PATTERN.finditer(content):
            label = match.group(1).strip()
            if len(label) >= 2:
                eid = self._graph.add_entity(label, "topic")
                if eid:
                    entity_ids.append(eid)
                    self._store.persist_entity(eid, label, "topic")

        for match in _TOPIC_PATTERN.finditer(content):
            label = match.group(1).strip()
            if len(label) >= 2:
                eid = self._graph.add_entity(label, "topic")
                if eid:
                    entity_ids.append(eid)
                    self._store.persist_entity(eid, label, "topic")

        # Extract obvious named entities (CamelCase words > 4 chars)
        words = set(re.findall(r'\b[A-Z][a-z]{2,}(?:[A-Z][a-z]{2,})+\b', content))
        for word in words:
            if len(word) >= 4:
                eid = self._graph.add_entity(word, "concept")
                if eid:
                    entity_ids.append(eid)
                    self._store.persist_entity(eid, word, "concept")

        return entity_ids

    def _summarize_content(self, content: str) -> str:
        """Simple extractive summarization — first meaningful sentence."""
        if not content:
            return ""
        # Take first non-empty, non-short line
        for line in content.split("\n"):
            stripped = line.strip()
            if len(stripped) > 20 and not stripped.startswith(("```", "#", "-", "*", ">")):
                return stripped[:150]
        return content[:150]

    def _update_session_summary(self) -> None:
        """Accumulate session summary from recent turns."""
        if not self._store:
            return
        existing = self._store.get_session_summary(self._session_id)
        if existing:
            # Extend existing summary
            new = f"{existing} (+{self._turn_counter} turns)"
            self._store.update_session_summary(self._session_id, new)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Register the GlidingCache memory provider.

    Called during plugin discovery by _load_provider_from_dir().
    """
    ctx.register_memory_provider(GlidingCacheMemoryProvider())