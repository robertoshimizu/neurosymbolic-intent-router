"""Neuro-symbolic action router: Classifier ranking, MiniLM fallback, session policy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from dotenv import load_dotenv

from intent_understanding import split_requests
import policy
from policy import (
    ACTION_CATALOG,
    ACTION_PRECEDENCE,
    NONE_ACTION,
    Outcome,
    ParsedRequest,
    Session,
    evaluate_action,
    parse_request,
)
from transfer import Ledger, WireTransfer, confirm_or_refuse

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Hub download cache (safetensors live here after the first fetch).
MODEL_CACHE_DIR = Path(__file__).resolve(
).parents[1] / ".cache" / "sentence-transformers"
# Persistent embedding vectors so CLI runs can skip loading weights into RAM.
EMBEDDING_CACHE_DIR = Path(__file__).resolve(
).parents[1] / ".cache" / "embeddings"
_MODEL_CACHE: dict[str, object] = {}
_ENV_LOADED = False


def _load_env() -> None:
    """Load project .env so Hub and TypeSafe clients can read their API keys."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path, override=False)
    _ENV_LOADED = True


_load_env()
MIN_SCORE = 0.45
MIN_MARGIN = 0.08

@dataclass(frozen=True)
class DescriptionMatch:
    """MiniLM closeness of the utterance to the action descriptions.

    Jev still chooses the action. This only explains that choice.
    """

    chosen: str
    nearest: str
    nearest_cosine: float
    chosen_cosine: float
    similarities: dict[str, float]

    @property
    def agrees(self) -> bool:
        return self.nearest == self.chosen

    @property
    def description_gap(self) -> bool:
        return (not self.agrees) or self.chosen_cosine < MIN_SCORE


@dataclass(frozen=True)
class Decision:
    action: str | None
    outcome: Outcome
    reason: str
    parsed: ParsedRequest
    scores: dict[str, float]
    permissions: dict[str, bool]
    rule_trace: list[str] = field(default_factory=list)
    suggestion: str | None = None
    source: Literal["jev", "minilm"] = "minilm"
    description_match: DescriptionMatch | None = None
    follow_ups: tuple[str, ...] = ()


def evaluate_permissions(
    session: Session,
    balance: Decimal,
    parsed: ParsedRequest,
    actions: list[str],
) -> tuple[dict[str, bool], dict[str, str]]:
    permissions: dict[str, bool] = {}
    reasons: dict[str, str] = {}
    for action in actions:
        allowed, reason = evaluate_action(action, session, balance, parsed)
        permissions[action] = allowed
        reasons[action] = reason
    return permissions, reasons


def cosine_scores(
    query_vector: np.ndarray, action_embeddings: dict[str, np.ndarray]
) -> dict[str, float]:
    query_norm = query_vector / np.linalg.norm(query_vector)
    scores: dict[str, float] = {}
    for action_id, action_vec in action_embeddings.items():
        action_norm = action_vec / np.linalg.norm(action_vec)
        scores[action_id] = float(np.dot(query_norm, action_norm))
    return scores


def _catalog_fingerprint(model_name: str, catalog: dict[str, str]) -> str:
    import hashlib
    import json

    payload = json.dumps(
        {"model": model_name, "catalog": catalog},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _query_fingerprint(model_name: str, text: str) -> str:
    import hashlib

    return hashlib.sha256(f"{model_name}\0{text}".encode()).hexdigest()


class ActionEmbedder:
    """MiniLM encoder with Hub weight cache and on-disk embedding cache.

    Hugging Face weights already persist under MODEL_CACHE_DIR. Each new CLI
    process still has to map those weights into RAM unless the needed vectors
    are already saved under EMBEDDING_CACHE_DIR — in that case the model is
    never loaded.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        cache_folder: Path | str | None = MODEL_CACHE_DIR,
        embedding_cache_dir: Path | str | None = EMBEDDING_CACHE_DIR,
        use_embedding_cache: bool = True,
    ) -> None:
        self.model_name = model_name
        self.cache_folder = Path(
            cache_folder) if cache_folder else MODEL_CACHE_DIR
        self.embedding_cache_dir = (
            Path(embedding_cache_dir) if embedding_cache_dir else EMBEDDING_CACHE_DIR
        )
        self.use_embedding_cache = use_embedding_cache
        self._model = None
        self._action_embeddings: dict[str, np.ndarray] | None = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        cache_key = f"{self.model_name}|{self.cache_folder.resolve()}"
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            self._model = cached
            return self._model

        _load_env()
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        from sentence_transformers import SentenceTransformer

        print(
            f"Loading MiniLM weights into memory from {self.cache_folder} "
            "(one-time per process; embedding cache avoids this on repeat runs)...",
            flush=True,
        )
        try:
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_folder),
                local_files_only=True,
            )
        except OSError:
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_folder),
                local_files_only=False,
            )
        _MODEL_CACHE[cache_key] = self._model
        return self._model

    def _query_cache_path(self, text: str) -> Path:
        digest = _query_fingerprint(self.model_name, text)
        return self.embedding_cache_dir / "queries" / f"{digest}.npy"

    def _action_cache_paths(
        self, catalog: dict[str, str]
    ) -> tuple[Path, Path]:
        digest = _catalog_fingerprint(self.model_name, catalog)
        base = self.embedding_cache_dir / "actions" / digest
        return base.with_suffix(".npz"), base.with_suffix(".json")

    def embed(self, text: str) -> np.ndarray:
        if self.use_embedding_cache:
            path = self._query_cache_path(text)
            if path.exists():
                return np.asarray(np.load(path), dtype=np.float64)

        vector = np.asarray(
            self._load_model().encode(text, normalize_embeddings=True),
            dtype=np.float64,
        )
        if self.use_embedding_cache:
            path = self._query_cache_path(text)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vector)
        return vector

    def action_embeddings(
        self, catalog: dict[str, str] | None = None
    ) -> dict[str, np.ndarray]:
        catalog = catalog or ACTION_CATALOG
        if self._action_embeddings is not None:
            return self._action_embeddings

        if self.use_embedding_cache:
            npz_path, meta_path = self._action_cache_paths(catalog)
            if npz_path.exists() and meta_path.exists():
                import json

                meta = json.loads(meta_path.read_text())
                if meta.get("model") == self.model_name and set(
                    meta.get("actions", [])
                ) == set(catalog):
                    loaded = np.load(npz_path)
                    self._action_embeddings = {
                        action_id: np.asarray(
                            loaded[action_id], dtype=np.float64)
                        for action_id in catalog
                    }
                    return self._action_embeddings

        self._action_embeddings = {
            action_id: self.embed(description)
            for action_id, description in catalog.items()
        }

        if self.use_embedding_cache:
            import json

            npz_path, meta_path = self._action_cache_paths(catalog)
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(npz_path, **self._action_embeddings)
            meta_path.write_text(
                json.dumps(
                    {
                        "model": self.model_name,
                        "actions": list(catalog.keys()),
                    },
                    indent=2,
                )
            )
        return self._action_embeddings


@dataclass(frozen=True)
class IntentRank:
    """System 1 proposal. Policy has not run yet."""

    action: str
    scores: dict[str, float]
    confidence: float
    source: Literal["jev", "minilm"]
    runner_up: float = 0.0
    request_count: str | None = None


class Classifier(Protocol):
    """Picks catalog actions. Unsure answers come back as `none`; None means unavailable."""

    def classify(self, text: str) -> IntentRank | None: ...

    def label(self, texts: tuple[str, ...]) -> list[IntentRank] | None: ...


def _rank_minilm(
    query_vector: np.ndarray, action_embeddings: dict[str, np.ndarray]
) -> IntentRank:
    scores = cosine_scores(query_vector, action_embeddings)
    ranked = sorted(scores, key=scores.get, reverse=True)
    top_action = ranked[0]
    runner_up = scores[ranked[1]] if len(ranked) > 1 else 0.0
    return IntentRank(
        action=top_action,
        scores=scores,
        confidence=scores[top_action],
        source="minilm",
        runner_up=runner_up,
    )


def _rank_minilm_from_text(utterance: str, embedder: ActionEmbedder | None) -> IntentRank:
    embedder = embedder or ActionEmbedder()
    return _rank_minilm(embedder.embed(utterance), embedder.action_embeddings())


def _select_rank(
    utterance: str,
    *,
    classifier: Classifier | None,
    query_vector: np.ndarray | None,
    action_embeddings: dict[str, np.ndarray] | None,
    embedder: ActionEmbedder | None,
) -> IntentRank:
    """The classifier first. MiniLM when there is none, or it is unavailable."""
    injected = query_vector is not None and action_embeddings is not None

    def minilm() -> IntentRank:
        if injected:
            return _rank_minilm(query_vector, action_embeddings)
        return _rank_minilm_from_text(utterance, embedder)

    if classifier is None:
        return minilm()
    try:
        rank = classifier.classify(utterance)
    except Exception:
        rank = None
    if rank is None:
        print("Classifier unavailable; using MiniLM fallback.", flush=True)
        return minilm()
    return rank


def _explain_with_minilm(
    utterance: str,
    chosen_action: str,
    *,
    classifier: Classifier | None,
    query_vector: np.ndarray | None,
    action_embeddings: dict[str, np.ndarray] | None,
    embedder: ActionEmbedder | None,
) -> DescriptionMatch | None:
    """Cosines for a committed Jev label. Fake classifiers skip the model unless vectors are injected."""
    injected = query_vector is not None and action_embeddings is not None
    if injected:
        similarity = _rank_minilm(query_vector, action_embeddings)
    elif classifier is None:
        similarity = _rank_minilm_from_text(utterance, embedder)
    else:
        return None
    return DescriptionMatch(
        chosen=chosen_action,
        nearest=similarity.action,
        nearest_cosine=similarity.confidence,
        chosen_cosine=float(similarity.scores.get(chosen_action, 0.0)),
        similarities=similarity.scores,
    )


def decide(
    utterance: str,
    session: Session,
    ledger: Ledger,
    *,
    query_vector: np.ndarray | None = None,
    action_embeddings: dict[str, np.ndarray] | None = None,
    embedder: ActionEmbedder | None = None,
    classifier: Classifier | None = None,
    min_score: float = MIN_SCORE,
    min_margin: float = MIN_MARGIN,
    rank: IntentRank | None = None,
) -> Decision:
    """Rank the utterance, judge the top raw intent, and return a Decision.

    A `none` action stops before policy. A denied
    catalog intent stays denied and is never replaced by FAQ. A precomputed
    `rank` skips ranking.
    """
    rank = rank or _select_rank(
        utterance,
        classifier=classifier,
        query_vector=query_vector,
        action_embeddings=action_embeddings,
        embedder=embedder,
    )
    balance = ledger.get_balance(session.account_id)
    scores = rank.scores

    if rank.action == NONE_ACTION:
        return Decision(
            action=NONE_ACTION,
            outcome="deny",
            reason="no matching action",
            parsed=parse_request(utterance, session.payee_allowlist),
            scores=scores,
            permissions={},
            rule_trace=[
                f"source={rank.source} action={rank.action} confidence={rank.confidence:.4f}",
                "abstain=before_policy",
            ],
            source=rank.source,
        )

    top_action = rank.action
    description_match = None
    if rank.source == "jev" and top_action in ACTION_CATALOG:
        description_match = _explain_with_minilm(
            utterance,
            top_action,
            classifier=classifier,
            query_vector=query_vector,
            action_embeddings=action_embeddings,
            embedder=embedder,
        )

    catalog_actions = [action for action in scores if action in ACTION_CATALOG]
    if top_action not in ACTION_CATALOG:
        catalog_actions = list(ACTION_CATALOG)
    verdict = policy.decide(utterance, top_action, session, ledger)
    parsed = verdict.parsed
    policy_reason = verdict.reason
    permissions, _reasons = evaluate_permissions(
        session, balance, parsed, catalog_actions or [top_action]
    )

    rule_trace = [
        f"source={rank.source}",
        f"top_raw_intent={top_action} confidence={rank.confidence:.4f}",
        f"runner_up_score={rank.runner_up:.4f}",
        f"balance={balance}",
        f"parsed_amount={parsed.amount}",
        f"parsed_payee={parsed.payee}",
        f"policy[{top_action}]={policy_reason}",
    ]
    if description_match is not None:
        rule_trace.append(
            f"minilm_nearest={description_match.nearest} "
            f"cosine={description_match.nearest_cosine:.4f}"
        )
        rule_trace.append(
            f"minilm_chosen_cosine={description_match.chosen_cosine:.4f}"
        )
        if description_match.agrees and not description_match.description_gap:
            rule_trace.append("minilm_agrees=yes")
        elif not description_match.agrees:
            rule_trace.append("description_gap=nearest_differs")
        else:
            rule_trace.append("description_gap=chosen_cosine_below_min_score")

    def _decision(
        outcome: Outcome,
        reason: str,
        *,
        suggestion: str | None = None,
        trace_extra: list[str] | None = None,
    ) -> Decision:
        return Decision(
            action=top_action,
            outcome=outcome,
            reason=reason,
            parsed=parsed,
            scores=scores,
            permissions=permissions,
            rule_trace=rule_trace if not trace_extra else rule_trace + trace_extra,
            suggestion=suggestion,
            source=rank.source,
            description_match=description_match,
        )

    if verdict.outcome == "deny":
        return _decision("deny", policy_reason, suggestion=verdict.suggestion)

    if rank.source == "minilm" and rank.confidence < min_score:
        return _decision(
            "deny",
            "similarity below minimum confidence",
            trace_extra=["confidence=below_min_score"],
        )

    if (
        rank.source == "minilm"
        and len(rank.scores) > 1
        and (rank.confidence - rank.runner_up) < min_margin
    ):
        return _decision(
            "deny",
            "ambiguous intent: margin below minimum",
            trace_extra=["confidence=below_min_margin"],
        )

    if verdict.outcome == "needs_confirmation":
        return _decision(
            "needs_confirmation",
            policy_reason,
            trace_extra=["high_stakes=confirmation_required"],
        )

    return _decision("execute", policy_reason)


Splitter = Callable[[str], tuple[str, ...]]


def _deny_split(utterance: str, session: Session, reason: str, trace: list[str]) -> Decision:
    return Decision(
        action=NONE_ACTION,
        outcome="deny",
        reason=reason,
        parsed=parse_request(utterance, session.payee_allowlist),
        scores={},
        permissions={},
        rule_trace=["request_count=several", *trace],
    )


def route(
    utterance: str,
    session: Session,
    ledger: Ledger,
    *,
    splitter: Splitter = split_requests,
    **decide_kwargs,
) -> Decision:
    """One classifier call. Several requests: MedGemma splits, the classifier labels them in one call,
    the rule orders, policy decides the first, and the rest are listed.
    Never mutates the ledger.
    """
    classifier = decide_kwargs.get("classifier")
    rank = _select_rank(
        utterance,
        classifier=classifier,
        query_vector=decide_kwargs.get("query_vector"),
        action_embeddings=decide_kwargs.get("action_embeddings"),
        embedder=decide_kwargs.get("embedder"),
    )
    if rank.request_count != "several":
        return decide(utterance, session, ledger, rank=rank, **decide_kwargs)

    try:
        requests = splitter(utterance)
    except Exception:
        requests = ()
    if len(requests) < 2:
        return _deny_split(utterance, session, "several requests could not be separated", [f"split_count={len(requests)}"])

    ranks = classifier.label(requests) if classifier is not None else None
    if ranks is None or len(ranks) != len(requests):
        return _deny_split(utterance, session, "several requests could not be labeled", [f"split_count={len(requests)}"])

    labels = [r.action for r in ranks]
    ordered = sorted(range(len(requests)), key=lambda i: ACTION_PRECEDENCE.get(
        labels[i], len(ACTION_PRECEDENCE)))
    head = ordered[0]
    first = decide(requests[head], session, ledger,
                   rank=ranks[head], **decide_kwargs)
    trace = [
        "request_count=several",
        *(f"split[{i}]={requests[i]!r} label={labels[i]} confidence={ranks[i].confidence:.4f}" for i in range(len(requests))),
        f"logical_order={[labels[i] for i in ordered]}",
    ]
    follow_ups = () if first.outcome == "deny" else tuple(
        requests[i] for i in ordered[1:])
    return replace(first, rule_trace=trace + first.rule_trace, follow_ups=follow_ups)


def print_decision(utterance: str, session: Session, decision: Decision) -> None:
    print(f"\nUser Query: '{utterance}'")
    print(
        f"Context: Auth={session.is_authenticated}, Role='{session.role}', "
        f"Status='{session.status}', Account='{session.account_id}'"
    )
    score_label = "Jev Probability" if decision.source == "jev" else "Raw Similarity"
    show_similarity = decision.description_match is not None
    print(f"\nRanker: {decision.source}")
    print("\nAction Decision Matrix:")
    if show_similarity:
        print(
            f"{'Candidate Action':<24} | {'Jev Probability':<16} | "
            f"{'Raw Similarity':<15} | {'Allowed?':<8}"
        )
        print("-" * 74)
    else:
        print(
            f"{'Candidate Action':<24} | {score_label:<16} | {'Allowed?':<8}"
        )
        print("-" * 56)
    for act, score in sorted(
        decision.scores.items(), key=lambda item: item[1], reverse=True
    ):
        if not decision.permissions:
            allowed_str = "—"
        else:
            allowed_str = "YES" if decision.permissions.get(
                act, False) else "NO"
        if show_similarity:
            cosine = decision.description_match.similarities.get(act)
            cosine_str = f"{cosine:.4f}" if cosine is not None else "—"
            print(
                f"{act:<24} | {score:<16.4f} | {cosine_str:<15} | {allowed_str:<8}"
            )
        else:
            print(f"{act:<24} | {score:<16.4f} | {allowed_str:<8}")

    print(
        f"\nDecision: action={decision.action} outcome={decision.outcome} "
        f"reason={decision.reason}"
    )
    if decision.description_match is not None:
        match = decision.description_match
        if match.agrees and not match.description_gap:
            print(
                f"MiniLM agrees: {match.nearest} cosine={match.chosen_cosine:.4f}"
            )
        else:
            print(
                "Description gap: "
                f"MiniLM nearest={match.nearest} ({match.nearest_cosine:.4f}); "
                f"Jev action cosine={match.chosen_cosine:.4f}"
            )
    if decision.follow_ups:
        print("You also asked (ask again to proceed):")
        for follow_up in decision.follow_ups:
            print(f"  - {follow_up}")
    if decision.suggestion:
        print(f"Suggestion: {decision.suggestion}")
    print()


def build_demo_world() -> tuple[dict[str, Session], Ledger]:
    ledger = Ledger()
    ledger.set_balance("acct-guest", Decimal("0"))
    ledger.set_balance("acct-zero", Decimal("0"))
    ledger.set_balance("acct-funded", Decimal("10000"))

    sessions = {
        "guest": Session(
            session_id="guest",
            is_authenticated=False,
            role="guest",
            status="inactive",
            account_id="acct-guest",
            payee_allowlist=(),
        ),
        "zero": Session(
            session_id="zero",
            is_authenticated=True,
            role="customer",
            status="active",
            account_id="acct-zero",
            payee_allowlist=("external bank account",),
        ),
        "funded": Session(
            session_id="funded",
            is_authenticated=True,
            role="customer",
            status="active",
            account_id="acct-funded",
            payee_allowlist=("external bank account",),
        ),
    }
    return sessions, ledger


if __name__ == "__main__":
    from jev import JevClassifier

    utterance = "Close this account and send $500 to my external bank account."
    sessions, ledger = build_demo_world()
    embedder = ActionEmbedder()
    classifier = JevClassifier()

    for key in ("guest", "zero", "funded"):
        session = sessions[key]
        decision = route(utterance, session, ledger,
                         classifier=classifier, embedder=embedder)
        balance = ledger.get_balance(session.account_id)
        print(f"\n--- Session '{key}' (balance=${balance}) ---")
        print_decision(utterance, session, decision)

        if decision.outcome == "needs_confirmation" and decision.action == (
            "wire_transfer_funds"
        ):
            assert decision.parsed.amount is not None
            transfer = WireTransfer(
                ledger=ledger,
                account_id=session.account_id,
                amount=decision.parsed.amount,
                transfer_id="wire-demo-1",
            )
            transfer.send("request_confirmation")
            confirmed = confirm_or_refuse(transfer)
            print(f"Confirmation accepted: {confirmed}")
            if confirmed:
                transfer.send("submit")
                transfer.send("settle")
                print(
                    f"Settled. New balance: ${ledger.get_balance(session.account_id)}"
                )
