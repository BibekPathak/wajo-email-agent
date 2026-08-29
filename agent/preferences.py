"""User preference memory backed by SQLite.

The preference model is deliberately lightweight: explicit preference
statistics (positive/negative feedback counts) rather than full RL. Learned
preferences can only influence decisions the hard safety floor allows; they
can never weaken it.

Every preference is keyed by ``(user_id, action_type, sender_category,
context)`` and records how much evidence exists that the user wants a
particular level of autonomy.

Evidence model
--------------

* New preferences start with no evidence and near-zero confidence.
* Repeated consistent feedback increases confidence.
* Conflicting feedback reduces confidence.
* Explicit feedback is weighted more strongly than inferred feedback, and
  explicit *negative* feedback is weighted hardest so that autonomy drops
  quickly when the user pushes back.
* The stored ``preferred_decision`` tracks the direction of the evidence:
  more positive signals lean autonomous, more negative signals lean ASK.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Optional

from .models import AutonomyDecision, Preference, PreferenceKey

# Explicit positive feedback counts for 1.0; inferred positive counts for less.
INFERRED_WEIGHT = 0.5
# Negative feedback reduces autonomy quickly, so explicit negatives count more.
EXPLICIT_NEGATIVE_WEIGHT = 2.0
# Total (weighted) evidence that roughly doubles confidence.
EVIDENCE_SCALE = 3.0
CONFIDENCE_EPSILON = 1e-9


def effective_counts(pref: Preference) -> tuple[float, float]:
    """Weighted positive and negative evidence for a preference.

    Explicit feedback is stronger than inferred feedback, and explicit
    negative feedback is the strongest signal of all.
    """
    eff_pos = pref.explicit_positive_count + (
        pref.positive_count - pref.explicit_positive_count
    ) * INFERRED_WEIGHT
    eff_neg = pref.explicit_negative_count * EXPLICIT_NEGATIVE_WEIGHT + (
        pref.negative_count - pref.explicit_negative_count
    ) * 1.0
    return eff_pos, eff_neg


def compute_confidence(pref: Preference) -> float:
    """Confidence that ``pref.preferred_decision`` matches the user's wish.

    ``confidence = agreement * strength`` where:

    * ``agreement`` is the fraction of weighted evidence supporting the stored
      decision (autonomous decisions are supported by positive evidence; ASK
      is supported by negative evidence),
    * ``strength`` grows from 0 to 1 as weighted evidence accumulates.

    The number is deliberately an evidence-derived score, not a calibrated
    probability; the evaluation harness measures calibration separately.
    """
    if pref.preferred_decision is None:
        return 0.0
    eff_pos, eff_neg = effective_counts(pref)
    total = eff_pos + eff_neg
    if total <= CONFIDENCE_EPSILON:
        return 0.0
    if pref.preferred_decision.is_autonomous:
        agreement = eff_pos / total
    else:
        agreement = eff_neg / total
    strength = 1 - 0.5 ** (total / EVIDENCE_SCALE)
    return agreement * strength


def derive_preferred_decision(
    pref: Preference,
    default_decision: Optional[AutonomyDecision] = None,
) -> Optional[AutonomyDecision]:
    """Choose the decision the current evidence supports.

    Negative evidence pushes toward ASK (the user wants to be consulted).
    Positive evidence pushes toward an autonomous decision: keep an existing
    autonomous preference, fall back to the given default when it is already
    autonomous (e.g. SILENT for newsletters), otherwise ACT_NOTIFY.
    """
    eff_pos, eff_neg = effective_counts(pref)
    if eff_neg > eff_pos:
        return AutonomyDecision.ASK
    if eff_pos > 0:
        if pref.preferred_decision is not None and pref.preferred_decision.is_autonomous:
            return pref.preferred_decision
        if default_decision is not None and default_decision.is_autonomous:
            return default_decision
        return AutonomyDecision.ACT_NOTIFY
    return None


class PreferenceStore:
    """SQLite-backed per-user preference memory.

    Uses a single connection guarded by a lock. ``db_path=":memory:"`` keeps a
    fresh, isolated in-memory database per store (used heavily in tests).
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- schema ------------------------------------------------------------

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    user_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    sender_category TEXT NOT NULL DEFAULT '*',
                    context TEXT NOT NULL DEFAULT '',
                    preferred_decision TEXT,
                    positive_count INTEGER NOT NULL DEFAULT 0,
                    negative_count INTEGER NOT NULL DEFAULT 0,
                    explicit_positive_count INTEGER NOT NULL DEFAULT 0,
                    explicit_negative_count INTEGER NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, action_type, sender_category, context)
                )
                """
            )

    # --- lookups ------------------------------------------------------------

    def get(self, user_id: str, key: PreferenceKey) -> Optional[Preference]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM preferences
                WHERE user_id = ? AND action_type = ?
                  AND sender_category = ? AND context = ?
                """,
                (user_id, key.action_type, key.sender_category, key.context),
            ).fetchone()
        return self._row_to_preference(row) if row is not None else None

    def lookup(
        self,
        user_id: str,
        action_type: str,
        sender_category: str,
        context: str = "",
    ) -> Optional[Preference]:
        """Best-match lookup: exact sender, then wildcard sender category.

        Falls back to a context-free wildcard so a preference learned for one
        sender still generalizes sensibly.
        """
        candidates = [
            (sender_category, context),
            ("*", context),
            ("*", ""),
        ]
        for cat, ctx in candidates:
            found = self.get(
                user_id,
                PreferenceKey(action_type=action_type, sender_category=cat, context=ctx),
            )
            if found is not None:
                return found
        return None

    def get_all(self, user_id: str) -> list[Preference]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM preferences
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [self._row_to_preference(r) for r in rows]

    # --- writes -------------------------------------------------------------

    def upsert(self, user_id: str, preference: Preference) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO preferences (
                    user_id, action_type, sender_category, context,
                    preferred_decision, positive_count, negative_count,
                    explicit_positive_count, explicit_negative_count, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, action_type, sender_category, context)
                DO UPDATE SET
                    preferred_decision = excluded.preferred_decision,
                    positive_count = excluded.positive_count,
                    negative_count = excluded.negative_count,
                    explicit_positive_count = excluded.explicit_positive_count,
                    explicit_negative_count = excluded.explicit_negative_count,
                    confidence = excluded.confidence,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    user_id,
                    preference.key.action_type,
                    preference.key.sender_category,
                    preference.key.context,
                    preference.preferred_decision.value
                    if preference.preferred_decision is not None
                    else None,
                    preference.positive_count,
                    preference.negative_count,
                    preference.explicit_positive_count,
                    preference.explicit_negative_count,
                    preference.confidence,
                ),
            )

    def record_signal(
        self,
        user_id: str,
        key: PreferenceKey,
        *,
        positive: bool,
        explicit: bool = True,
        default_decision: Optional[AutonomyDecision] = None,
    ) -> Preference:
        """Record one piece of feedback and return the updated preference.

        ``positive=True`` means the user was happy with autonomous handling;
        ``positive=False`` means the user wanted to be consulted.
        """
        with self._lock, self._conn:
            preference = self.get_locked(user_id, key) or Preference(key=key)
            if positive:
                preference.positive_count += 1
                if explicit:
                    preference.explicit_positive_count += 1
            else:
                preference.negative_count += 1
                if explicit:
                    preference.explicit_negative_count += 1
            preference.preferred_decision = derive_preferred_decision(
                preference, default_decision
            )
            preference.confidence = compute_confidence(preference)
            self.upsert_locked(user_id, preference)
        return preference

    def delete_all(self, user_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM preferences WHERE user_id = ?",
                (user_id,),
            )

    # --- internals (lock must already be held) --------------------------------

    def get_locked(self, user_id: str, key: PreferenceKey) -> Optional[Preference]:
        row = self._conn.execute(
            """
            SELECT * FROM preferences
            WHERE user_id = ? AND action_type = ?
              AND sender_category = ? AND context = ?
            """,
            (user_id, key.action_type, key.sender_category, key.context),
        ).fetchone()
        return self._row_to_preference(row) if row is not None else None

    def upsert_locked(self, user_id: str, preference: Preference) -> None:
        self._conn.execute(
            """
            INSERT INTO preferences (
                user_id, action_type, sender_category, context,
                preferred_decision, positive_count, negative_count,
                explicit_positive_count, explicit_negative_count, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, action_type, sender_category, context)
            DO UPDATE SET
                preferred_decision = excluded.preferred_decision,
                positive_count = excluded.positive_count,
                negative_count = excluded.negative_count,
                explicit_positive_count = excluded.explicit_positive_count,
                explicit_negative_count = excluded.explicit_negative_count,
                confidence = excluded.confidence,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                user_id,
                preference.key.action_type,
                preference.key.sender_category,
                preference.key.context,
                preference.preferred_decision.value
                if preference.preferred_decision is not None
                else None,
                preference.positive_count,
                preference.negative_count,
                preference.explicit_positive_count,
                preference.explicit_negative_count,
                preference.confidence,
            ),
        )

    @staticmethod
    def _row_to_preference(row: sqlite3.Row) -> Preference:
        return Preference(
            key=PreferenceKey(
                action_type=row["action_type"],
                sender_category=row["sender_category"],
                context=row["context"],
            ),
            preferred_decision=(
                AutonomyDecision(row["preferred_decision"])
                if row["preferred_decision"] is not None
                else None
            ),
            positive_count=row["positive_count"],
            negative_count=row["negative_count"],
            explicit_positive_count=row["explicit_positive_count"],
            explicit_negative_count=row["explicit_negative_count"],
            confidence=row["confidence"],
        )
