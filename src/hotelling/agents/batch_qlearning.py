"""Vectorized batch Q-learning.

Holds N INDEPENDENT per-store Q-tables in a single (N, state_size, action_size)
ndarray, stacked along axis 0 purely for vectorized indexing/updates — this is
NOT a shared Q-table. Store i reads and writes only _q[i]; its TD update uses
only its own state, action, and reward. Equivalent to N separate QLearningAgent
instances (ADR-004, per-store independent Q-tables); vectorized for speed at
N~494 stores. See ADR-004.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class BatchQLearningAgent:
    """Tabular Q-learning for N symmetric agents with vectorized updates."""

    def __init__(
        self,
        n_agents: int,
        m: int,
        m_effort: int,
        k: int,
        alpha: float,
        beta_decay: float,
        delta: float,
        seed: int | None = None,
        max_qtable_gib: float = 8.0,
        state_mode: str = "neighbors",
        state_size: int | None = None,
        epsilon_min: float = 3e-4,
        beta1: float | None = None,
        beta2: float | None = None,
        t0: int = 0,
        epsilon_transition: float = 0.10,
    ) -> None:
        self.n = n_agents
        self.m = m
        self.m_effort = m_effort
        self.k = k
        self.alpha = alpha
        self.beta_decay = beta_decay
        self.delta = delta
        self.state_mode = state_mode
        self.action_size = m * m_effort
        # ── Two-stage exploration schedule ─────────────────────────────────────
        # When two_stage=True: stage-1 uses β₁, stage-2 uses β₂ starting from
        # ε_transition (continuous at t₀). Single-stage path: uses beta_decay
        # with an ε_min floor (backward-compatible, slightly stricter than old code).
        self.epsilon_min = float(epsilon_min)
        self.epsilon_transition = float(epsilon_transition)
        self.two_stage = (
            beta1 is not None and beta2 is not None and int(t0) > 0
        )
        self.beta1 = float(beta1) if beta1 is not None else float(beta_decay)
        self.beta2 = float(beta2) if beta2 is not None else float(beta_decay)
        self.t0 = int(t0)
        self.state_size = (
            int(state_size) if state_size is not None else self.action_size ** k
        )

        n_qvals = n_agents * self.state_size * self.action_size
        gib = n_qvals * 8 / (1024**3)
        logger.info(
            "Q-table: %d agents x %d states x %d actions = %.3g values "
            "(%.2f GiB, k=%d, action_size=%d)",
            n_agents,
            self.state_size,
            self.action_size,
            n_qvals,
            gib,
            k,
            self.action_size,
        )
        if gib > max_qtable_gib:
            raise MemoryError(
                f"Q-table would need {gib:.1f} GiB > max_qtable_gib={max_qtable_gib} "
                f"(k={k}, action_size={self.action_size}, states={self.state_size}). "
                f"Reduce k_neighbors, freeze effort (m_effort=1), or raise "
                f"max_qtable_gib explicitly."
            )

        self._rng = np.random.default_rng(seed)
        self._t = np.zeros(n_agents, dtype=np.int64)
        # Axis 0 indexes stores: _q[i] is store i's OWN independent Q-table (ADR-004).
        # Stacked into one ndarray only for vectorized act()/update(); no parameter sharing.
        self._q = np.zeros(
            (n_agents, self.state_size, self.action_size),
            dtype=np.float64,
        )
        # Optional non-zero Q-table initialization. When set via set_q_init(),
        # reset() restores _q to _q_init instead of zeros. Default: zeros.
        self._q_init: np.ndarray | None = None

        # Envelope enforcement (set externally by the CEO layer in Phase 2).
        # _action_mask: (N, action_size) bool — True = action allowed for that store.
        # _epsilon_override: (N,) float — per-store exploration rate overriding the
        # exp(-beta_decay*t) schedule. Both None => unconstrained baseline behaviour.
        self._action_mask: np.ndarray | None = None
        self._epsilon_override: np.ndarray | None = None

    def reset(self) -> None:
        """Reset Q-table to initial values and clear exploration counters.

        If :meth:`set_q_init` was called, the Q-table resets to the provided
        initialization tensor (Calvano eq. 8 or variant). Otherwise zeros.
        """
        if self._q_init is not None:
            self._q[:] = self._q_init
        else:
            self._q[:] = 0.0
        self._t[:] = 0
        self._action_mask = None
        self._epsilon_override = None

    def set_q_init(self, q_init: np.ndarray) -> None:
        """Set non-zero initial Q-values used by :meth:`reset`.

        When set, ``reset()`` restores ``_q`` to ``q_init`` instead of zeros.
        This implements the Calvano et al. (2020) equation (8) initialization
        and related strategies.

        Parameters
        ----------
        q_init : (N, state_size, action_size) float64 array.
            Must match the shape of the Q-table. Typically state-independent
            (all state slices identical), but this is not enforced.
        """
        q_init = np.asarray(q_init, dtype=np.float64)
        expected = (self.n, self.state_size, self.action_size)
        if q_init.shape != expected:
            raise ValueError(
                f"q_init shape {q_init.shape} != Q-table shape {expected}"
            )
        self._q_init = q_init.copy()
        logger.info(
            "Q-table init set: shape=%s, mean=%.6f, std=%.6f, "
            "min=%.6f, max=%.6f",
            q_init.shape,
            float(q_init.mean()),
            float(q_init.std()),
            float(q_init.min()),
            float(q_init.max()),
        )

    def set_action_mask(self, mask: np.ndarray | None) -> None:
        """Restrict each store's selectable joint actions to an in-envelope subset.

        Parameters
        ----------
        mask : (N, action_size) bool array, or None to clear.
            ``mask[i, a] == True`` means store ``i`` may play joint action ``a``.
            Every row must contain at least one True (guaranteed by the Step-2
            mask builder via snap-to-nearest). The Q-table itself is unchanged;
            only selection in ``act()`` is constrained, so learned Q-values stay
            valid across envelope (epoch) changes — the grid is never re-discretised.
        """
        if mask is None:
            self._action_mask = None
            return
        mask = np.ascontiguousarray(mask, dtype=bool)
        if mask.shape != (self.n, self.action_size):
            raise ValueError(
                f"action mask shape {mask.shape} != (N={self.n}, A={self.action_size})"
            )
        self._action_mask = mask

    def set_epsilon_override(self, eps: np.ndarray | None) -> None:
        """Override the exp(-beta_decay*t) exploration schedule with per-store epsilon.

        Parameters
        ----------
        eps : (N,) float array in (0, 1), or None to revert to the decay schedule.
            Used in Phase 2 where the CEO sets exploration per store group.
        """
        if eps is None:
            self._epsilon_override = None
            return
        eps = np.ascontiguousarray(eps, dtype=np.float64)
        if eps.shape != (self.n,):
            raise ValueError(f"epsilon override shape {eps.shape} != (N={self.n},)")
        self._epsilon_override = eps

    def save_qtable(self, path) -> None:
        """Persist the Q-tensor + counters + identifying metadata to a .npz file."""
        import numpy as np  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            q=self._q,
            t=self._t,
            n_agents=np.int64(self.n),
            state_size=np.int64(self.state_size),
            action_size=np.int64(self.action_size),
            m=np.int64(self.m),
            m_effort=np.int64(self.m_effort),
            k=np.int64(self.k),
            state_mode=np.array(self.state_mode),
            beta_schedule=np.array("two_stage" if self.two_stage else "single"),
            beta1=np.float64(self.beta1),
            beta2=np.float64(self.beta2),
            t0_schedule=np.int64(self.t0),
            epsilon_min=np.float64(self.epsilon_min),
            epsilon_transition=np.float64(self.epsilon_transition),
        )

    def load_qtable(self, path) -> None:
        """Load a Q-tensor saved by ``save_qtable``, validating compatibility.

        Raises ValueError if (n_agents, state_size, action_size) or state_mode
        do not match this agent — a mismatch means the env/agent config differs
        from the run that produced the checkpoint, so the Q-values are not
        transferable (the action grid / state encoding would differ).
        """
        import numpy as np  # noqa: PLC0415

        d = np.load(path, allow_pickle=True)
        got = (int(d["n_agents"]), int(d["state_size"]), int(d["action_size"]))
        want = (self.n, self.state_size, self.action_size)
        if got != want:
            raise ValueError(
                f"Q-table shape mismatch: checkpoint {got} != agent {want}. "
                "The strategic run's m / m_effort / k / state_mode must match the "
                "baseline run that produced the checkpoint."
            )
        ckpt_mode = str(d["state_mode"])
        if ckpt_mode != self.state_mode:
            raise ValueError(
                f"state_mode mismatch: checkpoint {ckpt_mode!r} != agent {self.state_mode!r}."
            )
        self._q[:] = d["q"]
        self._t[:] = d["t"]
        logger.info(
            "Loaded Q-table from %s (n=%d, state_size=%d, action_size=%d, mean t=%.0f).",
            path, self.n, self.state_size, self.action_size, float(self._t.mean()),
        )

    def _encode_states(self, signal: np.ndarray) -> np.ndarray:
        """Encode state signal → flat state indices (N,)."""
        if self.state_mode in (
            "local_summary",
            "design4_ownprice", "design5_full", "calvano_local", "strategic_hybrid",
            "hybrid_profit_gap", "graph_states",
        ):
            s = np.asarray(signal, dtype=np.int64)
            assert s.ndim == 1 and s.shape[0] == self.n, (
                f"{self.state_mode} signal must be (N,), got {signal.shape}"
            )
            return s
        neighbor_actions = signal
        if self.k == 1:
            return neighbor_actions[:, 0].astype(np.int64)
        multipliers = self.action_size ** np.arange(self.k, dtype=np.int64)
        return (neighbor_actions.astype(np.int64) * multipliers[None, :]).sum(axis=1)

    def act(self, neighbor_actions: np.ndarray) -> np.ndarray:
        """Choose joint action indices for all agents.

        Parameters
        ----------
        neighbor_actions : (N, k) int array of neighbors' joint action indices
            in ``neighbors`` mode, or (N,) state-index vector in ``local_summary``.

        Returns
        -------
        (N,) int array of chosen joint action indices. When an action mask is
        set, all chosen actions lie in the masked (in-envelope) subset.
        """
        states = self._encode_states(neighbor_actions)

        # Exploration rate: CEO override if set, else scheduled decay.
        if self._epsilon_override is not None:
            epsilons = self._epsilon_override
        elif self.two_stage:
            # Stage 1 (t ≤ t₀): standard exponential from ε(0)=1.
            # Stage 2 (t > t₀): continue from ε_transition → collapse to ε_min.
            # Continuous at t₀ — no jump.  Both branches floored at ε_min.
            stage1_mask = self._t <= self.t0
            eps_s1 = np.exp(-self.beta1 * self._t)
            eps_s2 = self.epsilon_transition * np.exp(
                -self.beta2 * np.maximum(np.int64(0), self._t - self.t0)
            )
            epsilons = np.maximum(
                self.epsilon_min, np.where(stage1_mask, eps_s1, eps_s2)
            )
        else:
            epsilons = np.maximum(
                self.epsilon_min, np.exp(-self.beta_decay * self._t)
            )

        # Greedy action: argmax over (optionally masked) Q-row + tie-break noise.
        q_rows = self._q[np.arange(self.n), states]
        scored = q_rows + self._rng.random((self.n, self.action_size)) * 1e-10
        if self._action_mask is not None:
            scored = np.where(self._action_mask, scored, -np.inf)
        greedy_actions = np.argmax(scored, axis=1)

        # Exploratory action: uniform over allowed actions per store.
        if self._action_mask is None:
            random_actions = self._rng.integers(0, self.action_size, size=self.n)
        else:
            rand_scores = self._rng.random((self.n, self.action_size))
            rand_scores = np.where(self._action_mask, rand_scores, -np.inf)
            random_actions = np.argmax(rand_scores, axis=1)

        explore_mask = self._rng.random(self.n) < epsilons
        actions = np.where(explore_mask, random_actions, greedy_actions).astype(np.int64)

        self._t += 1
        return actions

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
    ) -> None:
        """Vectorized TD update for all N agents.

        When an envelope action mask is active (Phase 2), the bootstrap
        ``best_next`` is taken over the FEASIBLE (in-envelope) actions only, so
        the continuation value is consistent with what the masked policy can
        actually play. With no mask set (burn-in / baseline) this is identical
        to the unconstrained max over all actions.
        """
        idx_n = np.arange(self.n)
        current_q = self._q[idx_n, states, actions]
        next_q = self._q[idx_n, next_states]                       # (N, action_size)
        if self._action_mask is not None:
            next_q = np.where(self._action_mask, next_q, -np.inf)
        best_next = next_q.max(axis=1)
        td_error = rewards + self.delta * best_next - current_q
        self._q[idx_n, states, actions] += self.alpha * td_error

    @property
    def epsilon_mean(self) -> float:
        """Mean exploration probability across agents (override if set, else scheduled decay)."""
        if self._epsilon_override is not None:
            return float(self._epsilon_override.mean())
        if self.two_stage:
            stage1_mask = self._t <= self.t0
            eps_s1 = np.exp(-self.beta1 * self._t)
            eps_s2 = self.epsilon_transition * np.exp(
                -self.beta2 * np.maximum(np.int64(0), self._t - self.t0)
            )
            raw = np.where(stage1_mask, eps_s1, eps_s2)
            return float(np.maximum(self.epsilon_min, raw).mean())
        return float(
            np.maximum(self.epsilon_min, np.exp(-self.beta_decay * self._t)).mean()
        )
