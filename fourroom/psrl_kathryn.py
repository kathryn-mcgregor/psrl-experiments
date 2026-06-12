"""
JAX implementation of Value Iteration (VI) and Posterior Sampling for
Reinforcement Learning (PSRL) for finite tabular MDPs.

This file is self-contained except for the environment.  The PSRLAgent class
can be used with any environment that exposes
    (obs, action, reward, next_obs, done)
style interaction.

Dependencies: jax, jax.numpy, numpy.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Value Iteration  (infinite-horizon discounted, tabular)
# ---------------------------------------------------------------------------

def value_iteration(R: jnp.ndarray,
                    T: jnp.ndarray,
                    gamma: float,
                    tol: float = 1e-6,
                    max_iters: int = 2000):
    """Compute Q* and the greedy policy for a finite tabular MDP.

    Args:
        R: shape (S, A) — expected immediate reward R[s, a].
        T: shape (S, A, S) — transition probabilities T[s, a, s'].
        gamma: discount factor in [0, 1).
        tol: Bellman residual tolerance for early stopping.
        max_iters: hard cap on iterations (static, used by lax.while_loop).

    Returns:
        Q_star: shape (S, A).
        pi_star: shape (S,) — greedy actions (int32).
    """
    S, A = R.shape
    Q = jnp.zeros((S, A), dtype=jnp.float32)

    def _body(val):
        Q_curr, i = val
        V = jnp.max(Q_curr, axis=-1)                        # (S,)
        Q_next = R + gamma * jnp.einsum('sat,t->sa', T, V)  # (S,A)
        return Q_next, i + 1

    def _cond(val):
        Q_curr, i = val
        # Run at least one iteration, then check convergence.
        Q_next, _ = _body(val)
        converged = jnp.max(jnp.abs(Q_next - Q_curr)) < tol
        return (~converged) & (i < max_iters)

    Q_star, _total = jax.lax.while_loop(_cond, _body, (Q, jnp.int32(0)))
    pi_star = jnp.argmax(Q_star, axis=-1).astype(jnp.int32)
    return Q_star, pi_star

# JIT-compile VI.  max_iters and gamma are static so that the loop bound
# and arithmetic are compiled once; R and T are traced normally.
_vi_jit = jax.jit(value_iteration, static_argnames=('gamma', 'tol', 'max_iters'))


# ---------------------------------------------------------------------------
# MDP Posterior Sampling  (functional kernels, JIT-friendly)
# ---------------------------------------------------------------------------

def sample_mdp(key: jnp.ndarray,
               R_post: jnp.ndarray,
               T_post: jnp.ndarray,
               R_vals: jnp.ndarray):
    """Sample a transition model and expected-reward matrix from the posterior.

    Args:
        key: JAX PRNG key.
        R_post: shape (S, A, K) — Dirichlet concentration for reward categories.
        T_post: shape (S, A, S) — Dirichlet concentration for transitions.
        R_vals: shape (K,) — discrete reward support values.

    Returns:
        R_sample: shape (S, A) — sampled expected rewards.
        T_sample: shape (S, A, S) — sampled transition probabilities.
    """
    key_t, key_r = jax.random.split(key)

    # Sample transitions: one Dirichlet per (s, a) row.
    # jax.random.dirichlet can broadcast over leading dims of alpha.
    S, A, _ = T_post.shape
    flat_T_alpha = T_post.reshape(S * A, -1)                # (S*A, S)
    keys_t = jax.random.split(key_t, S * A)
    flat_T = jax.vmap(jax.random.dirichlet)(keys_t, flat_T_alpha)  # (S*A, S)
    T_sample = flat_T.reshape(S, A, -1)

    # Sample reward distributions: Dirichlet per (s, a), then compute
    # expected reward = p_sampled · R_vals.
    K = R_vals.shape[0]
    flat_R_alpha = R_post.reshape(S * A, K)                  # (S*A, K)
    keys_r = jax.random.split(key_r, S * A)
    flat_p = jax.vmap(jax.random.dirichlet)(keys_r, flat_R_alpha)  # (S*A, K)
    R_sample = (flat_p @ R_vals).reshape(S, A)               # (S, A)

    return R_sample, T_sample

_sample_mdp_jit = jax.jit(sample_mdp)


def update_posterior(R_post: jnp.ndarray,
                     T_post: jnp.ndarray,
                     obs: jnp.ndarray,
                     actions: jnp.ndarray,
                     reward_idxs: jnp.ndarray,
                     next_obs: jnp.ndarray):
    """Batch-update Dirichlet posteriors from a trajectory.

    Each array has shape (L,) where L = trajectory length.
    """
    R_post = R_post.at[obs, actions, reward_idxs].add(1.0)
    T_post = T_post.at[obs, actions, next_obs].add(1.0)
    return R_post, T_post

_update_posterior_jit = jax.jit(update_posterior)


# ---------------------------------------------------------------------------
# PRNG helper  (replaces hk.PRNGSequence without the Haiku dependency)
# ---------------------------------------------------------------------------

class PRNGSequence:
    """Infinite iterator of JAX PRNG keys."""
    def __init__(self, seed):
        if isinstance(seed, int):
            self._key = jax.random.PRNGKey(seed)
        else:
            self._key = seed

    def __next__(self):
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def __iter__(self):
        return self


# ---------------------------------------------------------------------------
# Agent State  (JAX PyTree-compatible — no None fields)
# ---------------------------------------------------------------------------

class PSRLAgentState(NamedTuple):
    R_post: jnp.ndarray      # (S, A, K)
    T_post: jnp.ndarray      # (S, A, S)
    pi_star: jnp.ndarray     # (S,)   — current greedy policy
    has_policy: jnp.ndarray  # scalar bool — True if pi_star is valid


# ---------------------------------------------------------------------------
# PSRL Agent
# ---------------------------------------------------------------------------

class PSRLAgent:
    """Posterior Sampling for Reinforcement Learning (episodic, tabular).

    Usage::

        agent = PSRLAgent(num_states=6, num_actions=2, gamma=0.99,
                          seed=42, R_vals=[0., 0.005, 1.])
        for episode in range(num_episodes):
            obs = env.reset()
            done = False
            while not done:
                action = agent.act(obs)
                next_obs, reward, done, info = env.step(action)
                agent.update(obs, action, reward, next_obs, done)
                obs = next_obs
    """

    def __init__(self,
                 num_states: int,
                 num_actions: int,
                 gamma: float,
                 seed: int,
                 R_vals,
                 T_alpha0: float = 1.0,
                 R_alpha0: float = 0.01):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.R_vals_list = list(R_vals)
        self.R_vals = jnp.array(R_vals, dtype=jnp.float32)
        self.rng = PRNGSequence(seed)

        K = len(self.R_vals_list)

        # Dirichlet priors — uniform concentration.
        R_prior = jnp.full((num_states, num_actions, K), R_alpha0, dtype=jnp.float32)
        T_prior = jnp.full((num_states, num_actions, num_states), T_alpha0, dtype=jnp.float32)

        self.agent_state = PSRLAgentState(
            R_post=R_prior,
            T_post=T_prior,
            pi_star=jnp.zeros(num_states, dtype=jnp.int32),
            has_policy=jnp.bool_(False),
        )

        # Episode bookkeeping (Python-side; small episodes are fine).
        self._trajectory = []
        self._episode_reward = 0.0

        # Pre-build a reward-value lookup for robust float→index mapping.
        # We round to 8 decimal places to tolerate minor float noise.
        self._r_val_to_idx = {round(float(v), 8): i for i, v in enumerate(self.R_vals_list)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, obs: int) -> int:
        """Return an action for the current observation.

        At the start of each episode (no valid policy), samples a new MDP
        from the posterior and solves it.  Within an episode the policy is
        held fixed per standard PSRL.
        """
        state = self.agent_state

        if not state.has_policy:
            # Sample MDP and solve.
            key = next(self.rng)
            key_sample, key_vi = jax.random.split(key)
            R_sample, T_sample = _sample_mdp_jit(key_sample,
                                                  state.R_post,
                                                  state.T_post,
                                                  self.R_vals)
            _, pi_star = _vi_jit(R_sample, T_sample, self.gamma)
            self.agent_state = state._replace(pi_star=pi_star,
                                              has_policy=jnp.bool_(True))

        action = int(self.agent_state.pi_star[obs])
        return action

    def update(self, obs, action, reward, next_obs, done):
        """Store transition; on episode end, update posteriors and reset policy."""
        self._trajectory.append((obs, action, reward, next_obs))
        self._episode_reward += float(reward)

        if done:
            # Build trajectory arrays.
            obs_arr = jnp.array([t[0] for t in self._trajectory], dtype=jnp.int32)
            act_arr = jnp.array([t[1] for t in self._trajectory], dtype=jnp.int32)
            ridx_arr = jnp.array([self._reward_to_idx(t[2]) for t in self._trajectory],
                                 dtype=jnp.int32)
            nobs_arr = jnp.array([t[3] for t in self._trajectory], dtype=jnp.int32)

            R_post_new, T_post_new = _update_posterior_jit(
                self.agent_state.R_post,
                self.agent_state.T_post,
                obs_arr, act_arr, ridx_arr, nobs_arr,
            )

            # Invalidate policy so next act() samples a fresh MDP.
            self.agent_state = self.agent_state._replace(
                R_post=R_post_new,
                T_post=T_post_new,
                has_policy=jnp.bool_(False),
            )

            self._trajectory = []
            self._episode_reward = 0.0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _reward_to_idx(self, r) -> int:
        """Map an observed reward to its index in R_vals.

        Uses rounded lookup to tolerate minor floating-point noise.
        """
        key = round(float(r), 8)
        idx = self._r_val_to_idx.get(key)
        if idx is None:
            raise ValueError(
                f"Observed reward {r} not found in R_vals {self.R_vals_list}. "
                "Ensure environment rewards are exactly in R_vals."
            )
        return idx


# ============================================================================
# Example trial runner (requires RiverSwim + run_agent_env to be provided)
# ============================================================================

_example_config = {
    'river_len': 3,
    'discount': 0.99,
    'num_episodes': 35,
    'seed': 635,
}

_example_psrl_config = {
    'T_alpha0': 1.0 / _example_config['river_len'],
    'R_alpha0': 0.01,
    'R_vals': [0., 0.005, 1.],
}


def run_vanilla_psrl_trial(env_cls, run_agent_env_fn, config, psrl_config,
                           gen_plot=True):
    """Lightweight trial wrapper.

    Args:
        env_cls: callable returning an environment with .reset() / .step().
        run_agent_env_fn: function(config, agent, env, gen_plot) -> results.
        config: experiment config dict (must include 'river_len', 'discount',
                'seed', 'num_episodes', and optionally 'horizon').
        psrl_config: dict with 'T_alpha0', 'R_alpha0', 'R_vals'.
        gen_plot: whether to generate plots.
    """
    import random as _random
    _random.seed(config['seed'])
    np.random.seed(config['seed'])

    env = env_cls(river_len=config['river_len'], seed=config['seed'])
    agent = PSRLAgent(
        num_states=config['river_len'],
        num_actions=2,
        gamma=config['discount'],
        seed=config['seed'],
        R_vals=psrl_config['R_vals'],
        T_alpha0=psrl_config['T_alpha0'],
        R_alpha0=psrl_config['R_alpha0'],
    )
    return run_agent_env_fn(config, agent, env, gen_plot=gen_plot)
