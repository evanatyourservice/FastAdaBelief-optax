from typing import Union, NamedTuple, Optional, Any, Callable

import jax.numpy as jnp
import jax
import chex

from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import transform


ScalarOrSchedule = Union[float, base.Schedule]


def _scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return transform.scale_by_schedule(lambda count: m * learning_rate(count))
  return transform.scale(m * learning_rate)


class ScaleByFastBeliefState(NamedTuple):
  """State for the rescaling by FastAdaBelief algorithm."""
  count: chex.Array  # # shape=(), dtype=jnp.int32.
  m: base.Updates
  s: base.Updates
  max_s: base.Updates


def scale_by_fastbelief(
    b1: float = 0.9,
    eps_root: float = 0.0
) -> base.GradientTransformation:
  """Rescale updates according to the FastAdaBelief algorithm.

  References:
    Zhou et al, 2021: https://arxiv.org/abs/2104.13790

  Args:
    b1: decay rate for the exponentially weighted average of grads.
    eps_root: term added to the second moment of the prediction error to
      improve numerical stability. If backpropagating gradients through the
      gradient transformation (e.g. for meta-learning), this must be non-zero.

  Returns:
    An (init_fn, update_fn) tuple.
  """

  def init_fn(params):
    m = jax.tree_map(jnp.zeros_like, params)  # First moment
    s = jax.tree_map(jnp.zeros_like, params)  # Second moment
    max_s = jax.tree_map(jnp.zeros_like, params)  # Authors use AMSGrad by default
    return ScaleByFastBeliefState(
      count=jnp.zeros([], jnp.int32), m=m, s=s, max_s=max_s
    )

  def update_fn(updates, state: ScaleByFastBeliefState, params=None):
    del params
    t = numerics.safe_int32_increment(state.count)
    b2 = 1.0 - 0.9 / t
    m = jax.tree_multimap(lambda m, g: b1 * m + (1.0 - b1) * g, state.m, updates)
    s = jax.tree_multimap(
      lambda s, g, m: b2 * s + (1.0 - b2) * (g - m) ** 2 + eps_root,
      state.s,
      updates,
      m,
    )
    max_s = jax.tree_multimap(lambda ms, s: jnp.maximum(ms, s), state.max_s, s)
    max_su = jax.tree_map(lambda ms: ms + 0.01 / t, max_s)
    updates = jax.tree_multimap(lambda m, msu: m / msu, m, max_su)
    return updates, ScaleByFastBeliefState(count=t, m=m, s=s, max_s=max_s)

  return base.GradientTransformation(init_fn, update_fn)


def fastadabelief(
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
) -> base.GradientTransformation:
  """The FastAdaBelief optimizer.

  FastAdaBelief exploits strong convexity in order to achieve faster convergence
  rates while maintaining excellent generalization. It is a modification of
  AdaBelief with O(logT) regret bound to satisfy the property of strongly convex
  optimization, time-dependent beta2, and time-dependent epsilon. It has three
  sets of parameters versus Adam's and AdaBelief's two.

  WARNING: Sometimes you may want to skip weight decay for BatchNorm scale or
  for the bias parameters. You can use `optax.masked` to apply weight decay only
  to a subset of `params`.

  References:
    Zhou et al, 2021: https://arxiv.org/abs/2104.13790

  Args:
    learning_rate: this is a fixed global scaling factor.
    b1: the exponential decay rate to track the first moment of past gradients.
    eps_root: (default `0`), a small constant applied to denominator inside the
      square root (as in RMSProp), to avoid dividing by zero when rescaling.
      This is needed for instance when computing (meta-)gradients through Adam.
    weight_decay: strength of the weight decay regularization.
    mask: a tree with same structure as (or a prefix of) the params PyTree,
      or a Callable that returns such a pytree given the params/updates.
      The leaves should be booleans, `True` for leaves/subtrees you want to
      apply the transformation to, and `False` for those you want to skip.

  Returns:
    the corresponding `GradientTransformation`.
  """
  return combine.chain(
    scale_by_fastbelief(b1=b1, eps_root=eps_root),
    transform.add_decayed_weights(weight_decay, mask),
    _scale_by_learning_rate(learning_rate),
  )
