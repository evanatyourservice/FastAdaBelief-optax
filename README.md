# FastAdaBelief-optax

Implementation of FastAdaBelief optimizer described [here](https://arxiv.org/abs/2104.13790)
for [JAX](https://github.com/google/jax) 
using [optax](https://github.com/deepmind/optax)

FastAdaBelief exploits strong convexity in order to achieve faster convergence
rates while maintaining excellent generalization. It is a modification of
AdaBelief with O(logT) regret bound to satisfy the property of strongly convex
optimization, time-dependent beta2, and time-dependent epsilon. It has three
sets of parameters versus Adam's and AdaBelief's two.

Haiku example of using a mask for weight decay:
```
# only apply weight decay to weights, not bias or any norm layers
mask = partial(hk.data_structures.map, lambda mname, name, val: name == 'w')
optimizer = fastadabelief(learning_rate=0.001, weight_decay=0.001, mask=mask)
```
