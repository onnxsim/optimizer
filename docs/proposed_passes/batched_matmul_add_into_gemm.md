# Proposed: fuse batched MatMul + bias Add into Gemm

**Status:** design draft — NOT implemented, NOT built, NOT tested.
**Cause:** transformer models (bart, mvp, swin_s) where onnxslim lands fewer nodes than onnxsim.

## Evidence

On the onnxmodelzoo regression set, the largest node-count divergences in
onnxslim's favour are dominated by this one rewrite. Measured op-type deltas
(onnxsim → onnxslim) on the simplified graphs:

| model | MatMul | Add | Gemm | Reshape | net nodes |
| --- | --- | --- | --- | --- | --- |
| bart_Opset18 | −96 | −96 | +96 | +85 | −29 |
| mvp_Opset18 | −192 | −192 | +192 | +169 | −59 |
| swin_s_Opset18 | −96 | −96 | +96 | +81 | −237¹ |

¹ swin_s also removes residual dynamic-shape ops (separate cause).

## Why onnxsim leaves them

`fuse_matmul_add_bias_into_gemm` only handles the 2-D case: its predicate
requires `x_shape.size() == 2` and `y_shape.size() == 2`. Transformer models
apply linear layers to rank-3 activations `[B, S, K] · [K, N]`, so the MatMul is
batched and the pass bails. onnxslim instead reshapes to 2-D, emits a `Gemm`,
and reshapes back.

## Proposed transform

For `Z = MatMul(X, W); A = Z + b` with:
- `X` rank ≥ 3 with a static trailing dim `K` (`X: [d0, …, d_{r-2}, K]`),
- `W` a 2-D **constant** `[K, N]`,
- `b` 1-D `[N]` (or broadcastable to `[…, N]`),
- `Z` used only by the `Add`:

rewrite to

```
X2 = Reshape(X, [-1, K])          # collapse leading dims
G  = Gemm(X2, W, b)               # alpha=beta=1, transA=transB=0  -> [-1, N]
A  = Reshape(G, concat(shape(X)[:-1], [N]))
```

The trailing-dim reshape target is `-1, K` (a constant initializer). The output
reshape target must reconstruct the original leading dims; when they are static,
emit a constant shape `[d0, …, d_{r-2}, N]`, otherwise build it dynamically from
`Shape(X)` (`Slice` off the last dim, `Concat` with `[N]`).

## Sketch (onnxoptimizer IR)

```cpp
// predicate: CheckKind(n, kAdd, 0, kMatMul), MatMul used once,
//   x rank >= 3 with static last dim, W 2-D constant, bias 1-D/2-D compatible.
// transform:
//   Value* x   = matmul->inputs()[0];
//   Value* w   = matmul->inputs()[1];
//   Node* pre  = graph.create(kReshape, {x, const_2d_shape});   // [-1, K]
//   Node* gemm = graph.create(kGemm, {pre->output(), w, bias});
//   gemm->f_(kalpha,1.0)->f_(kbeta,1.0)->i_(ktransA,0)->i_(ktransB,0);
//   Node* post = graph.create(kReshape, {gemm->output(), out_shape});
//   ... insertBefore(n); tryReplacingAllUsesWith(n, post); destroy MatMul ...
```

## Caveat — is this worth upstreaming?

This is **node-count-only**. It trades `MatMul` for `Reshape + Gemm + Reshape`,
so it removes ~2 nodes per layer but *adds* Reshapes (net −29 on bart, −59 on
mvp). Batched `MatMul` and 2-D `Gemm` are equivalent work and modern runtimes
execute batched MatMul natively — often the reshaped form is no faster and can
be marginally slower. onnxslim does it mainly for a cleaner Gemm-centric graph.

Recommend treating this as **optional / cosmetic**, gated behind a flag, rather
than adding it to the default pass set. Included here for completeness because
it is the single biggest contributor to onnxslim's lower node counts on
transformers.
