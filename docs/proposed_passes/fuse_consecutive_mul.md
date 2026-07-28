# Proposed: fuse consecutive constant-scaled Mul (poolformer)

**Status:** design draft — NOT implemented, NOT built, NOT tested.
**Cause:** poolformer_m48 (onnxsim 1114 → onnxslim 1066; the −48 gap is entirely
`Mul`).

## Evidence

Node-level inspection of onnxsim's simplified poolformer graph (288 `Mul`
nodes):

- **48 `Mul` nodes feed directly into another `Mul`** (`consumers of Mul
  outputs: {Add: 192, Mul: 48, Conv: 48}`), and **48 `Mul` nodes are fed by a
  `Mul`** (`producers feeding Mul: {Reshape: 96, Conv: 96, Sub: 48, Add: 48,
  Mul: 48}`). i.e. there are 48 `Mul → Mul` chains.
- Their constant operands are `()` (scalar) ×48 and per-channel `(C,1,1)` for
  C ∈ {96,192,384,768}.

This is PoolFormer's LayerScale: a per-channel constant scale composed with a
scalar factor, exported as two back-to-back constant-operand `Mul`s. onnxslim
collapses each pair into one `Mul`, removing 48 nodes. onnxsim keeps both — it
has `eliminate_nop_with_unit` (Mul-by-1) and constant folding, but neither
merges two runtime-input Muls that each also carry a constant operand.

## Proposed pass: `fuse_consecutive_mul`

Fold `Mul(Mul(X, C1), C2) → Mul(X, C1 * C2)` when `C1` and `C2` are constants
and the inner Mul is used only by the outer one.

Predicate:
- `n` is `Mul`; one input is a constant `C2`, the other is produced by a `Mul`
  `m` whose output is used only by `n`;
- `m` has one constant input `C1` and one non-constant input `X`;
- `C1` and `C2` are broadcast-compatible (both are scalar or `(C,1,1)`-style
  channel vectors here, so the product broadcasts trivially).

Transform:
- materialise `C = C1 * C2` (numpy-style broadcast) as a new initializer;
- rewrite to `Mul(X, C)`; drop the inner Mul (DCE removes the dead constants).

## Sketch (onnxoptimizer IR)

```cpp
// patternMatchPredicate: CheckKind(n, kMul); exactly one constant operand;
//   other operand produced by a Mul `m` with uses()==1 and exactly one
//   constant operand.
// runTransform:
//   Tensor c1 = *FetchConstantTensor(m_const_input);
//   Tensor c2 = *FetchConstantTensor(n_const_input);
//   Tensor c  = ElementwiseMul(c1, c2);      // broadcasting helper
//   Value*  cv = graph.addInitializerAndCreateValue(c);
//   Node*   fused = graph.create(kMul, {x, cv});
//   fused->insertBefore(n); tryReplacingAllUsesWith(n, fused);
//   destroy n; DCE removes m and the dead constants.
```

Needs a small broadcasting element-wise multiply over `Tensor` (scalar × scalar,
scalar × 1-D/3-D channel vector, and equal shapes cover every case seen here;
`tensor_util.h` already has the primitives to iterate raw data). General and
safe: it only ever combines two provably-constant scales, so numerics are
preserved (onnxslim's equivalence check passes on this model).

Not implemented or tested here — the broadcasting constant multiply is the only
non-trivial part and wants unit tests before landing.
