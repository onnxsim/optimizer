# Proposed: clear residual dynamic-shape scaffolding (swin_s, FasterRCNN)

**Status:** analysis + design draft — NOT a single implementable pass.
**Cause:** detector / windowed-attention models where onnxslim removes more
shape-arithmetic and control-flow ops than onnxsim.

## Evidence

Op-type deltas (onnxsim → onnxslim) on the simplified graphs:

**swin_s_Opset18** (onnxsim 1295 → onnxslim 1058; −237, of which ~−96 is the
separate MatMul→Gemm cause):

| op | onnxsim | onnxslim |
| --- | --- | --- |
| Unsqueeze | 63 | 0 |
| Expand | 54 | 0 |
| Slice | 133 | 100 |
| ScatterND | 27 | 0 |
| Concat | 74 | 47 |
| Where | 6 | 0 |
| Sub / Range / Equal | 3 each | 0 |

**FasterRCNN-10** (2824 → 2622; −202):

| op | onnxsim | onnxslim |
| --- | --- | --- |
| Unsqueeze | 379 | 310 |
| Mul | 119 | 59 |
| Add | 127 | 74 |
| Gather | 764 | 755 |
| Slice / Div / Cast / Shape / Concat | … | … (all lower) |

These are **dynamic-shape scaffolding**: Swin's window-attention mask
construction (`Range`/`Equal`/`Where`/`ScatterND`/`Expand`) and FasterRCNN's
anchor/ROI shape arithmetic (`Shape`→`Gather`→`Unsqueeze`→`Concat`→`Reshape`
chains). onnxsim already reduces them dramatically; onnxslim drives the residue
to zero.

## Why onnxslim gets further

This is **not one missing pattern** — there is no single "eliminate FooBar" pass
to add. onnxslim reaches a smaller fixpoint because it:

1. Runs `optimize → shape_infer` in a loop until the graph stops changing
   (up to 10 iterations). Each shape-inference round turns more dynamic shapes
   static, which unlocks another round of constant folding, which removes more
   `Shape`/`Gather`/`Unsqueeze`/`Concat` scaffolding, and so on.
2. Constant-folds shape subgraphs whose inputs became static after fusion.

onnxsim already has all the individual elimination passes
(`eliminate_shape_op`, `eliminate_shape_gather`, `eliminate_slice_after_shape`,
`eliminate_nop_expand`, `eliminate_deadend`, `fuse_concat_into_reshape`, …). The
gap is **how many fold/infer/eliminate iterations run**, not which passes exist.

## Proposed approach (no new op-specific pass)

1. **Iterate to a fixpoint.** Drive the fuse+eliminate pass set together with
   ONNX shape inference and onnxsim's constant folding in a loop until the node
   count is stable (bounded, e.g. ≤10 iterations), mirroring onnxslim's
   `MAX_ITER` loop. Most of the residue disappears once shapes are re-inferred
   between rounds.
2. **Re-run shape inference between rounds**, not just once up front — the
   scaffolding only becomes foldable after upstream fusions make its inputs
   static.
3. Composes with the `eliminate_initializer_from_input` fix (separate branch):
   on legacy models, unblocking constants is a prerequisite for any of this
   folding to fire at all.

This is primarily an **onnxsim driver-loop change** (how the passes are
scheduled) rather than a new onnxoptimizer pass. Filed here as the analysis
behind the swin_s / FasterRCNN divergence so the fix can be scoped correctly.

Not implemented or benchmarked.
