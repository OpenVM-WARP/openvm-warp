# Reduced-SWIRL WARP proving

This document describes the WARP proving lane added by this fork. The lane replaces recursive aggregation of complete segment proofs; it does not replace OpenVM execution, AIR constraints, LogUp, or final recursive compression.

## End-to-end flow

```text
segmented OpenVM execution
        |
        v
AIR + LogUp + ordered SWIRL stacking
        |
        | stop before per-segment WHIR
        v
original constrained stacked RS source
        |
        v
fixed-arity resident WARP transitions
        |
        +----> recursively proved transition leaves and tree
        |
        v
terminal Decide: relation + RS adjoint + WHIR
        |
        v
transition finalizer and standard OpenVM recursive adapter
        |
        v
one succinct VmStarkProof
```

Each segment is executed once. Its AIR/LogUp and stacking work is also performed once. The WARP lane retains the original RS commitment and codeword owner instead of constructing a complete segment WHIR proof or re-encoding a projected accumulator source.

## Application relation

The fixed WARP relation is the residual constrained-code equation exported by the authenticated SWIRL reduction. Its witness is the batched stacked RS message; its explicit instance contains SWIRL's reduction point and claimed value. This statement derives from all preceding AIR and LogUp obligations. A PCS opening is never substituted for the application relation.

The relation and source shape are setup-bound. The relation degree depends on that fixed message dimension and does not grow with execution depth. A random block may change the number of segments, but not the relation, schedule rules, or transition-leaf capacity.

## Deterministic accumulation

The submitted benchmark configuration uses transition-leaf capacity eight. The first WARP call consumes up to eight fresh sources; every continuation call consumes the prior accumulator plus up to seven fresh sources. The final call may use a shorter active prefix. Slot activity, call index, source cursor, and total source count are constrained, so there is no block-specific ordering profile.

Each completed WARP call is verified at the native boundary and certified by one recursive transition leaf. Leaves bind:

- protocol, relation, WARP index, code parameters, and schedule digests;
- the prior and next accumulator roots and instance digests;
- source indices, ordered manifests, program identity, and VM state boundaries;
- every transcript checkpoint and challenge count;
- the recursive application verifying-key commitment.

The ordinary OpenVM recursion tree authenticates all leaves in order. The finalizer opens the initial and terminal transition states, reconciles the rolling and flat manifests, verifies terminal Decide, and hands one fixed proof to the existing recursive adapter.

## Terminal decision

WARP transition verification and terminal Decide are separate obligations. Transition proofs authenticate how fresh sources are folded into the final accumulator. Decide proves that the final accumulator satisfies both the application relation and the linear-code claim. The latter uses the exact shared RS encoder adjoint before compatible linear claims are batched into terminal WHIR under the same root.

The final artifact is an ordinary `VmStarkProof` and is verified through the standard OpenVM host verifier. A direct finalizer proof may also be emitted for diagnostics, but it is not the submission artifact.

## CUDA ownership and memory

The CUDA path keeps the source matrices, Merkle owners, WARP accumulator, Claim 6.5 work, and terminal WHIR data resident where their lifetimes overlap. It uses the same stream and memory-manager primitives as the recursive prover. The stream is poisoned on error; there is no CPU fallback, nondeterministic spill policy, or host reconstruction of an opaque device vector.

After each transition, consumed sources are dropped. At phase boundaries the implementation releases scratch allocations before the recursion tree and finalizer are proved. The type boundary gives terminal Decide the retained commitment owner directly; no API accepts an uncommitted accumulator message on this path. Runtime telemetry checks that terminal proving consumes exactly one existing accumulator root, accounts for bounded proof downloads, records all PCIe transfers, and samples live and driver-reported GPU memory. These measurements are regression data rather than substitutes for the ownership invariant.

## Parameters and security status

The benchmark CLI exposes WARP family target bits. The publication matrix uses EF4 and an 80-bit target to compare the recursive and WARP architectures with the same extension field. That profile is a benchmarking choice, not a production security recommendation. Parameter selection also accounts for the number of sources, code rate, exceptional-set term, and all Fiat--Shamir invocations.

The WARP extension is unaudited. Its production-readiness work covers fail-closed validation, transcript binding, mutation tests, memory safety, deterministic scheduling, and reproducible benchmarks; it does not substitute for an external cryptographic audit.

## Relevant modules

- `crates/sdk/src/prover/native_warp`: orchestration, typed boundary, CUDA transport, transition tree, terminal component, and recursive adapter.
- `crates/recursion/src/native_warp`: WARP/VACC and terminal verifier AIRs.
- `crates/continuations/src/circuit/reduced_swirl_*`: source receipts, transition leaves, tree finalizer, and typed buses.
- `crates/recursion/src/system/deferred_opening.rs`: typed post-stacking checkpoint used when WHIR is deferred.

CPU and CUDA end-to-end tests are in `crates/sdk/tests/reduced_swirl_*`. The publication benchmark driver is maintained in the companion `openvm-eth-warp` repository.
