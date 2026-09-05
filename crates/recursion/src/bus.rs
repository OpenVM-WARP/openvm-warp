use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::interaction::InteractionBuilder;
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, D_EF};
use p3_field::PrimeCharacteristicRing;

#[macro_export]
macro_rules! define_typed_lookup_bus {
    ($Bus:ident, $Msg:ident) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $Bus(openvm_stark_backend::interaction::LookupBus);

        impl $Bus {
            #[inline]
            pub fn new(bus_index: openvm_stark_backend::interaction::BusIndex) -> Self {
                Self(openvm_stark_backend::interaction::LookupBus::new(bus_index))
            }

            pub fn lookup_key<AB>(
                &self,
                builder: &mut AB,
                key: $Msg<impl Into<AB::Expr> + Clone>,
                enabled: impl Into<AB::Expr>,
            ) where
                AB: openvm_stark_backend::interaction::InteractionBuilder,
            {
                self.0.lookup_key(builder, key.to_vec(), enabled);
            }

            #[inline]
            pub fn add_key_with_lookups<AB>(
                &self,
                builder: &mut AB,
                key: $Msg<impl Into<AB::Expr> + Clone>,
                num_lookups: impl Into<AB::Expr>,
            ) where
                AB: openvm_stark_backend::interaction::InteractionBuilder,
            {
                self.0
                    .add_key_with_lookups(builder, key.to_vec(), num_lookups);
            }
        }
    };
}

#[macro_export]
macro_rules! define_typed_per_proof_lookup_bus {
    ($Bus:ident, $Msg:ident) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $Bus(openvm_stark_backend::interaction::LookupBus);

        impl $Bus {
            #[inline]
            pub fn new(bus_index: openvm_stark_backend::interaction::BusIndex) -> Self {
                Self(openvm_stark_backend::interaction::LookupBus::new(bus_index))
            }

            /// Return the underlying interaction namespace for setup-time
            /// collision and wiring checks.
            #[inline]
            #[allow(dead_code)]
            pub fn index(&self) -> openvm_stark_backend::interaction::BusIndex {
                self.0.index
            }

            pub fn lookup_key<AB>(
                &self,
                builder: &mut AB,
                proof_idx: impl Into<AB::Expr>,
                key: $Msg<impl Into<AB::Expr> + Clone>,
                enabled: impl Into<AB::Expr>,
            ) where
                AB: openvm_stark_backend::interaction::InteractionBuilder,
            {
                let key = core::iter::once(proof_idx.into())
                    .chain(key.to_vec().into_iter().map(|x| x.into()))
                    .collect::<Vec<_>>();
                self.0.lookup_key(builder, key.to_vec(), enabled);
            }

            #[inline]
            pub fn add_key_with_lookups<AB>(
                &self,
                builder: &mut AB,
                proof_idx: impl Into<AB::Expr>,
                key: $Msg<impl Into<AB::Expr> + Clone>,
                num_lookups: impl Into<AB::Expr>,
            ) where
                AB: openvm_stark_backend::interaction::InteractionBuilder,
            {
                let key = core::iter::once(proof_idx.into())
                    .chain(key.to_vec().into_iter().map(|x| x.into()))
                    .collect::<Vec<_>>();
                self.0
                    .add_key_with_lookups(builder, key.to_vec(), num_lookups);
            }
        }
    };
}

#[macro_export]
macro_rules! define_typed_permutation_bus {
    ($Bus:ident, $Msg:ident) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $Bus(openvm_stark_backend::interaction::PermutationCheckBus);

        impl $Bus {
            #[inline]
            pub fn new(bus_index: openvm_stark_backend::interaction::BusIndex) -> Self {
                Self(openvm_stark_backend::interaction::PermutationCheckBus::new(
                    bus_index,
                ))
            }

            /// The underlying bus index, so callers can assert that two references name the
            /// same bus. Binding an AIR to a bus nothing drives surfaces only as an
            /// unbalanced LogUp sum, a long way from the wiring that caused it.
            #[inline]
            pub fn index(&self) -> openvm_stark_backend::interaction::BusIndex {
                self.0.index
            }

            #[inline]
            pub fn send<AB>(
                &self,
                builder: &mut AB,
                message: $Msg<impl Into<AB::Expr> + Clone>,
                enabled: impl Into<AB::Expr>,
            ) where
                AB: openvm_stark_backend::interaction::InteractionBuilder,
            {
                self.0.send(builder, message.to_vec(), enabled);
            }

            #[inline]
            pub fn receive<AB>(
                &self,
                builder: &mut AB,
                message: $Msg<impl Into<AB::Expr> + Clone>,
                enabled: impl Into<AB::Expr>,
            ) where
                AB: openvm_stark_backend::interaction::InteractionBuilder,
            {
                self.0.receive(builder, message.to_vec(), enabled);
            }
        }
    };
}

#[macro_export]
macro_rules! define_typed_per_proof_permutation_bus {
    ($Bus:ident, $Msg:ident) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $Bus(openvm_stark_backend::interaction::PermutationCheckBus);

        impl $Bus {
            #[inline]
            pub fn new(bus_index: openvm_stark_backend::interaction::BusIndex) -> Self {
                Self(openvm_stark_backend::interaction::PermutationCheckBus::new(
                    bus_index,
                ))
            }

            /// The underlying bus index, used when auditing setup-fixed
            /// cross-module transcript routes.
            #[inline]
            #[allow(dead_code)]
            pub fn index(&self) -> openvm_stark_backend::interaction::BusIndex {
                self.0.index
            }

            #[inline]
            pub fn send<AB>(
                &self,
                builder: &mut AB,
                proof_idx: impl Into<AB::Expr>,
                message: $Msg<impl Into<AB::Expr> + Clone>,
                enabled: impl Into<AB::Expr>,
            ) where
                AB: openvm_stark_backend::interaction::InteractionBuilder,
            {
                let message = core::iter::once(proof_idx.into())
                    .chain(message.to_vec().into_iter().map(|x| x.into()))
                    .collect::<Vec<_>>();
                self.0.send(builder, message, enabled);
            }

            #[inline]
            pub fn receive<AB>(
                &self,
                builder: &mut AB,
                proof_idx: impl Into<AB::Expr>,
                message: $Msg<impl Into<AB::Expr> + Clone>,
                enabled: impl Into<AB::Expr>,
            ) where
                AB: openvm_stark_backend::interaction::InteractionBuilder,
            {
                let message = core::iter::once(proof_idx.into())
                    .chain(message.to_vec().into_iter().map(|x| x.into()))
                    .collect::<Vec<_>>();
                self.0.receive(builder, message, enabled);
            }
        }
    };
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct GkrModuleMessage<T> {
    pub tidx: T,
    pub n_logup: T,
    pub n_max: T,
    pub is_n_max_greater: T,
}

define_typed_per_proof_permutation_bus!(GkrModuleBus, GkrModuleMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FractionFolderInputMessage<T> {
    pub num_present_airs: T,
}

define_typed_per_proof_permutation_bus!(FractionFolderInputBus, FractionFolderInputMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ExpressionClaimNMaxMessage<T> {
    pub n_max: T,
}

define_typed_per_proof_permutation_bus!(ExpressionClaimNMaxBus, ExpressionClaimNMaxMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NLiftMessage<T> {
    pub air_idx: T,
    pub n_lift: T,
}

define_typed_per_proof_permutation_bus!(NLiftBus, NLiftMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct BatchConstraintModuleMessage<T> {
    pub tidx: T,
    pub gkr_input_layer_claim: [[T; D_EF]; 2],
}

define_typed_per_proof_permutation_bus!(BatchConstraintModuleBus, BatchConstraintModuleMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct StackingModuleMessage<T> {
    pub tidx: T,
}

define_typed_per_proof_permutation_bus!(StackingModuleBus, StackingModuleMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirModuleMessage<T> {
    /// The `tidx` _after_ batching randomness `mu` is sampled.
    pub tidx: T,
    /// The reduced opening claim after batching.
    pub claim: [T; 4],
}

define_typed_per_proof_permutation_bus!(WhirModuleBus, WhirModuleMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirMuMessage<T> {
    /// The batching randomness to combine stacking claims.
    pub mu: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(WhirMuBus, WhirMuMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct TranscriptBusMessage<T> {
    pub tidx: T,
    pub value: T,
    pub is_sample: T,
}

define_typed_per_proof_permutation_bus!(TranscriptBus, TranscriptBusMessage);

impl TranscriptBus {
    pub fn observe<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        proof_idx: impl Into<AB::Expr>,
        tidx: impl Into<AB::Expr>,
        value: impl Into<AB::Expr>,
        is_enabled: impl Into<AB::Expr>,
    ) {
        let proof_idx = proof_idx.into();
        self.receive(
            builder,
            proof_idx,
            TranscriptBusMessage {
                tidx: tidx.into(),
                value: value.into(),
                is_sample: AB::Expr::ZERO,
            },
            is_enabled,
        )
    }

    pub fn observe_ext<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        proof_idx: impl Into<AB::Expr>,
        tidx: impl Into<AB::Expr>,
        value: [impl Into<AB::Expr>; D_EF],
        is_enabled: impl Into<AB::Expr>,
    ) {
        let proof_idx = proof_idx.into();
        let tidx = tidx.into();
        let is_enabled = is_enabled.into();
        for (i, x) in value.into_iter().enumerate() {
            self.receive(
                builder,
                proof_idx.clone(),
                TranscriptBusMessage {
                    tidx: tidx.clone() + AB::Expr::from_usize(i),
                    value: x.into(),
                    is_sample: AB::Expr::ZERO,
                },
                is_enabled.clone(),
            )
        }
    }

    pub fn observe_commit<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        proof_idx: impl Into<AB::Expr>,
        tidx: impl Into<AB::Expr>,
        commit: [impl Into<AB::Expr>; DIGEST_SIZE],
        is_enabled: impl Into<AB::Expr>,
    ) {
        let proof_idx = proof_idx.into();
        let tidx = tidx.into();
        let is_enabled = is_enabled.into();
        for (i, x) in commit.into_iter().enumerate() {
            self.receive(
                builder,
                proof_idx.clone(),
                TranscriptBusMessage {
                    tidx: tidx.clone() + AB::Expr::from_usize(i),
                    value: x.into(),
                    is_sample: AB::Expr::ZERO,
                },
                is_enabled.clone(),
            )
        }
    }

    pub fn sample<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        proof_idx: impl Into<AB::Expr>,
        tidx: impl Into<AB::Expr>,
        value: impl Into<AB::Expr>,
        is_enabled: impl Into<AB::Expr>,
    ) {
        let proof_idx = proof_idx.into();
        self.receive(
            builder,
            proof_idx,
            TranscriptBusMessage {
                tidx: tidx.into(),
                value: value.into(),
                is_sample: AB::Expr::ONE,
            },
            is_enabled,
        )
    }

    pub fn sample_ext<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        proof_idx: impl Into<AB::Expr>,
        tidx: impl Into<AB::Expr>,
        value: [impl Into<AB::Expr>; D_EF],
        is_enabled: impl Into<AB::Expr>,
    ) {
        let proof_idx = proof_idx.into();
        let tidx = tidx.into();
        let is_enabled = is_enabled.into();
        for (i, x) in value.into_iter().enumerate() {
            self.receive(
                builder,
                proof_idx.clone(),
                TranscriptBusMessage {
                    tidx: tidx.clone() + AB::Expr::from_usize(i),
                    value: x.into(),
                    is_sample: AB::Expr::ONE,
                },
                is_enabled.clone(),
            )
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct Poseidon2PermuteMessage<T> {
    pub input: [T; POSEIDON2_WIDTH],
    pub output: [T; POSEIDON2_WIDTH],
}

define_typed_lookup_bus!(Poseidon2PermuteBus, Poseidon2PermuteMessage);

impl Poseidon2PermuteBus {
    #[must_use]
    pub const fn index(&self) -> openvm_stark_backend::interaction::BusIndex {
        self.0.index
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct Poseidon2CompressMessage<T> {
    pub input: [T; POSEIDON2_WIDTH],
    pub output: [T; DIGEST_SIZE],
}

define_typed_lookup_bus!(Poseidon2CompressBus, Poseidon2CompressMessage);

impl Poseidon2CompressBus {
    #[must_use]
    pub const fn index(&self) -> openvm_stark_backend::interaction::BusIndex {
        self.0.index
    }
}

#[repr(u8)]
#[derive(Debug, Copy, Clone)]
pub(crate) enum AirShapeProperty {
    AirId,
    NumInteractions,
    NeedRot,
}

impl AirShapeProperty {
    pub fn to_field<T: PrimeCharacteristicRing>(self) -> T {
        T::from_u8(self as u8)
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct MerkleVerifyBusMessage<T> {
    /// The idx of the merkle proof in the proof, might have additional bits (so not 0 at the root)
    /// It will be the same for all the rows in the hashing leaves part.
    pub merkle_idx_bit_src: T,
    /// Merkle idx suffix after shifting right max(0, height - k) bits
    pub current_idx_bit_src: T,
    /// The total depth of the merkle proof including the leaves part, equal to merkle_proof.len()
    /// + 1 + k
    pub total_depth: T,
    /// The height of this value, [0, k) are for the hashing leaves part, [k, total_depth) are for
    /// the merkle proof part.
    pub height: T,
    /// Boolean value that indicates if this message is for the hashing leaves or Merkle proof part
    pub is_leaf: T,
    /// For the leaves, it will be 0 ~ 2^k - 1, for the next intermediate values, it will be 0 ~
    /// 2^{k-1} - 1 0 for merkle proof part.
    pub leaf_sub_idx: T,
    /// Either the leaf hash, or the intermediate hash, or the sibling hash
    pub value: [T; DIGEST_SIZE],

    pub commit_major: T,
    pub commit_minor: T,
}

define_typed_per_proof_permutation_bus!(MerkleVerifyBus, MerkleVerifyBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct AirShapeBusMessage<T> {
    pub sort_idx: T,
    /// The property this message encodes.
    /// See associated enum `AirShapeProperty`.
    pub property_idx: T,
    /// The value of the corresponding property.
    pub value: T,
}

define_typed_per_proof_lookup_bus!(AirShapeBus, AirShapeBusMessage);

impl AirShapeBus {
    /// Typed lookup for the proof-shape `(sorted index -> AIR id)` export.
    ///
    /// Keeping the property discriminator here prevents enclosing recursive
    /// circuits from depending on the crate-private enum or duplicating its
    /// numeric representation.
    pub fn lookup_air_id<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        proof_idx: impl Into<AB::Expr>,
        sort_idx: impl Into<AB::Expr>,
        air_idx: impl Into<AB::Expr>,
        enabled: impl Into<AB::Expr>,
    ) {
        self.lookup_key(
            builder,
            proof_idx,
            AirShapeBusMessage {
                sort_idx: sort_idx.into(),
                property_idx: AB::Expr::from_u8(AirShapeProperty::AirId as u8),
                value: air_idx.into(),
            },
            enabled,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct AirPresenceBusMessage<T> {
    pub air_idx: T,
    pub is_present: T,
}

define_typed_per_proof_lookup_bus!(AirPresenceBus, AirPresenceBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct HyperdimBusMessage<T> {
    pub sort_idx: T,
    /// Sender constrains this is `abs(log_height - l_skip)`.
    pub n_abs: T,
    /// Sender constrains this is `n < 0 ? 1 : 0`.
    pub n_sign_bit: T,
}

define_typed_per_proof_lookup_bus!(HyperdimBus, HyperdimBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct LiftedHeightsBusMessage<T> {
    pub sort_idx: T,
    pub part_idx: T,
    pub commit_idx: T,
    pub hypercube_dim: T,
    /// Sender must constraint this equals `2^log_lifted_height`.
    pub lifted_height: T,
    /// Sender must constrain this equals `max(log_height, l_skip)`.
    pub log_lifted_height: T,
}

define_typed_per_proof_lookup_bus!(LiftedHeightsBus, LiftedHeightsBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct StackingIndexMessage<T> {
    pub commit_idx: T,
    pub col_idx: T,
}

define_typed_per_proof_lookup_bus!(StackingIndicesBus, StackingIndexMessage);

/// Carries all commitments in the proof.
///
/// The stacking commitments have `major_idx = 0` and `minor_idx =
/// stacking_matrix_idx`. The WHIR commitments have `major_idx = whir_round + 1`
/// and `minor_idx = 0`.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct CommitmentsBusMessage<T> {
    pub major_idx: T,
    pub minor_idx: T,
    pub commitment: [T; DIGEST_SIZE],
}

define_typed_per_proof_lookup_bus!(CommitmentsBus, CommitmentsBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct CachedCommitBusMessage<T> {
    pub air_idx: T,
    pub cached_idx: T,
    pub global_cached_idx: T,
    pub cached_commit: [T; DIGEST_SIZE],
}

define_typed_per_proof_permutation_bus!(CachedCommitBus, CachedCommitBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FinalTranscriptStateMessage<T> {
    pub state: [T; POSEIDON2_WIDTH],
}

define_typed_per_proof_permutation_bus!(FinalTranscriptStateBus, FinalTranscriptStateMessage);

/// A row-aligned intermediate transcript state exposed to a companion AIR.
///
/// Unlike the final/resume buses, this carries the absolute operation index
/// and the number of samples in the selected transcript row.  A checkpoint
/// selected after a squeeze therefore determines the complete duplex cursor:
/// no prover-supplied `absorb_idx` or `sample_idx` is trusted.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct CertifiedTranscriptCheckpointMessage<T> {
    pub kind: T,
    pub tidx: T,
    pub sample_count: T,
    pub state: [T; POSEIDON2_WIDTH],
}

define_typed_per_proof_permutation_bus!(
    CertifiedTranscriptCheckpointBus,
    CertifiedTranscriptCheckpointMessage
);

/// The duplex sponge state a transcript proof resumes from.
///
/// A proof normally starts at the canonical zero sponge. A recursive history
/// stage instead continues its child's transcript, and receives the child's
/// final state here rather than replaying every earlier stage's operations to
/// re-derive it -- which is what made the circuit's hashing quadratic in the
/// stage count.
///
/// The whole state travels, not just the lanes the resumed proof preserves. The
/// sender cannot know which rate lanes the first row overwrites with its own
/// operands, so it hands over everything and the receiver compares only the
/// lanes it is able to see. `tidx` travels with it so the resumed rows keep
/// absolute operation indices and every consumer's lookup stays where it was.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ResumeTranscriptStateMessage<T> {
    pub tidx: T,
    pub state: [T; POSEIDON2_WIDTH],
}

define_typed_per_proof_permutation_bus!(ResumeTranscriptStateBus, ResumeTranscriptStateMessage);

/// The operation index one past a transcript proof's last operation.
///
/// Separate from the final-state message because that one also feeds the step
/// binding, whose preimage layout and public values must not move.
///
/// Handing a sponge state from one proof to another must not also hand over a
/// chosen *length*. Without this the resume index would be a free witness: a
/// prover could end the prefix early or late and, as long as every
/// schedule-derived index downstream shifted by the same amount, the
/// permutation buses would still balance.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct TranscriptEndIndexMessage<T> {
    pub tidx: T,
}

define_typed_per_proof_permutation_bus!(TranscriptEndIndexBus, TranscriptEndIndexMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct PreHashMessage<T> {
    pub vk_pre_hash: [T; DIGEST_SIZE],
}

define_typed_per_proof_permutation_bus!(PreHashBus, PreHashMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct XiRandomnessMessage<T> {
    pub idx: T,
    pub xi: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(XiRandomnessBus, XiRandomnessMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct SelHypercubeBusMessage<T> {
    pub n: T,
    pub is_first: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_lookup_bus!(SelHypercubeBus, SelHypercubeBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct SelUniBusMessage<T> {
    pub n: T,
    pub is_first: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_lookup_bus!(SelUniBus, SelUniBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintSumcheckRandomness<T> {
    pub idx: T,
    pub challenge: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(
    ConstraintSumcheckRandomnessBus,
    ConstraintSumcheckRandomness
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ColumnClaimsMessage<T> {
    // pub idx: T,
    pub sort_idx: T,
    pub part_idx: T,
    pub col_idx: T,
    pub claim: [T; D_EF],
    pub is_rot: T,
}

define_typed_per_proof_permutation_bus!(ColumnClaimsBus, ColumnClaimsMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct WhirOpeningPointMessage<T> {
    pub idx: T,
    pub value: [T; D_EF],
}

// Permutation bus for opening points produced by the stacking sumcheck and eq_base AIRs.
// Each point is sent exactly once by the producer and received exactly once by the consumer
// (the first node in each layer of the MLE evaluation tree).
define_typed_per_proof_permutation_bus!(WhirOpeningPointBus, WhirOpeningPointMessage);
// Lookup bus for distributing opening points within the MLE evaluation tree. The first node
// in each layer registers the point (received via WhirOpeningPointBus) as a lookup key,
// and the remaining nodes in the layer look it up.
define_typed_per_proof_lookup_bus!(WhirOpeningPointLookupBus, WhirOpeningPointMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct PublicValuesBusMessage<T> {
    pub air_idx: T,
    pub pv_idx: T,
    pub value: T,
}

define_typed_per_proof_permutation_bus!(PublicValuesBus, PublicValuesBusMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqNegResultMessage<T> {
    // hypercube dimension where n < 0
    pub n: T,
    // 2^{l_skip + n} * eq_n(u_0, r_0)
    pub eq: [T; D_EF],
    // 2^{l_skip + n} * k_rot_n(u_0, r_0)
    pub k_rot: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(EqNegResultBus, EqNegResultMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqNegBaseRandMessage<T> {
    // sampled value u_0
    pub u: [T; D_EF],
    // sampled value r_0
    pub r: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(EqNegBaseRandBus, EqNegBaseRandMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct EqNsNLogupMaxMessage<T> {
    pub n_logup: T,
    pub n_max: T,
}

define_typed_per_proof_lookup_bus!(EqNsNLogupMaxBus, EqNsNLogupMaxMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct Eq3bShapeMessage<T> {
    pub sort_idx: T,
    pub n_lift: T,
    pub n_logup: T,
    pub num_interactions: T,
}

define_typed_per_proof_lookup_bus!(Eq3bShapeBus, Eq3bShapeMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ConstraintsFoldingInputMessage<T> {
    pub tidx: T,
}

define_typed_per_proof_permutation_bus!(ConstraintsFoldingInputBus, ConstraintsFoldingInputMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct InteractionsFoldingInputMessage<T> {
    pub tidx: T,
}

define_typed_per_proof_permutation_bus!(
    InteractionsFoldingInputBus,
    InteractionsFoldingInputMessage
);
