use openvm_stark_backend::p3_field::{PrimeCharacteristicRing, PrimeField32};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, F};
use openvm_verify_stark_host::pvs::{VerifierBasePvs, VkCommit, VmPvs};
use serde::{Deserialize, Serialize};

/// Public protocol version of the fixed-capacity recursive boundary.
pub const FINITE_WARP_V3_PROTOCOL_VERSION: u32 = 3;
/// Maximum number of complete fixed-PESAT sources accepted by one wrapper.
pub const FINITE_WARP_V3_MAX_SOURCES: u32 = 128;
/// Maximum number of ordinary WARP invocations accepted by one wrapper.
pub const FINITE_WARP_V3_MAX_CALLS: usize = 3;
/// Largest ordinary WARP input arity accepted by this wrapper profile.
pub const FINITE_WARP_V3_MAX_INPUT_ARITY: u32 = 64;

pub const FINITE_WARP_V3_VERIFIER_PVS_AIR_ID: usize = 0;
pub const FINITE_WARP_V3_VM_PVS_AIR_ID: usize = 1;
pub const FINITE_WARP_V3_STATEMENT_AIR_ID: usize = 2;

/// Fixed serialized statement width. Counts use one canonical BabyBear
/// element; the statement AIR range-constrains them to the bounds above. The
/// statement is private in the final wrapper proof and is linked to the public
/// `VmPvs` through the execution bus.
pub const FINITE_WARP_V3_PUBLIC_STATEMENT_WIDTH: usize = 101;

pub(crate) const PROTOCOL_VERSION: usize = 0;
pub(crate) const PROTOCOL_DIGEST: core::ops::Range<usize> = 1..9;
pub(crate) const RELATION_DIGEST: core::ops::Range<usize> = 9..17;
pub(crate) const WARP_INDEX_DIGEST: core::ops::Range<usize> = 17..25;
pub(crate) const TERMINAL_INDEX_DIGEST: core::ops::Range<usize> = 25..33;
pub(crate) const COMPONENT_DIGEST: core::ops::Range<usize> = 33..41;
pub(crate) const SCHEDULE_DIGEST: core::ops::Range<usize> = 41..49;
pub(crate) const MANIFEST_DIGEST: core::ops::Range<usize> = 49..57;
pub(crate) const SOURCE_COUNT: usize = 57;
pub(crate) const CALL_COUNT: usize = 58;
pub(crate) const PROGRAM_COMMITMENT: core::ops::Range<usize> = 59..67;
pub(crate) const INITIAL_PC: usize = 67;
pub(crate) const INITIAL_ROOT: core::ops::Range<usize> = 68..76;
pub(crate) const FINAL_PC: usize = 76;
pub(crate) const FINAL_ROOT: core::ops::Range<usize> = 77..85;
pub(crate) const FINAL_ACCUMULATOR_DIGEST: core::ops::Range<usize> = 85..93;
pub(crate) const FINAL_ACCUMULATOR_ROOT: core::ops::Range<usize> = 93..101;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FiniteWarpV3Binding {
    pub protocol_version: u32,
    pub protocol_digest: Digest,
    pub relation_digest: Digest,
    pub warp_index_digest: Digest,
    pub terminal_index_digest: Digest,
    /// Digest of the exact manifest/transition/terminal verifier AIR set.
    pub verifier_component_digest: Digest,
    /// Existing recursive identity propagated as the app VK commitment by the
    /// WARP verifier relation and protocol binding.
    pub source_app_vk_commit: VkCommit<F>,
    /// The same source verifier cached table re-committed with this wrapper's
    /// PCS parameters. This is the key lineage propagated by the wrapper's
    /// leaf-compatible `VerifierBasePvs`, exactly as an ordinary OpenVM
    /// recursive parent re-commits its child VK table.
    pub recursive_app_vk_commit: VkCommit<F>,
}

impl FiniteWarpV3Binding {
    pub fn validate(&self) -> Result<(), FiniteWarpV3Error> {
        if self.protocol_version != FINITE_WARP_V3_PROTOCOL_VERSION {
            return Err(FiniteWarpV3Error::ProtocolVersion {
                expected: FINITE_WARP_V3_PROTOCOL_VERSION,
                actual: self.protocol_version,
            });
        }
        for (kind, digest) in [
            (FiniteWarpV3DigestKind::Protocol, self.protocol_digest),
            (FiniteWarpV3DigestKind::Relation, self.relation_digest),
            (FiniteWarpV3DigestKind::WarpIndex, self.warp_index_digest),
            (
                FiniteWarpV3DigestKind::TerminalIndex,
                self.terminal_index_digest,
            ),
            (
                FiniteWarpV3DigestKind::VerifierComponent,
                self.verifier_component_digest,
            ),
        ] {
            if is_zero_digest(digest) {
                return Err(FiniteWarpV3Error::UnsetDigest(kind));
            }
        }
        if is_zero_digest(self.source_app_vk_commit.cached_commit)
            && is_zero_digest(self.source_app_vk_commit.vk_pre_hash)
        {
            return Err(FiniteWarpV3Error::UnsetSourceAppVk);
        }
        if is_zero_digest(self.recursive_app_vk_commit.cached_commit)
            || self.recursive_app_vk_commit.vk_pre_hash != self.source_app_vk_commit.vk_pre_hash
        {
            return Err(FiniteWarpV3Error::UnsetSourceAppVk);
        }
        Ok(())
    }

    #[must_use]
    pub fn verifier_pvs(&self) -> VerifierBasePvs<F> {
        let unset = VkCommit {
            cached_commit: [F::ZERO; DIGEST_SIZE],
            vk_pre_hash: [F::ZERO; DIGEST_SIZE],
        };
        VerifierBasePvs {
            internal_flag: F::ZERO,
            app_vk_commit: self.recursive_app_vk_commit,
            leaf_vk_commit: unset,
            internal_for_leaf_vk_commit: unset,
            recursion_depth: F::ZERO,
            internal_recursive_vk_commit: unset,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FiniteWarpV3PublicStatement {
    pub protocol_version: u32,
    pub protocol_digest: Digest,
    pub relation_digest: Digest,
    pub warp_index_digest: Digest,
    pub terminal_index_digest: Digest,
    pub verifier_component_digest: Digest,
    pub schedule_digest: Digest,
    pub manifest_digest: Digest,
    pub source_count: u32,
    pub call_count: u32,
    pub program_commitment: Digest,
    pub initial_pc: F,
    pub initial_root: Digest,
    pub final_pc: F,
    pub final_root: Digest,
    pub final_accumulator_digest: Digest,
    pub final_accumulator_root: Digest,
}

impl FiniteWarpV3PublicStatement {
    pub fn validate(&self, binding: &FiniteWarpV3Binding) -> Result<(), FiniteWarpV3Error> {
        binding.validate()?;
        if self.protocol_version != FINITE_WARP_V3_PROTOCOL_VERSION {
            return Err(FiniteWarpV3Error::ProtocolVersion {
                expected: FINITE_WARP_V3_PROTOCOL_VERSION,
                actual: self.protocol_version,
            });
        }
        for (kind, expected, actual) in [
            (
                FiniteWarpV3DigestKind::Protocol,
                binding.protocol_digest,
                self.protocol_digest,
            ),
            (
                FiniteWarpV3DigestKind::Relation,
                binding.relation_digest,
                self.relation_digest,
            ),
            (
                FiniteWarpV3DigestKind::WarpIndex,
                binding.warp_index_digest,
                self.warp_index_digest,
            ),
            (
                FiniteWarpV3DigestKind::TerminalIndex,
                binding.terminal_index_digest,
                self.terminal_index_digest,
            ),
            (
                FiniteWarpV3DigestKind::VerifierComponent,
                binding.verifier_component_digest,
                self.verifier_component_digest,
            ),
        ] {
            if expected != actual {
                return Err(FiniteWarpV3Error::BindingDigest(kind));
            }
        }
        if self.source_count < 2 || self.source_count > FINITE_WARP_V3_MAX_SOURCES {
            return Err(FiniteWarpV3Error::SourceCount(self.source_count));
        }
        if self.call_count == 0 || self.call_count as usize > FINITE_WARP_V3_MAX_CALLS {
            return Err(FiniteWarpV3Error::CallCount(self.call_count));
        }
        if is_zero_digest(self.schedule_digest) || is_zero_digest(self.manifest_digest) {
            return Err(FiniteWarpV3Error::UnsetDigest(
                FiniteWarpV3DigestKind::Manifest,
            ));
        }
        Ok(())
    }

    #[must_use]
    pub fn vm_pvs(&self) -> VmPvs<F> {
        VmPvs {
            program_commit: self.program_commitment,
            initial_pc: self.initial_pc,
            final_pc: self.final_pc,
            exit_code: F::ZERO,
            is_terminate: F::ONE,
            initial_root: self.initial_root,
            final_root: self.final_root,
        }
    }

    #[must_use]
    pub fn to_fields(&self) -> [F; FINITE_WARP_V3_PUBLIC_STATEMENT_WIDTH] {
        let mut fields = [F::ZERO; FINITE_WARP_V3_PUBLIC_STATEMENT_WIDTH];
        fields[PROTOCOL_VERSION] = F::from_u32(self.protocol_version);
        fields[PROTOCOL_DIGEST].copy_from_slice(&self.protocol_digest);
        fields[RELATION_DIGEST].copy_from_slice(&self.relation_digest);
        fields[WARP_INDEX_DIGEST].copy_from_slice(&self.warp_index_digest);
        fields[TERMINAL_INDEX_DIGEST].copy_from_slice(&self.terminal_index_digest);
        fields[COMPONENT_DIGEST].copy_from_slice(&self.verifier_component_digest);
        fields[SCHEDULE_DIGEST].copy_from_slice(&self.schedule_digest);
        fields[MANIFEST_DIGEST].copy_from_slice(&self.manifest_digest);
        fields[SOURCE_COUNT] = F::from_u32(self.source_count);
        fields[CALL_COUNT] = F::from_u32(self.call_count);
        fields[PROGRAM_COMMITMENT].copy_from_slice(&self.program_commitment);
        fields[INITIAL_PC] = self.initial_pc;
        fields[INITIAL_ROOT].copy_from_slice(&self.initial_root);
        fields[FINAL_PC] = self.final_pc;
        fields[FINAL_ROOT].copy_from_slice(&self.final_root);
        fields[FINAL_ACCUMULATOR_DIGEST].copy_from_slice(&self.final_accumulator_digest);
        fields[FINAL_ACCUMULATOR_ROOT].copy_from_slice(&self.final_accumulator_root);
        fields
    }

    pub fn try_from_slice(values: &[F]) -> Result<Self, FiniteWarpV3Error> {
        if values.len() != FINITE_WARP_V3_PUBLIC_STATEMENT_WIDTH {
            return Err(FiniteWarpV3Error::PublicValuesWidth {
                expected: FINITE_WARP_V3_PUBLIC_STATEMENT_WIDTH,
                actual: values.len(),
            });
        }
        let digest = |range: core::ops::Range<usize>| -> Digest {
            core::array::from_fn(|index| values[range.start + index])
        };
        Ok(Self {
            protocol_version: values[PROTOCOL_VERSION].as_canonical_u32(),
            protocol_digest: digest(PROTOCOL_DIGEST),
            relation_digest: digest(RELATION_DIGEST),
            warp_index_digest: digest(WARP_INDEX_DIGEST),
            terminal_index_digest: digest(TERMINAL_INDEX_DIGEST),
            verifier_component_digest: digest(COMPONENT_DIGEST),
            schedule_digest: digest(SCHEDULE_DIGEST),
            manifest_digest: digest(MANIFEST_DIGEST),
            source_count: values[SOURCE_COUNT].as_canonical_u32(),
            call_count: values[CALL_COUNT].as_canonical_u32(),
            program_commitment: digest(PROGRAM_COMMITMENT),
            initial_pc: values[INITIAL_PC],
            initial_root: digest(INITIAL_ROOT),
            final_pc: values[FINAL_PC],
            final_root: digest(FINAL_ROOT),
            final_accumulator_digest: digest(FINAL_ACCUMULATOR_DIGEST),
            final_accumulator_root: digest(FINAL_ACCUMULATOR_ROOT),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FiniteWarpV3CallReceipt {
    pub active: bool,
    pub call_index: u32,
    pub source_start: u32,
    pub source_count: u32,
    pub input_arity: u32,
    pub fresh_stacked_root: Digest,
    pub prior_accumulator_digest: Digest,
    pub output_accumulator_digest: Digest,
}

impl FiniteWarpV3CallReceipt {
    #[must_use]
    pub const fn inactive() -> Self {
        Self {
            active: false,
            call_index: 0,
            source_start: 0,
            source_count: 0,
            input_arity: 0,
            fresh_stacked_root: [F::ZERO; DIGEST_SIZE],
            prior_accumulator_digest: [F::ZERO; DIGEST_SIZE],
            output_accumulator_digest: [F::ZERO; DIGEST_SIZE],
        }
    }

    fn is_canonical_inactive(&self) -> bool {
        self == &Self::inactive()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FiniteWarpV3TerminalReceipt {
    pub final_accumulator_digest: Digest,
    pub final_accumulator_root: Digest,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FiniteWarpV3Record {
    pub statement: FiniteWarpV3PublicStatement,
    pub calls: [FiniteWarpV3CallReceipt; FINITE_WARP_V3_MAX_CALLS],
    pub terminal: FiniteWarpV3TerminalReceipt,
}

impl FiniteWarpV3Record {
    pub fn validate(&self, binding: &FiniteWarpV3Binding) -> Result<(), FiniteWarpV3Error> {
        self.statement.validate(binding)?;
        let mut active_count = 0u32;
        let mut source_start = 0u32;
        let mut previous_output = [F::ZERO; DIGEST_SIZE];
        let mut inactive_seen = false;
        for (position, call) in self.calls.iter().enumerate() {
            if !call.active {
                inactive_seen = true;
                if !call.is_canonical_inactive() {
                    return Err(FiniteWarpV3Error::NonCanonicalInactiveCall(position));
                }
                continue;
            }
            if inactive_seen {
                return Err(FiniteWarpV3Error::NonPrefixCalls);
            }
            active_count += 1;
            if call.call_index != position as u32 {
                return Err(FiniteWarpV3Error::CallIndex {
                    position,
                    actual: call.call_index,
                });
            }
            if call.source_start != source_start || call.source_count == 0 {
                return Err(FiniteWarpV3Error::SourceRange(position));
            }
            let prior_count = u32::from(position != 0);
            if call.input_arity != call.source_count + prior_count
                || call.input_arity < 2
                || call.input_arity > FINITE_WARP_V3_MAX_INPUT_ARITY
                || !call.input_arity.is_power_of_two()
            {
                return Err(FiniteWarpV3Error::InputArity(position));
            }
            if call.prior_accumulator_digest != previous_output {
                return Err(FiniteWarpV3Error::AccumulatorChain(position));
            }
            source_start = source_start
                .checked_add(call.source_count)
                .ok_or(FiniteWarpV3Error::SourceRange(position))?;
            previous_output = call.output_accumulator_digest;
        }
        if active_count != self.statement.call_count {
            return Err(FiniteWarpV3Error::CallCount(active_count));
        }
        if source_start != self.statement.source_count {
            return Err(FiniteWarpV3Error::SourceCount(source_start));
        }
        if previous_output != self.statement.final_accumulator_digest {
            return Err(FiniteWarpV3Error::FinalAccumulatorDigest);
        }
        if self.terminal.final_accumulator_digest != self.statement.final_accumulator_digest
            || self.terminal.final_accumulator_root != self.statement.final_accumulator_root
        {
            return Err(FiniteWarpV3Error::TerminalLink);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiniteWarpV3DigestKind {
    Protocol,
    Relation,
    WarpIndex,
    TerminalIndex,
    VerifierComponent,
    Manifest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FiniteWarpV3Error {
    ProtocolVersion { expected: u32, actual: u32 },
    UnsetDigest(FiniteWarpV3DigestKind),
    BindingDigest(FiniteWarpV3DigestKind),
    UnsetSourceAppVk,
    PublicValuesWidth { expected: usize, actual: usize },
    SourceCount(u32),
    CallCount(u32),
    NonCanonicalInactiveCall(usize),
    NonPrefixCalls,
    CallIndex { position: usize, actual: u32 },
    SourceRange(usize),
    InputArity(usize),
    AccumulatorChain(usize),
    FinalAccumulatorDigest,
    TerminalLink,
    VerifierComponentDigest,
    MissingVerifierComponents,
}

#[must_use]
pub(crate) fn is_zero_digest(digest: Digest) -> bool {
    digest == [F::ZERO; DIGEST_SIZE]
}

pub(crate) mod offsets {
    pub(crate) use super::{
        CALL_COUNT, COMPONENT_DIGEST, FINAL_ACCUMULATOR_DIGEST, FINAL_ACCUMULATOR_ROOT, FINAL_PC,
        FINAL_ROOT, INITIAL_PC, INITIAL_ROOT, MANIFEST_DIGEST, PROGRAM_COMMITMENT, PROTOCOL_DIGEST,
        PROTOCOL_VERSION, RELATION_DIGEST, SCHEDULE_DIGEST, SOURCE_COUNT, TERMINAL_INDEX_DIGEST,
        WARP_INDEX_DIGEST,
    };
}

const _: () = assert!(DIGEST_SIZE == 8);
const _: () = assert!(FINAL_ACCUMULATOR_ROOT.end == FINITE_WARP_V3_PUBLIC_STATEMENT_WIDTH);
