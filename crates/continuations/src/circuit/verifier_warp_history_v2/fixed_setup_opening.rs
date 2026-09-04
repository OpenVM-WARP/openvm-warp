//! Setup-owned SWIRL opening schedule and compatibility messages.
//!
//! Production code retains only the verifier-key-owned matrix/claim schedule
//! consumed by the V3 setup-PCS authority path, plus the narrow source-point
//! and History-certificate compatibility buses.  The retired C1 table
//! evaluator is kept below solely as a `#[cfg(test)]` differential oracle.
//!
//! In production, setup-opening authority comes from ordinary retained setup
//! PCS commitments, ordered stacking, and multi-WHIR.  Nothing in this module
//! evaluates a setup table or certifies a proof-carried opening value.

#[cfg(test)]
use core::borrow::{Borrow, BorrowMut};
use std::{collections::BTreeMap, sync::Arc};

#[cfg(test)]
use openvm_circuit_primitives::{utils::assert_array_eq, StructReflection, StructReflectionHelper};
#[cfg(test)]
use openvm_recursion_circuit::bus::{
    ColumnClaimsBus, ColumnClaimsMessage, Poseidon2CompressBus, Poseidon2CompressMessage,
};
use openvm_recursion_circuit::define_typed_permutation_bus;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::interaction::{BusIndex, InteractionBuilder, LookupBus};
#[cfg(test)]
use openvm_stark_backend::{
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::{
        extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing,
        TwoAdicField,
    },
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
#[cfg(test)]
use openvm_stark_sdk::config::baby_bear_poseidon2::{poseidon2_compress_with_capacity, EF, F};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, D_EF};

/// Domain separator retained by the narrow V2-compatible certificate bus.
pub const FIXED_SETUP_OPENING_CERTIFICATE_TAG_V2: u32 = 0x4653_4f32;
/// Protocol version independently bound by the certificate bus.
pub const FIXED_SETUP_OPENING_PROTOCOL_V2: u32 = 3;

/// Schedule metadata for one setup-owned fixed matrix.
///
/// Production retains only identity and shape. The actual matrix is owned by
/// the retained setup PCS; row-major values exist here only for the retired
/// test oracle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedSetupMatrixV2 {
    /// Stable index into the fixed multi-AIR setup-trace catalog.
    pub setup_index: u32,
    pub air_id: u32,
    /// Digest of the exact direct-AIR relation containing this matrix.
    pub relation_digest: Digest,
    pub width: usize,
    pub height: usize,
    #[cfg(test)]
    pub values: Arc<[F]>,
}

impl FixedSetupMatrixV2 {
    fn validate(&self) -> Result<(), FixedSetupOpeningErrorV2> {
        if self.width == 0 || self.height == 0 || !self.height.is_power_of_two() {
            return Err(FixedSetupOpeningErrorV2::InvalidMatrix(self.air_id));
        }
        #[cfg(test)]
        if self.values.len() != self.width.saturating_mul(self.height) {
            return Err(FixedSetupOpeningErrorV2::InvalidMatrix(self.air_id));
        }
        Ok(())
    }

    #[cfg(test)]
    fn value(&self, row: usize, column: usize) -> F {
        self.values[row * self.width + column]
    }
}

/// One active occurrence of a setup matrix in one complete SWIRL proof.
///
/// Instances are setup/profile data. The constructor requires strict
/// `(proof_index, sort_idx, air_id)` order and expands every column exactly
/// once, with a rotated claim iff `need_rot` is set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedSetupOpeningInstanceV2 {
    pub proof_index: u32,
    pub matrix_index: usize,
    pub sort_idx: u32,
    /// SWIRL proof part index. For the ordinary common/preprocessed/cached
    /// ordering this is one, but it remains explicit and setup-bound.
    pub part_idx: u32,
    pub l_skip: usize,
    pub log_height: usize,
    pub need_rot: bool,
}

/// One verifier-produced opening pair supplied to the trace generator.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedSetupOpeningClaimRecordV2 {
    pub setup_index: u32,
    pub air_id: u32,
    pub sort_idx: u32,
    pub part_idx: u32,
    pub col_idx: u32,
    pub current: EF,
    pub rotated: Option<EF>,
}

/// All fixed openings belonging to one complete SWIRL proof.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedSetupOpeningProofRecordV2 {
    pub proof_index: u32,
    /// Exact `r_0, ..., r_n` exported by the SWIRL batch-constraint verifier.
    pub opening_point: Vec<EF>,
    /// Canonical setup order, one entry per fixed column.
    pub claims: Vec<FixedSetupOpeningClaimRecordV2>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedSetupOpeningErrorV2 {
    EmptyProfile,
    InvalidMatrix(u32),
    MatrixSetupIndexDuplicate(u32),
    UnusedMatrix(u32),
    InstanceMatrixIndex(usize),
    InstanceHeight(u32),
    InstanceOrder,
    DuplicateInstance(u32, u32),
    DuplicateSortIndex(u32, u32),
    ProofIndexGap(u32),
    ShiftOverflow,
    CountOverflow,
    #[cfg(test)]
    MissingProof(u32),
    #[cfg(test)]
    UnexpectedProof(u32),
    #[cfg(test)]
    OpeningPointLength(u32),
    #[cfg(test)]
    ClaimCount(u32),
    #[cfg(test)]
    ClaimSchedule(u32, usize),
    #[cfg(test)]
    PleDenominator(u32),
    #[cfg(test)]
    InternalPlan,
}

/// Private canonical setup-opening schedule material used by V3 authority.
#[derive(Clone, Debug)]
struct ClaimPlanV2 {
    proof_index: u32,
    claim_ordinal: u32,
    claim_count: u32,
    matrix_index: usize,
    setup_index: u32,
    air_id: u32,
    sort_idx: u32,
    part_idx: u32,
    col_idx: u32,
    l_skip: usize,
    log_height: usize,
    need_rot: bool,
    point_len: usize,
    relation_digest: Digest,
}

/// Public, immutable view of one canonical claim slot. The producer bridge
/// should use this schedule instead of reimplementing proof-part ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedSetupOpeningClaimDescriptorV2 {
    pub proof_index: u32,
    pub claim_ordinal: u32,
    pub claim_count: u32,
    pub matrix_index: usize,
    /// Stable fixed-multi-AIR setup identity; unlike `matrix_index`, this is
    /// part of the certificate and receipt identity.
    pub setup_index: u32,
    pub air_id: u32,
    pub sort_idx: u32,
    pub part_idx: u32,
    pub col_idx: u32,
    pub l_skip: usize,
    pub log_height: usize,
    pub need_rot: bool,
    pub point_len: usize,
    pub relation_digest: Digest,
}

#[cfg(test)]
#[derive(Clone, Debug)]
enum HashBlockPlanV2 {
    Static([F; DIGEST_SIZE]),
    PointPair { first: usize, second: Option<usize> },
    ClaimPair,
}

#[cfg(test)]
#[derive(Clone, Debug)]
enum RowPlanV2 {
    Leaf {
        claim: usize,
        folded_row: usize,
        z: usize,
        skip: usize,
        omega_power: F,
        first: bool,
        last: bool,
        root: bool,
    },
    Fold {
        claim: usize,
        level: usize,
        node: usize,
        root: bool,
    },
    Hash {
        claim: usize,
        block: HashBlockPlanV2,
        first: bool,
        last: bool,
    },
}

/// Verifier-key-owned setup matrix and canonical opening-claim schedule.
///
/// V3 uses this profile only to bind exact setup/column identities into its
/// retained-PCS authority statement. It is not an opening evaluator.
#[derive(Clone, Debug)]
pub struct FixedSetupOpeningProfileV2 {
    pub source_relation_vk_digest: Digest,
    pub matrices: Arc<[FixedSetupMatrixV2]>,
    pub instances: Arc<[FixedSetupOpeningInstanceV2]>,
    claims: Arc<[ClaimPlanV2]>,
    #[cfg(test)]
    rows: Arc<[RowPlanV2]>,
    #[cfg(test)]
    point_demands: Arc<[(u32, u32, u32)]>,
}

/// Exact allocation preflight for the C1 fixed-setup opening AIR.
///
/// `main_trace_bytes` and `preprocessed_trace_bytes` are the uncompressed
/// prover matrices at the power-of-two AIR height. `setup_matrix_bytes` is
/// counted once and is not multiplied by the transition count.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedSetupOpeningPreflightV2 {
    pub transition_count: usize,
    pub setup_matrix_count: usize,
    pub setup_matrix_elements: usize,
    pub setup_matrix_bytes: usize,
    pub max_setup_matrix_height: usize,
    pub claim_count: usize,
    pub logical_row_count: usize,
    pub trace_height: usize,
    pub main_trace_width: usize,
    pub preprocessed_trace_width: usize,
    pub main_trace_bytes: usize,
    pub preprocessed_trace_bytes: usize,
}

#[cfg(test)]
impl FixedSetupOpeningPreflightV2 {
    #[must_use]
    pub fn total_air_trace_bytes(self) -> usize {
        self.main_trace_bytes
            .saturating_add(self.preprocessed_trace_bytes)
    }

    /// Exact production-layout estimate when every transition opens every
    /// setup matrix with one uniform `l_skip`, as History-v2 reconstruction
    /// requires. This does not materialize claim or row plans.
    pub fn for_uniform_transition_profile(
        matrix_shapes: &[(usize, usize)],
        transition_count: usize,
        l_skip: usize,
    ) -> Result<Self, FixedSetupOpeningErrorV2> {
        if matrix_shapes.is_empty() || transition_count == 0 {
            return Err(FixedSetupOpeningErrorV2::EmptyProfile);
        }
        let skip = 1usize
            .checked_shl(l_skip as u32)
            .ok_or(FixedSetupOpeningErrorV2::ShiftOverflow)?;
        let mut rows_per_transition = 0usize;
        let mut claims_per_transition = 0usize;
        let mut setup_matrix_elements = 0usize;
        let mut max_setup_matrix_height = 0usize;
        for &(width, height) in matrix_shapes {
            if width == 0 || height == 0 || !height.is_power_of_two() {
                return Err(FixedSetupOpeningErrorV2::InternalPlan);
            }
            let expanded_height = height.max(skip);
            let folded_height = expanded_height / skip;
            let folded_log = folded_height.ilog2() as usize;
            let point_len = folded_log + 1;
            let hash_rows = 5usize
                .checked_add(point_len.div_ceil(2))
                .ok_or(FixedSetupOpeningErrorV2::CountOverflow)?;
            let rows_per_claim = expanded_height
                .checked_add(folded_height - 1)
                .and_then(|rows| rows.checked_add(hash_rows))
                .ok_or(FixedSetupOpeningErrorV2::CountOverflow)?;
            rows_per_transition = rows_per_transition
                .checked_add(
                    width
                        .checked_mul(rows_per_claim)
                        .ok_or(FixedSetupOpeningErrorV2::CountOverflow)?,
                )
                .ok_or(FixedSetupOpeningErrorV2::CountOverflow)?;
            claims_per_transition = claims_per_transition
                .checked_add(width)
                .ok_or(FixedSetupOpeningErrorV2::CountOverflow)?;
            setup_matrix_elements = setup_matrix_elements
                .checked_add(
                    width
                        .checked_mul(height)
                        .ok_or(FixedSetupOpeningErrorV2::CountOverflow)?,
                )
                .ok_or(FixedSetupOpeningErrorV2::CountOverflow)?;
            max_setup_matrix_height = max_setup_matrix_height.max(height);
        }
        let logical_row_count = rows_per_transition
            .checked_mul(transition_count)
            .ok_or(FixedSetupOpeningErrorV2::CountOverflow)?;
        let trace_height = logical_row_count
            .checked_next_power_of_two()
            .ok_or(FixedSetupOpeningErrorV2::CountOverflow)?
            .max(2);
        let claim_count = claims_per_transition
            .checked_mul(transition_count)
            .ok_or(FixedSetupOpeningErrorV2::CountOverflow)?;
        let main_trace_width = FixedSetupOpeningColsV2::<u8>::width();
        let preprocessed_trace_width = FixedSetupOpeningPrepColsV2::<u8>::width();
        let field_bytes = core::mem::size_of::<F>();
        let bytes = |width: usize| {
            trace_height
                .checked_mul(width)
                .and_then(|cells| cells.checked_mul(field_bytes))
                .ok_or(FixedSetupOpeningErrorV2::CountOverflow)
        };
        Ok(Self {
            transition_count,
            setup_matrix_count: matrix_shapes.len(),
            setup_matrix_elements,
            setup_matrix_bytes: setup_matrix_elements
                .checked_mul(field_bytes)
                .ok_or(FixedSetupOpeningErrorV2::CountOverflow)?,
            max_setup_matrix_height,
            claim_count,
            logical_row_count,
            trace_height,
            main_trace_width,
            preprocessed_trace_width,
            main_trace_bytes: bytes(main_trace_width)?,
            preprocessed_trace_bytes: bytes(preprocessed_trace_width)?,
        })
    }
}

impl FixedSetupOpeningProfileV2 {
    pub fn new(
        source_relation_vk_digest: Digest,
        matrices: Vec<FixedSetupMatrixV2>,
        instances: Vec<FixedSetupOpeningInstanceV2>,
    ) -> Result<Self, FixedSetupOpeningErrorV2> {
        if matrices.is_empty() || instances.is_empty() {
            return Err(FixedSetupOpeningErrorV2::EmptyProfile);
        }
        let mut setup_indices = BTreeMap::new();
        for matrix in &matrices {
            matrix.validate()?;
            if setup_indices.insert(matrix.setup_index, ()).is_some() {
                return Err(FixedSetupOpeningErrorV2::MatrixSetupIndexDuplicate(
                    matrix.setup_index,
                ));
            }
        }

        let mut previous_key = None;
        let mut seen = BTreeMap::new();
        let mut seen_sort_part = BTreeMap::new();
        let mut air_by_sort = BTreeMap::new();
        let mut referenced_matrices = vec![false; matrices.len()];
        let mut expected_proof = 0u32;
        let mut claims = Vec::new();
        let mut claim_count_by_proof = BTreeMap::<u32, u32>::new();
        for instance in &instances {
            let matrix = matrices.get(instance.matrix_index).ok_or(
                FixedSetupOpeningErrorV2::InstanceMatrixIndex(instance.matrix_index),
            )?;
            if instance.log_height != matrix.height.ilog2() as usize {
                return Err(FixedSetupOpeningErrorV2::InstanceHeight(matrix.air_id));
            }
            let key = (
                instance.proof_index,
                instance.sort_idx,
                instance.part_idx,
                matrix.setup_index,
            );
            if previous_key.is_some_and(|prior| prior >= key) {
                return Err(FixedSetupOpeningErrorV2::InstanceOrder);
            }
            previous_key = Some(key);
            if seen
                .insert((instance.proof_index, matrix.setup_index), ())
                .is_some()
            {
                return Err(FixedSetupOpeningErrorV2::DuplicateInstance(
                    instance.proof_index,
                    matrix.air_id,
                ));
            }
            if seen_sort_part
                .insert(
                    (instance.proof_index, instance.sort_idx, instance.part_idx),
                    (),
                )
                .is_some()
            {
                return Err(FixedSetupOpeningErrorV2::DuplicateSortIndex(
                    instance.proof_index,
                    instance.sort_idx,
                ));
            }
            if air_by_sort
                .insert((instance.proof_index, instance.sort_idx), matrix.air_id)
                .is_some_and(|air_id| air_id != matrix.air_id)
            {
                return Err(FixedSetupOpeningErrorV2::InstanceOrder);
            }
            referenced_matrices[instance.matrix_index] = true;
            if instance.proof_index > expected_proof {
                return Err(FixedSetupOpeningErrorV2::ProofIndexGap(expected_proof));
            }
            if instance.proof_index == expected_proof {
                expected_proof = expected_proof
                    .checked_add(1)
                    .ok_or(FixedSetupOpeningErrorV2::CountOverflow)?;
            }
            let skip = 1usize
                .checked_shl(instance.l_skip as u32)
                .ok_or(FixedSetupOpeningErrorV2::ShiftOverflow)?;
            let folded_height = matrix.height.max(skip) / skip;
            let folded_log = folded_height.ilog2() as usize;
            let count = claim_count_by_proof
                .entry(instance.proof_index)
                .or_default();
            for column in 0..matrix.width {
                let ordinal = *count;
                *count = count
                    .checked_add(1)
                    .ok_or(FixedSetupOpeningErrorV2::CountOverflow)?;
                claims.push(ClaimPlanV2 {
                    proof_index: instance.proof_index,
                    claim_ordinal: ordinal,
                    claim_count: 0,
                    matrix_index: instance.matrix_index,
                    setup_index: matrix.setup_index,
                    air_id: matrix.air_id,
                    sort_idx: instance.sort_idx,
                    part_idx: instance.part_idx,
                    col_idx: u32::try_from(column)
                        .map_err(|_| FixedSetupOpeningErrorV2::CountOverflow)?,
                    l_skip: instance.l_skip,
                    log_height: instance.log_height,
                    need_rot: instance.need_rot,
                    point_len: folded_log + 1,
                    relation_digest: matrix.relation_digest,
                });
            }
        }
        if let Some((matrix, _)) = matrices
            .iter()
            .zip(referenced_matrices)
            .find(|(_, referenced)| !referenced)
        {
            return Err(FixedSetupOpeningErrorV2::UnusedMatrix(matrix.air_id));
        }
        for claim in &mut claims {
            claim.claim_count = claim_count_by_proof[&claim.proof_index];
        }

        #[cfg(test)]
        let (rows, point_demands) = retired_c1_row_plan(
            source_relation_vk_digest,
            &matrices,
            &claims,
            expected_proof,
        );
        Ok(Self {
            source_relation_vk_digest,
            matrices: matrices.into(),
            instances: instances.into(),
            claims: claims.into(),
            #[cfg(test)]
            rows: rows.into(),
            #[cfg(test)]
            point_demands: point_demands.into(),
        })
    }

    #[must_use]
    pub fn claim_count(&self) -> usize {
        self.claims.len()
    }

    pub fn claim_schedule(
        &self,
    ) -> impl ExactSizeIterator<Item = FixedSetupOpeningClaimDescriptorV2> + '_ {
        self.claims
            .iter()
            .map(|claim| FixedSetupOpeningClaimDescriptorV2 {
                proof_index: claim.proof_index,
                claim_ordinal: claim.claim_ordinal,
                claim_count: claim.claim_count,
                matrix_index: claim.matrix_index,
                setup_index: claim.setup_index,
                air_id: claim.air_id,
                sort_idx: claim.sort_idx,
                part_idx: claim.part_idx,
                col_idx: claim.col_idx,
                l_skip: claim.l_skip,
                log_height: claim.log_height,
                need_rot: claim.need_rot,
                point_len: claim.point_len,
                relation_digest: claim.relation_digest,
            })
    }

    #[must_use]
    #[cfg(test)]
    pub fn logical_row_count(&self) -> usize {
        self.rows.len()
    }

    /// Allocation metrics for this already-admitted profile. This method is
    /// exact and performs no matrix or row-plan cloning.
    #[must_use]
    #[cfg(test)]
    pub fn preflight(&self) -> FixedSetupOpeningPreflightV2 {
        let trace_height = self.rows.len().next_power_of_two().max(2);
        let main_trace_width = FixedSetupOpeningColsV2::<u8>::width();
        let preprocessed_trace_width = FixedSetupOpeningPrepColsV2::<u8>::width();
        let setup_matrix_elements = self
            .matrices
            .iter()
            .map(|matrix| matrix.width * matrix.height)
            .sum::<usize>();
        let field_bytes = core::mem::size_of::<F>();
        FixedSetupOpeningPreflightV2 {
            transition_count: self
                .claims
                .last()
                .map_or(0, |claim| claim.proof_index as usize + 1),
            setup_matrix_count: self.matrices.len(),
            setup_matrix_elements,
            setup_matrix_bytes: setup_matrix_elements * field_bytes,
            max_setup_matrix_height: self
                .matrices
                .iter()
                .map(|matrix| matrix.height)
                .max()
                .unwrap_or(0),
            claim_count: self.claims.len(),
            logical_row_count: self.rows.len(),
            trace_height,
            main_trace_width,
            preprocessed_trace_width,
            main_trace_bytes: trace_height * main_trace_width * field_bytes,
            preprocessed_trace_bytes: trace_height * preprocessed_trace_width * field_bytes,
        }
    }

    /// Lookup multiplicities required from the certified SWIRL point fanout.
    #[must_use]
    #[cfg(test)]
    pub fn point_lookup_demands(&self) -> &[(u32, u32, u32)] {
        &self.point_demands
    }
}

/// Reconstruct the retired C1 row plan only for differential tests.
#[cfg(test)]
fn retired_c1_row_plan(
    source_relation_vk_digest: Digest,
    matrices: &[FixedSetupMatrixV2],
    claims: &[ClaimPlanV2],
    expected_proof: u32,
) -> (Vec<RowPlanV2>, Vec<(u32, u32, u32)>) {
    let mut rows = Vec::new();
    let mut demands = BTreeMap::<(u32, u32), u32>::new();
    for (claim_index, claim) in claims.iter().enumerate() {
        let matrix = &matrices[claim.matrix_index];
        let skip = 1usize << claim.l_skip;
        let folded_height = matrix.height.max(skip) / skip;
        let omega = F::two_adic_generator(claim.l_skip);
        for folded_row in 0..folded_height {
            let mut omega_power = F::ONE;
            for z in 0..skip {
                rows.push(RowPlanV2::Leaf {
                    claim: claim_index,
                    folded_row,
                    z,
                    skip,
                    omega_power,
                    first: z == 0,
                    last: z + 1 == skip,
                    root: folded_height == 1,
                });
                if folded_row == 0 && z == 0 {
                    *demands.entry((claim.proof_index, 0)).or_default() += 1;
                }
                omega_power *= omega;
            }
        }
        let folded_log = claim.point_len - 1;
        for level in 1..=folded_log {
            let node_count = folded_height >> level;
            for node in 0..node_count {
                rows.push(RowPlanV2::Fold {
                    claim: claim_index,
                    level,
                    node,
                    root: level == folded_log,
                });
                if node == 0 {
                    *demands
                        .entry((claim.proof_index, level as u32))
                        .or_default() += 1;
                }
            }
        }
    }

    // This hash chain is retained only to keep the old oracle useful for
    // mutation testing. V3 production certificates are derived by the setup
    // PCS authority completion/bridge path instead.
    for proof_index in 0..expected_proof {
        let proof_claims = claims
            .iter()
            .enumerate()
            .filter(|(_, claim)| claim.proof_index == proof_index)
            .collect::<Vec<_>>();
        for (position, (claim_index, claim)) in proof_claims.iter().enumerate() {
            let matrix = &matrices[claim.matrix_index];
            let mut header = [F::ZERO; DIGEST_SIZE];
            header[0] = F::from_u32(FIXED_SETUP_OPENING_CERTIFICATE_TAG_V2);
            header[1] = F::from_u32(FIXED_SETUP_OPENING_PROTOCOL_V2);
            header[2] = F::from_u32(proof_index);
            header[3] = F::from_u32(claim.claim_ordinal);
            header[4] = F::from_u32(claim.claim_count);
            header[5] = F::from_u32(claim.air_id);
            header[6] = F::from_u32(claim.sort_idx);
            header[7] = F::from_u32(claim.part_idx);
            let mut shape = [F::ZERO; DIGEST_SIZE];
            shape[0] = F::from_u32(claim.col_idx);
            shape[1] = F::from_bool(claim.need_rot);
            shape[2] = F::from_usize(claim.l_skip);
            shape[3] = F::from_usize(claim.log_height);
            shape[4] = F::from_usize(matrix.height);
            shape[5] = F::from_usize(matrix.width);
            shape[6] = F::from_usize(claim.point_len);
            shape[7] = F::from_u32(claim.setup_index);
            let mut blocks = vec![
                HashBlockPlanV2::Static(header),
                HashBlockPlanV2::Static(shape),
                HashBlockPlanV2::Static(claim.relation_digest),
                HashBlockPlanV2::Static(source_relation_vk_digest),
            ];
            for first in (0..claim.point_len).step_by(2) {
                let second = (first + 1 < claim.point_len).then_some(first + 1);
                blocks.push(HashBlockPlanV2::PointPair { first, second });
                *demands.entry((proof_index, first as u32)).or_default() += 1;
                if let Some(second) = second {
                    *demands.entry((proof_index, second as u32)).or_default() += 1;
                }
            }
            blocks.push(HashBlockPlanV2::ClaimPair);
            let block_count = blocks.len();
            for (block_index, block) in blocks.into_iter().enumerate() {
                rows.push(RowPlanV2::Hash {
                    claim: *claim_index,
                    block,
                    first: position == 0 && block_index == 0,
                    last: position + 1 == proof_claims.len() && block_index + 1 == block_count,
                });
            }
        }
    }

    let point_demands = demands
        .into_iter()
        .map(|((proof, index), count)| (proof, index, count))
        .collect();
    (rows, point_demands)
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct FixedSetupOpeningPointMessageV2<T> {
    pub proof_index: T,
    pub point_index: T,
    pub value: [T; D_EF],
}

/// Lookup/fanout bus for the point exported by the genuine SWIRL verifier.
/// V3 computes its exact fanout from the authority statement and ordered
/// stacking profiles; it does not use the retired C1 evaluator's demand plan.
#[derive(Copy, Clone, Debug)]
pub struct FixedSetupOpeningPointBusV2(LookupBus);

impl FixedSetupOpeningPointBusV2 {
    #[must_use]
    pub fn new(bus_index: BusIndex) -> Self {
        Self(LookupBus::new(bus_index))
    }

    #[must_use]
    pub fn index(&self) -> BusIndex {
        self.0.index
    }

    pub fn lookup_key<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        message: FixedSetupOpeningPointMessageV2<impl Into<AB::Expr> + Clone>,
        enabled: impl Into<AB::Expr>,
    ) {
        self.0.lookup_key(builder, message.to_vec(), enabled);
    }

    pub fn add_key_with_lookups<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        message: FixedSetupOpeningPointMessageV2<impl Into<AB::Expr> + Clone>,
        count: impl Into<AB::Expr>,
    ) {
        self.0
            .add_key_with_lookups(builder, message.to_vec(), count);
    }
}

#[repr(C)]
#[cfg(test)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct FixedSetupOpeningNodeMessageV2<T> {
    pub proof_index: T,
    pub claim_ordinal: T,
    pub level: T,
    pub node: T,
    pub is_rotated: T,
    pub value: [T; D_EF],
}
#[cfg(test)]
define_typed_permutation_bus!(FixedSetupOpeningNodeBusV2, FixedSetupOpeningNodeMessageV2);

#[repr(C)]
#[cfg(test)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifiedFixedSetupOpeningPairMessageV2<T> {
    pub proof_index: T,
    pub claim_ordinal: T,
    pub setup_index: T,
    pub air_id: T,
    pub sort_idx: T,
    pub part_idx: T,
    pub col_idx: T,
    pub need_rot: T,
    pub relation_digest: [T; DIGEST_SIZE],
    pub current: [T; D_EF],
    pub rotated: [T; D_EF],
}
#[cfg(test)]
define_typed_permutation_bus!(
    VerifiedFixedSetupOpeningPairBusV2,
    VerifiedFixedSetupOpeningPairMessageV2
);

/// Narrow V2-compatible projection of V3 setup-PCS authority.
///
/// This message has no standalone authority; its typed bus is produced by the
/// V3 completion/claim bridge and consumed by the legacy producer bridge.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct FixedSetupOpeningCertificateMessageV2<T> {
    pub protocol_version: T,
    pub proof_index: T,
    pub canonical_claim_count: T,
    pub source_relation_vk_digest: [T; DIGEST_SIZE],
    pub setup_openings_digest: [T; DIGEST_SIZE],
}
define_typed_permutation_bus!(
    FixedSetupOpeningCertificateBusV2,
    FixedSetupOpeningCertificateMessageV2
);

#[repr(C)]
#[cfg(test)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedSetupOpeningPrepColsV2<T> {
    pub active: T,
    pub is_leaf: T,
    pub is_fold: T,
    pub is_hash: T,
    pub is_group_first: T,
    pub is_group_last: T,
    pub is_claim_first: T,
    pub is_claim_last: T,
    pub is_root: T,
    pub need_rot: T,
    pub hash_is_static: T,
    pub hash_is_point: T,
    pub hash_is_claim: T,
    pub hash_is_first: T,
    pub hash_is_last: T,
    pub hash_point_b_used: T,
    pub proof_index: T,
    pub claim_ordinal: T,
    pub claim_count: T,
    pub setup_index: T,
    pub air_id: T,
    pub sort_idx: T,
    pub part_idx: T,
    pub col_idx: T,
    pub l_skip: T,
    pub log_height: T,
    pub matrix_height: T,
    pub matrix_width: T,
    pub point_len: T,
    pub point_index_a: T,
    pub point_index_b: T,
    pub node_level: T,
    pub node_index: T,
    pub omega_power: T,
    pub fixed_current: T,
    pub fixed_next: T,
    pub relation_digest: [T; DIGEST_SIZE],
    pub source_relation_vk_digest: [T; DIGEST_SIZE],
    pub static_hash_block: [T; DIGEST_SIZE],
}

#[repr(C)]
#[cfg(test)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedSetupOpeningColsV2<T> {
    pub active: T,
    pub point_a: [T; D_EF],
    pub point_b: [T; D_EF],
    pub denominator_inverse: [T; D_EF],
    pub barycentric: [T; D_EF],
    pub scaling: [T; D_EF],
    pub bary_sum_before: [T; D_EF],
    pub bary_sum_after: [T; D_EF],
    pub current_before: [T; D_EF],
    pub current_after: [T; D_EF],
    pub rotated_before: [T; D_EF],
    pub rotated_after: [T; D_EF],
    pub left_current: [T; D_EF],
    pub right_current: [T; D_EF],
    pub output_current: [T; D_EF],
    pub left_rotated: [T; D_EF],
    pub right_rotated: [T; D_EF],
    pub output_rotated: [T; D_EF],
    pub claim_current: [T; D_EF],
    pub claim_rotated: [T; D_EF],
    pub hash_before: [T; DIGEST_SIZE],
    pub hash_block: [T; DIGEST_SIZE],
    pub hash_after: [T; DIGEST_SIZE],
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub struct FixedSetupOpeningAirV2 {
    pub profile: FixedSetupOpeningProfileV2,
    pub point_bus: FixedSetupOpeningPointBusV2,
    pub node_bus: FixedSetupOpeningNodeBusV2,
    pub verified_pair_bus: VerifiedFixedSetupOpeningPairBusV2,
    pub column_claims_bus: ColumnClaimsBus,
    pub certificate_bus: FixedSetupOpeningCertificateBusV2,
    pub compress_bus: Poseidon2CompressBus,
}

#[cfg(test)]
impl BaseAir<F> for FixedSetupOpeningAirV2 {
    fn width(&self) -> usize {
        FixedSetupOpeningColsV2::<u8>::width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = FixedSetupOpeningPrepColsV2::<u8>::width();
        let height = self.profile.rows.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        for (row_index, row_plan) in self.profile.rows.iter().enumerate() {
            let prep: &mut FixedSetupOpeningPrepColsV2<F> =
                values[row_index * width..(row_index + 1) * width].borrow_mut();
            prep.active = F::ONE;
            let claim_index = match row_plan {
                RowPlanV2::Leaf { claim, .. }
                | RowPlanV2::Fold { claim, .. }
                | RowPlanV2::Hash { claim, .. } => *claim,
            };
            let claim = &self.profile.claims[claim_index];
            let matrix = &self.profile.matrices[claim.matrix_index];
            prep.proof_index = F::from_u32(claim.proof_index);
            prep.claim_ordinal = F::from_u32(claim.claim_ordinal);
            prep.claim_count = F::from_u32(claim.claim_count);
            prep.setup_index = F::from_u32(claim.setup_index);
            prep.air_id = F::from_u32(claim.air_id);
            prep.sort_idx = F::from_u32(claim.sort_idx);
            prep.part_idx = F::from_u32(claim.part_idx);
            prep.col_idx = F::from_u32(claim.col_idx);
            prep.l_skip = F::from_usize(claim.l_skip);
            prep.log_height = F::from_usize(claim.log_height);
            prep.matrix_height = F::from_usize(matrix.height);
            prep.matrix_width = F::from_usize(matrix.width);
            prep.point_len = F::from_usize(claim.point_len);
            prep.need_rot = F::from_bool(claim.need_rot);
            prep.relation_digest = claim.relation_digest;
            prep.source_relation_vk_digest = self.profile.source_relation_vk_digest;
            match row_plan {
                RowPlanV2::Leaf {
                    folded_row,
                    z,
                    skip,
                    omega_power,
                    first,
                    last,
                    root,
                    ..
                } => {
                    let physical = ((*folded_row << claim.l_skip) + *z) & (matrix.height - 1);
                    let next_physical = (physical + 1) & (matrix.height - 1);
                    prep.is_leaf = F::ONE;
                    prep.is_group_first = F::from_bool(*first);
                    prep.is_group_last = F::from_bool(*last);
                    prep.is_claim_first = F::from_bool(*folded_row == 0 && *z == 0);
                    prep.is_claim_last = F::from_bool(
                        *folded_row + 1 == matrix.height.max(*skip) / *skip && *z + 1 == *skip,
                    );
                    prep.is_root = F::from_bool(*root && *last);
                    prep.point_index_a = F::ZERO;
                    prep.node_level = F::ZERO;
                    prep.node_index = F::from_usize(*folded_row);
                    prep.omega_power = *omega_power;
                    prep.fixed_current = matrix.value(physical, claim.col_idx as usize);
                    prep.fixed_next = if claim.need_rot {
                        matrix.value(next_physical, claim.col_idx as usize)
                    } else {
                        F::ZERO
                    };
                    debug_assert!(*z < *skip);
                }
                RowPlanV2::Fold {
                    level, node, root, ..
                } => {
                    prep.is_fold = F::ONE;
                    prep.is_root = F::from_bool(*root);
                    let node_count =
                        (matrix.height.max(1usize << claim.l_skip) >> claim.l_skip) >> *level;
                    prep.is_group_first = F::from_bool(*node == 0);
                    prep.is_group_last = F::from_bool(*node + 1 == node_count);
                    prep.point_index_a = F::from_usize(*level);
                    prep.node_level = F::from_usize(*level);
                    prep.node_index = F::from_usize(*node);
                }
                RowPlanV2::Hash {
                    block, first, last, ..
                } => {
                    prep.is_hash = F::ONE;
                    prep.hash_is_first = F::from_bool(*first);
                    prep.hash_is_last = F::from_bool(*last);
                    match block {
                        HashBlockPlanV2::Static(block) => {
                            prep.hash_is_static = F::ONE;
                            prep.static_hash_block = *block;
                        }
                        HashBlockPlanV2::PointPair { first, second } => {
                            prep.hash_is_point = F::ONE;
                            prep.point_index_a = F::from_usize(*first);
                            if let Some(second) = second {
                                prep.hash_point_b_used = F::ONE;
                                prep.point_index_b = F::from_usize(*second);
                            }
                        }
                        HashBlockPlanV2::ClaimPair => prep.hash_is_claim = F::ONE,
                    }
                }
            }
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

#[cfg(test)]
impl BaseAirWithPublicValues<F> for FixedSetupOpeningAirV2 {}
#[cfg(test)]
impl PartitionedBaseAir<F> for FixedSetupOpeningAirV2 {}

#[cfg(test)]
fn ext_zero_expr<FA: PrimeCharacteristicRing>() -> [FA; D_EF] {
    core::array::from_fn(|_| FA::ZERO)
}

#[cfg(test)]
fn ext_one_expr<FA: PrimeCharacteristicRing>() -> [FA; D_EF] {
    core::array::from_fn(|index| if index == 0 { FA::ONE } else { FA::ZERO })
}

#[cfg(test)]
fn ext_add_expr<FA: PrimeCharacteristicRing>(
    left: [impl Into<FA>; D_EF],
    right: [impl Into<FA>; D_EF],
) -> [FA; D_EF] {
    let left = left.map(Into::into);
    let right = right.map(Into::into);
    core::array::from_fn(|index| left[index].clone() + right[index].clone())
}

#[cfg(test)]
fn ext_sub_expr<FA: PrimeCharacteristicRing>(
    left: [impl Into<FA>; D_EF],
    right: [impl Into<FA>; D_EF],
) -> [FA; D_EF] {
    let left = left.map(Into::into);
    let right = right.map(Into::into);
    core::array::from_fn(|index| left[index].clone() - right[index].clone())
}

#[cfg(test)]
fn ext_scale_expr<FA: PrimeCharacteristicRing>(
    value: [impl Into<FA>; D_EF],
    scalar: impl Into<FA> + Clone,
) -> [FA; D_EF] {
    let value = value.map(Into::into);
    let scalar = scalar.into();
    core::array::from_fn(|index| value[index].clone() * scalar.clone())
}

#[cfg(test)]
fn ext_mul_expr<FA>(left: [impl Into<FA>; D_EF], right: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
    FA::PrimeSubfield: BinomiallyExtendable<D_EF>,
{
    let left = left.map(Into::into);
    let right = right.map(Into::into);
    let w = FA::from_prime_subfield(FA::PrimeSubfield::W);
    let mut output = core::array::from_fn(|_| FA::ZERO);
    for (left_degree, left_value) in left.iter().enumerate() {
        for (right_degree, right_value) in right.iter().enumerate() {
            let degree = left_degree + right_degree;
            let mut term = left_value.clone() * right_value.clone();
            if degree >= D_EF {
                term *= w.clone();
            }
            output[degree % D_EF] = output[degree % D_EF].clone() + term;
        }
    }
    output
}

#[cfg(test)]
fn assert_ext_zero<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    enabled: impl Into<AB::Expr> + Clone,
    value: [AB::Var; D_EF],
) where
    AB::Var: Copy,
{
    for limb in value {
        builder.when(enabled.clone()).assert_zero(limb);
    }
}

#[cfg(test)]
impl<AB> Air<AB> for FixedSetupOpeningAirV2
where
    AB: AirBuilder<F = F> + InteractionBuilder + PairBuilder,
    AB::Var: Copy,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let prep_matrix = builder.preprocessed();
        let prep_row = prep_matrix.row_slice(0).expect("fixed setup prep row");
        let prep_next_row = prep_matrix.row_slice(1).expect("fixed setup next prep row");
        let prep: &FixedSetupOpeningPrepColsV2<AB::Var> = (*prep_row).borrow();
        let prep_next: &FixedSetupOpeningPrepColsV2<AB::Var> = (*prep_next_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("fixed setup row");
        let next_row = main.row_slice(1).expect("fixed setup next row");
        let local: &FixedSetupOpeningColsV2<AB::Var> = (*row).borrow();
        let next: &FixedSetupOpeningColsV2<AB::Var> = (*next_row).borrow();

        for bit in [
            prep.active,
            prep.is_leaf,
            prep.is_fold,
            prep.is_hash,
            prep.is_group_first,
            prep.is_group_last,
            prep.is_claim_first,
            prep.is_claim_last,
            prep.is_root,
            prep.need_rot,
            prep.hash_is_static,
            prep.hash_is_point,
            prep.hash_is_claim,
            prep.hash_is_first,
            prep.hash_is_last,
            prep.hash_point_b_used,
            local.active,
        ] {
            builder.assert_bool(bit);
        }
        builder.assert_eq(local.active, prep.active);
        builder.assert_eq(prep.active, prep.is_leaf + prep.is_fold + prep.is_hash);
        builder.assert_eq(
            prep.is_hash,
            prep.hash_is_static + prep.hash_is_point + prep.hash_is_claim,
        );
        let enabled = AB::Expr::from(prep.active);
        let leaf = AB::Expr::from(prep.is_leaf);
        let fold = AB::Expr::from(prep.is_fold);
        let hash = AB::Expr::from(prep.is_hash);
        let root = enabled.clone() * AB::Expr::from(prep.is_root);
        let need_rot = AB::Expr::from(prep.need_rot);

        // Padding rows are canonical zero witness rows.
        for &cell in row.iter() {
            builder
                .when(AB::Expr::ONE - enabled.clone())
                .assert_zero(cell);
        }

        // Every arithmetic row obtains its challenge from the verifier-owned
        // SWIRL point fanout. Hash point rows do the same below.
        let point_first = leaf.clone() * AB::Expr::from(prep.is_claim_first)
            + fold.clone() * AB::Expr::from(prep.is_group_first)
            + AB::Expr::from(prep.hash_is_point);
        self.point_bus.lookup_key(
            builder,
            FixedSetupOpeningPointMessageV2 {
                proof_index: prep.proof_index.into(),
                point_index: prep.point_index_a.into(),
                value: local.point_a.map(Into::into),
            },
            point_first,
        );
        self.point_bus.lookup_key(
            builder,
            FixedSetupOpeningPointMessageV2 {
                proof_index: prep.proof_index.into(),
                point_index: prep.point_index_b.into(),
                value: local.point_b.map(Into::into),
            },
            AB::Expr::from(prep.hash_is_point) * AB::Expr::from(prep.hash_point_b_used),
        );
        let point_carries = leaf.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_claim_last))
            + fold.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_group_last));
        assert_array_eq(
            &mut builder.when_transition().when(point_carries),
            next.point_a,
            local.point_a.map(Into::into),
        );

        // Leaf: unique PLE barycentric coefficient and running weighted sum.
        let omega_ext = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::from(prep.omega_power)
            } else {
                AB::Expr::ZERO
            }
        });
        let denominator = ext_sub_expr::<AB::Expr>(local.point_a, omega_ext.clone());
        assert_array_eq(
            &mut builder.when(leaf.clone()),
            ext_mul_expr::<AB::Expr>(denominator.clone(), local.denominator_inverse),
            ext_one_expr::<AB::Expr>(),
        );
        assert_array_eq(
            &mut builder.when(leaf.clone()),
            ext_mul_expr::<AB::Expr>(denominator, local.barycentric),
            ext_mul_expr::<AB::Expr>(omega_ext, local.scaling),
        );
        assert_array_eq(
            &mut builder.when(leaf.clone()),
            local.bary_sum_after,
            ext_add_expr::<AB::Expr>(local.bary_sum_before, local.barycentric),
        );
        assert_array_eq(
            &mut builder.when(leaf.clone()),
            local.current_after,
            ext_add_expr::<AB::Expr>(
                local.current_before,
                ext_scale_expr::<AB::Expr>(local.barycentric, prep.fixed_current),
            ),
        );
        assert_array_eq(
            &mut builder.when(leaf.clone() * need_rot.clone()),
            local.rotated_after,
            ext_add_expr::<AB::Expr>(
                local.rotated_before,
                ext_scale_expr::<AB::Expr>(local.barycentric, prep.fixed_next),
            ),
        );
        let leaf_first = leaf.clone() * AB::Expr::from(prep.is_group_first);
        let leaf_last = leaf.clone() * AB::Expr::from(prep.is_group_last);
        assert_array_eq(
            &mut builder.when(leaf_first.clone()),
            local.bary_sum_before,
            ext_zero_expr::<AB::Expr>(),
        );
        assert_array_eq(
            &mut builder.when(leaf_first.clone()),
            local.current_before,
            ext_zero_expr::<AB::Expr>(),
        );
        assert_array_eq(
            &mut builder.when(leaf_first.clone()),
            local.rotated_before,
            ext_zero_expr::<AB::Expr>(),
        );
        assert_array_eq(
            &mut builder.when(leaf_last.clone()),
            local.bary_sum_after,
            ext_one_expr::<AB::Expr>(),
        );
        let leaf_continues = leaf.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_group_last));
        assert_array_eq(
            &mut builder.when_transition().when(leaf_continues.clone()),
            next.scaling,
            local.scaling.map(Into::into),
        );
        assert_array_eq(
            &mut builder.when_transition().when(leaf_continues.clone()),
            next.bary_sum_before,
            local.bary_sum_after.map(Into::into),
        );
        assert_array_eq(
            &mut builder.when_transition().when(leaf_continues.clone()),
            next.current_before,
            local.current_after.map(Into::into),
        );
        assert_array_eq(
            &mut builder.when_transition().when(leaf_continues),
            next.rotated_before,
            local.rotated_after.map(Into::into),
        );
        assert_ext_zero(
            builder,
            leaf.clone() * (AB::Expr::ONE - need_rot.clone()),
            local.rotated_before,
        );
        assert_ext_zero(
            builder,
            leaf.clone() * (AB::Expr::ONE - need_rot.clone()),
            local.rotated_after,
        );

        // Non-root leaves feed level zero. Fold rows consume two children,
        // interpolate at opening_point[level], and either publish the parent
        // or certify the final opening pair.
        let leaf_nonroot = leaf_last.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_root));
        self.node_bus.send(
            builder,
            FixedSetupOpeningNodeMessageV2 {
                proof_index: prep.proof_index.into(),
                claim_ordinal: prep.claim_ordinal.into(),
                level: AB::Expr::ZERO,
                node: prep.node_index.into(),
                is_rotated: AB::Expr::ZERO,
                value: local.current_after.map(Into::into),
            },
            leaf_nonroot.clone(),
        );
        self.node_bus.send(
            builder,
            FixedSetupOpeningNodeMessageV2 {
                proof_index: prep.proof_index.into(),
                claim_ordinal: prep.claim_ordinal.into(),
                level: AB::Expr::ZERO,
                node: prep.node_index.into(),
                is_rotated: AB::Expr::ONE,
                value: local.rotated_after.map(Into::into),
            },
            leaf_nonroot * need_rot.clone(),
        );

        let child_level = AB::Expr::from(prep.node_level) - AB::Expr::ONE;
        let left_node = AB::Expr::from(prep.node_index) * AB::Expr::TWO;
        for (rotated, left_value, right_value) in [
            (false, local.left_current, local.right_current),
            (true, local.left_rotated, local.right_rotated),
        ] {
            let is_rotated = AB::Expr::from_bool(rotated);
            let rotation_enabled = if rotated {
                fold.clone() * need_rot.clone()
            } else {
                fold.clone()
            };
            self.node_bus.receive(
                builder,
                FixedSetupOpeningNodeMessageV2 {
                    proof_index: prep.proof_index.into(),
                    claim_ordinal: prep.claim_ordinal.into(),
                    level: child_level.clone(),
                    node: left_node.clone(),
                    is_rotated: is_rotated.clone(),
                    value: left_value.map(Into::into),
                },
                rotation_enabled.clone(),
            );
            self.node_bus.receive(
                builder,
                FixedSetupOpeningNodeMessageV2 {
                    proof_index: prep.proof_index.into(),
                    claim_ordinal: prep.claim_ordinal.into(),
                    level: child_level.clone(),
                    node: left_node.clone() + AB::Expr::ONE,
                    is_rotated,
                    value: right_value.map(Into::into),
                },
                rotation_enabled,
            );
        }
        let folded_current = ext_add_expr::<AB::Expr>(
            local.left_current,
            ext_mul_expr::<AB::Expr>(
                local.point_a,
                ext_sub_expr::<AB::Expr>(local.right_current, local.left_current),
            ),
        );
        let folded_rotated = ext_add_expr::<AB::Expr>(
            local.left_rotated,
            ext_mul_expr::<AB::Expr>(
                local.point_a,
                ext_sub_expr::<AB::Expr>(local.right_rotated, local.left_rotated),
            ),
        );
        assert_array_eq(
            &mut builder.when(fold.clone()),
            local.output_current,
            folded_current,
        );
        assert_array_eq(
            &mut builder.when(fold.clone() * need_rot.clone()),
            local.output_rotated,
            folded_rotated,
        );
        assert_ext_zero(
            builder,
            fold.clone() * (AB::Expr::ONE - need_rot.clone()),
            local.output_rotated,
        );
        let fold_nonroot = fold.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_root));
        self.node_bus.send(
            builder,
            FixedSetupOpeningNodeMessageV2 {
                proof_index: prep.proof_index.into(),
                claim_ordinal: prep.claim_ordinal.into(),
                level: prep.node_level.into(),
                node: prep.node_index.into(),
                is_rotated: AB::Expr::ZERO,
                value: local.output_current.map(Into::into),
            },
            fold_nonroot.clone(),
        );
        self.node_bus.send(
            builder,
            FixedSetupOpeningNodeMessageV2 {
                proof_index: prep.proof_index.into(),
                claim_ordinal: prep.claim_ordinal.into(),
                level: prep.node_level.into(),
                node: prep.node_index.into(),
                is_rotated: AB::Expr::ONE,
                value: local.output_rotated.map(Into::into),
            },
            fold_nonroot * need_rot.clone(),
        );

        let computed_current: [AB::Expr; D_EF] = core::array::from_fn(|limb| {
            leaf.clone() * AB::Expr::from(local.current_after[limb])
                + fold.clone() * AB::Expr::from(local.output_current[limb])
        });
        let computed_rotated: [AB::Expr; D_EF] = core::array::from_fn(|limb| {
            leaf.clone() * AB::Expr::from(local.rotated_after[limb])
                + fold.clone() * AB::Expr::from(local.output_rotated[limb])
        });
        assert_array_eq(
            &mut builder.when(root.clone()),
            local.claim_current,
            computed_current,
        );
        assert_array_eq(
            &mut builder.when(root.clone() * need_rot.clone()),
            local.claim_rotated,
            computed_rotated,
        );
        assert_ext_zero(
            builder,
            root.clone() * (AB::Expr::ONE - need_rot.clone()),
            local.claim_rotated,
        );

        let pair_message = VerifiedFixedSetupOpeningPairMessageV2 {
            proof_index: prep.proof_index.into(),
            claim_ordinal: prep.claim_ordinal.into(),
            setup_index: prep.setup_index.into(),
            air_id: prep.air_id.into(),
            sort_idx: prep.sort_idx.into(),
            part_idx: prep.part_idx.into(),
            col_idx: prep.col_idx.into(),
            need_rot: prep.need_rot.into(),
            relation_digest: prep.relation_digest.map(Into::into),
            current: local.claim_current.map(Into::into),
            rotated: local.claim_rotated.map(Into::into),
        };
        self.verified_pair_bus
            .send(builder, pair_message, root.clone());
        self.column_claims_bus.send(
            builder,
            prep.proof_index,
            ColumnClaimsMessage {
                sort_idx: prep.sort_idx.into(),
                part_idx: prep.part_idx.into(),
                col_idx: prep.col_idx.into(),
                claim: local.claim_current.map(Into::into),
                is_rot: AB::Expr::ZERO,
            },
            root.clone(),
        );
        // The genuine recursive SWIRL verifier has two consumers for each
        // ColumnClaims message: symbolic constraint evaluation and transcript
        // observation. Match that exact multiplicity for fixed claims too.
        self.column_claims_bus.send(
            builder,
            prep.proof_index,
            ColumnClaimsMessage {
                sort_idx: prep.sort_idx.into(),
                part_idx: prep.part_idx.into(),
                col_idx: prep.col_idx.into(),
                claim: local.claim_current.map(Into::into),
                is_rot: AB::Expr::ZERO,
            },
            root.clone(),
        );
        self.column_claims_bus.send(
            builder,
            prep.proof_index,
            ColumnClaimsMessage {
                sort_idx: prep.sort_idx.into(),
                part_idx: prep.part_idx.into(),
                col_idx: prep.col_idx.into(),
                claim: local.claim_rotated.map(Into::into),
                is_rot: AB::Expr::ONE,
            },
            root.clone() * need_rot.clone(),
        );
        self.column_claims_bus.send(
            builder,
            prep.proof_index,
            ColumnClaimsMessage {
                sort_idx: prep.sort_idx.into(),
                part_idx: prep.part_idx.into(),
                col_idx: prep.col_idx.into(),
                claim: local.claim_rotated.map(Into::into),
                is_rot: AB::Expr::ONE,
            },
            root.clone() * need_rot.clone(),
        );

        // Canonical hash stream. Static blocks are index data; point and claim
        // blocks are obtained only from their certified buses.
        let hash_static = AB::Expr::from(prep.hash_is_static);
        assert_array_eq(
            &mut builder.when(hash_static),
            local.hash_block,
            prep.static_hash_block.map(Into::into),
        );
        let point_block: [AB::Expr; DIGEST_SIZE] = core::array::from_fn(|index| {
            if index < D_EF {
                AB::Expr::from(local.point_a[index])
            } else {
                AB::Expr::from(local.point_b[index - D_EF]) * AB::Expr::from(prep.hash_point_b_used)
            }
        });
        assert_array_eq(
            &mut builder.when(prep.hash_is_point),
            local.hash_block,
            point_block,
        );
        self.verified_pair_bus.receive(
            builder,
            VerifiedFixedSetupOpeningPairMessageV2 {
                proof_index: prep.proof_index.into(),
                claim_ordinal: prep.claim_ordinal.into(),
                setup_index: prep.setup_index.into(),
                air_id: prep.air_id.into(),
                sort_idx: prep.sort_idx.into(),
                part_idx: prep.part_idx.into(),
                col_idx: prep.col_idx.into(),
                need_rot: prep.need_rot.into(),
                relation_digest: prep.relation_digest.map(Into::into),
                current: local.claim_current.map(Into::into),
                rotated: local.claim_rotated.map(Into::into),
            },
            prep.hash_is_claim,
        );
        let claim_block: [AB::Expr; DIGEST_SIZE] = core::array::from_fn(|index| {
            if index < D_EF {
                AB::Expr::from(local.claim_current[index])
            } else {
                AB::Expr::from(local.claim_rotated[index - D_EF])
            }
        });
        assert_array_eq(
            &mut builder.when(prep.hash_is_claim),
            local.hash_block,
            claim_block,
        );
        for limb in local.point_b {
            builder
                .when(
                    AB::Expr::from(prep.hash_is_point)
                        * (AB::Expr::ONE - AB::Expr::from(prep.hash_point_b_used)),
                )
                .assert_zero(limb);
        }
        for limb in 0..DIGEST_SIZE {
            builder
                .when(AB::Expr::from(prep.hash_is_first))
                .assert_zero(local.hash_before[limb]);
        }
        self.compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: core::array::from_fn(|index| {
                    if index < DIGEST_SIZE {
                        local.hash_before[index].into()
                    } else {
                        local.hash_block[index - DIGEST_SIZE].into()
                    }
                }),
                output: local.hash_after.map(Into::into),
            },
            hash.clone(),
        );
        let hash_continues = hash.clone()
            * AB::Expr::from(prep_next.is_hash)
            * (AB::Expr::ONE - AB::Expr::from(prep.hash_is_last));
        for limb in 0..DIGEST_SIZE {
            builder
                .when_transition()
                .when(hash_continues.clone())
                .assert_eq(next.hash_before[limb], local.hash_after[limb]);
        }
        self.certificate_bus.send(
            builder,
            FixedSetupOpeningCertificateMessageV2 {
                protocol_version: AB::Expr::from_u32(FIXED_SETUP_OPENING_PROTOCOL_V2),
                proof_index: prep.proof_index.into(),
                canonical_claim_count: prep.claim_count.into(),
                source_relation_vk_digest: prep.source_relation_vk_digest.map(Into::into),
                setup_openings_digest: local.hash_after.map(Into::into),
            },
            prep.hash_is_last,
        );
    }
}

#[cfg(test)]
#[derive(Debug)]
pub struct FixedSetupOpeningTraceV2 {
    pub matrix: RowMajorMatrix<F>,
    pub compression_inputs: Vec<[F; 2 * DIGEST_SIZE]>,
    pub certificates: Vec<(u32, u32, Digest)>,
}

#[cfg(test)]
fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

#[cfg(test)]
fn record_maps<'a>(
    profile: &FixedSetupOpeningProfileV2,
    records: &'a [FixedSetupOpeningProofRecordV2],
) -> Result<BTreeMap<u32, &'a FixedSetupOpeningProofRecordV2>, FixedSetupOpeningErrorV2> {
    let mut map = BTreeMap::new();
    for record in records {
        if map.insert(record.proof_index, record).is_some() {
            return Err(FixedSetupOpeningErrorV2::UnexpectedProof(
                record.proof_index,
            ));
        }
    }
    let max_proof = profile
        .claims
        .last()
        .ok_or(FixedSetupOpeningErrorV2::EmptyProfile)?
        .proof_index;
    for proof_index in 0..=max_proof {
        let record = map
            .get(&proof_index)
            .ok_or(FixedSetupOpeningErrorV2::MissingProof(proof_index))?;
        let expected = profile
            .claims
            .iter()
            .filter(|claim| claim.proof_index == proof_index)
            .collect::<Vec<_>>();
        let required_point_len = expected
            .iter()
            .map(|claim| claim.point_len)
            .max()
            .unwrap_or(0);
        if record.opening_point.len() != required_point_len {
            return Err(FixedSetupOpeningErrorV2::OpeningPointLength(proof_index));
        }
        if record.claims.len() != expected.len() {
            return Err(FixedSetupOpeningErrorV2::ClaimCount(proof_index));
        }
        for (index, (record_claim, plan)) in record.claims.iter().zip(expected).enumerate() {
            if record_claim.setup_index != plan.setup_index
                || record_claim.air_id != plan.air_id
                || record_claim.sort_idx != plan.sort_idx
                || record_claim.part_idx != plan.part_idx
                || record_claim.col_idx != plan.col_idx
                || record_claim.rotated.is_some() != plan.need_rot
            {
                return Err(FixedSetupOpeningErrorV2::ClaimSchedule(proof_index, index));
            }
        }
    }
    if map.len() != max_proof as usize + 1 {
        let unexpected = map
            .keys()
            .copied()
            .find(|&proof| proof > max_proof)
            .unwrap_or(max_proof + 1);
        return Err(FixedSetupOpeningErrorV2::UnexpectedProof(unexpected));
    }
    Ok(map)
}

/// Generate the honest main trace. The caller must feed `compression_inputs`
/// to the same Poseidon2 compression table used by the enclosing recursive
/// verifier.
#[cfg(test)]
pub fn generate_fixed_setup_opening_trace_v2(
    air: &FixedSetupOpeningAirV2,
    records: &[FixedSetupOpeningProofRecordV2],
) -> Result<FixedSetupOpeningTraceV2, FixedSetupOpeningErrorV2> {
    let record_map = record_maps(&air.profile, records)?;
    let width = FixedSetupOpeningColsV2::<u8>::width();
    let height = air.profile.rows.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    let mut nodes = BTreeMap::<(usize, usize, usize, bool), EF>::new();
    let mut compression_inputs = Vec::new();
    let mut hash_state_by_proof = BTreeMap::<u32, Digest>::new();
    let mut certificates = Vec::new();

    for (row_index, row_plan) in air.profile.rows.iter().enumerate() {
        let prior_leaf_state = match row_plan {
            RowPlanV2::Leaf { first: false, .. } => {
                let previous: &FixedSetupOpeningColsV2<F> =
                    values[(row_index - 1) * width..row_index * width].borrow();
                Some((
                    EF::from_basis_coefficients_slice(&previous.bary_sum_after)
                        .ok_or(FixedSetupOpeningErrorV2::InternalPlan)?,
                    EF::from_basis_coefficients_slice(&previous.current_after)
                        .ok_or(FixedSetupOpeningErrorV2::InternalPlan)?,
                    EF::from_basis_coefficients_slice(&previous.rotated_after)
                        .ok_or(FixedSetupOpeningErrorV2::InternalPlan)?,
                ))
            }
            _ => None,
        };
        let cols: &mut FixedSetupOpeningColsV2<F> =
            values[row_index * width..(row_index + 1) * width].borrow_mut();
        cols.active = F::ONE;
        let claim_index = match row_plan {
            RowPlanV2::Leaf { claim, .. }
            | RowPlanV2::Fold { claim, .. }
            | RowPlanV2::Hash { claim, .. } => *claim,
        };
        let plan = &air.profile.claims[claim_index];
        let record = record_map[&plan.proof_index];
        let claim_record = &record.claims[plan.claim_ordinal as usize];
        match row_plan {
            RowPlanV2::Leaf {
                folded_row,
                z,
                skip,
                omega_power,
                first: _,
                last,
                root,
                ..
            } => {
                let matrix = &air.profile.matrices[plan.matrix_index];
                let physical = ((*folded_row << plan.l_skip) + *z) & (matrix.height - 1);
                let next_physical = (physical + 1) & (matrix.height - 1);
                let fixed_current = matrix.value(physical, plan.col_idx as usize);
                let fixed_next = matrix.value(next_physical, plan.col_idx as usize);
                let r0 = record.opening_point[0];
                copy_ext(&mut cols.point_a, r0);
                let denominator = r0 - EF::from(*omega_power);
                if denominator == EF::ZERO {
                    return Err(FixedSetupOpeningErrorV2::PleDenominator(plan.proof_index));
                }
                let inverse = denominator.inverse();
                copy_ext(&mut cols.denominator_inverse, inverse);
                let scaling =
                    (r0.exp_u64(*skip as u64) - EF::ONE) * EF::from_usize(*skip).inverse();
                let barycentric = EF::from(*omega_power) * inverse * scaling;
                copy_ext(&mut cols.scaling, scaling);
                copy_ext(&mut cols.barycentric, barycentric);

                let (bary_before, current_before, rotated_before) =
                    prior_leaf_state.unwrap_or((EF::ZERO, EF::ZERO, EF::ZERO));
                let current_after = current_before + barycentric * EF::from(fixed_current);
                let rotated_after = if plan.need_rot {
                    rotated_before + barycentric * EF::from(fixed_next)
                } else {
                    EF::ZERO
                };
                copy_ext(&mut cols.bary_sum_before, bary_before);
                copy_ext(&mut cols.bary_sum_after, bary_before + barycentric);
                copy_ext(&mut cols.current_before, current_before);
                copy_ext(&mut cols.current_after, current_after);
                copy_ext(&mut cols.rotated_before, rotated_before);
                copy_ext(&mut cols.rotated_after, rotated_after);
                if *last {
                    if *root {
                        copy_ext(&mut cols.claim_current, claim_record.current);
                        copy_ext(
                            &mut cols.claim_rotated,
                            claim_record.rotated.unwrap_or(EF::ZERO),
                        );
                    } else {
                        nodes.insert((claim_index, 0, *folded_row, false), current_after);
                        if plan.need_rot {
                            nodes.insert((claim_index, 0, *folded_row, true), rotated_after);
                        }
                    }
                }
                debug_assert!(*z < *skip);
            }
            RowPlanV2::Fold {
                level, node, root, ..
            } => {
                let challenge = record.opening_point[*level];
                copy_ext(&mut cols.point_a, challenge);
                let child_level = level - 1;
                let left_index = 2 * node;
                let right_index = left_index + 1;
                let left_current = nodes
                    .remove(&(claim_index, child_level, left_index, false))
                    .ok_or(FixedSetupOpeningErrorV2::InternalPlan)?;
                let right_current = nodes
                    .remove(&(claim_index, child_level, right_index, false))
                    .ok_or(FixedSetupOpeningErrorV2::InternalPlan)?;
                let output_current = left_current + challenge * (right_current - left_current);
                copy_ext(&mut cols.left_current, left_current);
                copy_ext(&mut cols.right_current, right_current);
                copy_ext(&mut cols.output_current, output_current);
                let output_rotated = if plan.need_rot {
                    let left = nodes
                        .remove(&(claim_index, child_level, left_index, true))
                        .ok_or(FixedSetupOpeningErrorV2::InternalPlan)?;
                    let right = nodes
                        .remove(&(claim_index, child_level, right_index, true))
                        .ok_or(FixedSetupOpeningErrorV2::InternalPlan)?;
                    let output = left + challenge * (right - left);
                    copy_ext(&mut cols.left_rotated, left);
                    copy_ext(&mut cols.right_rotated, right);
                    copy_ext(&mut cols.output_rotated, output);
                    output
                } else {
                    EF::ZERO
                };
                if *root {
                    copy_ext(&mut cols.claim_current, claim_record.current);
                    copy_ext(
                        &mut cols.claim_rotated,
                        claim_record.rotated.unwrap_or(EF::ZERO),
                    );
                } else {
                    nodes.insert((claim_index, *level, *node, false), output_current);
                    if plan.need_rot {
                        nodes.insert((claim_index, *level, *node, true), output_rotated);
                    }
                }
            }
            RowPlanV2::Hash {
                block, first, last, ..
            } => {
                let state = hash_state_by_proof
                    .entry(plan.proof_index)
                    .or_insert([F::ZERO; DIGEST_SIZE]);
                if *first {
                    *state = [F::ZERO; DIGEST_SIZE];
                }
                cols.hash_before = *state;
                match block {
                    HashBlockPlanV2::Static(block) => cols.hash_block = *block,
                    HashBlockPlanV2::PointPair { first, second } => {
                        let first_value = record.opening_point[*first];
                        copy_ext(&mut cols.point_a, first_value);
                        cols.hash_block[..D_EF]
                            .copy_from_slice(first_value.as_basis_coefficients_slice());
                        if let Some(second) = second {
                            let second_value = record.opening_point[*second];
                            copy_ext(&mut cols.point_b, second_value);
                            cols.hash_block[D_EF..]
                                .copy_from_slice(second_value.as_basis_coefficients_slice());
                        }
                    }
                    HashBlockPlanV2::ClaimPair => {
                        copy_ext(&mut cols.claim_current, claim_record.current);
                        copy_ext(
                            &mut cols.claim_rotated,
                            claim_record.rotated.unwrap_or(EF::ZERO),
                        );
                        cols.hash_block[..D_EF]
                            .copy_from_slice(claim_record.current.as_basis_coefficients_slice());
                        cols.hash_block[D_EF..].copy_from_slice(
                            claim_record
                                .rotated
                                .unwrap_or(EF::ZERO)
                                .as_basis_coefficients_slice(),
                        );
                    }
                }
                compression_inputs.push(core::array::from_fn(|index| {
                    if index < DIGEST_SIZE {
                        cols.hash_before[index]
                    } else {
                        cols.hash_block[index - DIGEST_SIZE]
                    }
                }));
                *state = poseidon2_compress_with_capacity(*state, cols.hash_block).0;
                cols.hash_after = *state;
                if *last {
                    certificates.push((plan.proof_index, plan.claim_count, *state));
                }
            }
        }
    }
    if !nodes.is_empty() {
        return Err(FixedSetupOpeningErrorV2::InternalPlan);
    }
    Ok(FixedSetupOpeningTraceV2 {
        matrix: RowMajorMatrix::new(values, width),
        compression_inputs,
        certificates,
    })
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use openvm_recursion_circuit::system::BusIndexManager;
    use openvm_stark_backend::{
        air_builders::{
            debug::{check_constraints, check_logup},
            symbolic::get_symbolic_builder,
        },
        interaction::{InteractionBuilder, SymbolicInteraction},
        keygen::types::TraceWidth,
        p3_air::{Air, AirBuilder, BaseAir},
        p3_matrix::Matrix,
        AnyAir, BaseAirWithPublicValues, PartitionedBaseAir,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config as NativeSC;

    use super::*;

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|index| F::from_u32(seed + index as u32))
    }

    fn make_profile(
        values: Vec<F>,
        air_id: u32,
        relation_digest: Digest,
        vk_digest: Digest,
        height: usize,
        width: usize,
        l_skip: usize,
        need_rot: bool,
    ) -> FixedSetupOpeningProfileV2 {
        FixedSetupOpeningProfileV2::new(
            vk_digest,
            vec![FixedSetupMatrixV2 {
                setup_index: 0,
                air_id,
                relation_digest,
                width,
                height,
                values: values.into(),
            }],
            vec![FixedSetupOpeningInstanceV2 {
                proof_index: 0,
                matrix_index: 0,
                sort_idx: 3,
                part_idx: 1,
                l_skip,
                log_height: height.ilog2() as usize,
                need_rot,
            }],
        )
        .expect("fixed setup profile")
    }

    fn make_air(profile: FixedSetupOpeningProfileV2) -> FixedSetupOpeningAirV2 {
        let mut manager = BusIndexManager::new();
        FixedSetupOpeningAirV2 {
            profile,
            point_bus: FixedSetupOpeningPointBusV2::new(manager.new_bus_idx()),
            node_bus: FixedSetupOpeningNodeBusV2::new(manager.new_bus_idx()),
            verified_pair_bus: VerifiedFixedSetupOpeningPairBusV2::new(manager.new_bus_idx()),
            column_claims_bus: ColumnClaimsBus::new(manager.new_bus_idx()),
            certificate_bus: FixedSetupOpeningCertificateBusV2::new(manager.new_bus_idx()),
            compress_bus: Poseidon2CompressBus::new(manager.new_bus_idx()),
        }
    }

    fn reference_opening(
        matrix: &FixedSetupMatrixV2,
        column: usize,
        l_skip: usize,
        point: &[EF],
        rotated: bool,
    ) -> EF {
        let skip = 1usize << l_skip;
        let folded_height = matrix.height.max(skip) / skip;
        assert_eq!(point.len(), folded_height.ilog2() as usize + 1);
        let r0 = point[0];
        let omega = F::two_adic_generator(l_skip);
        let scaling = (r0.exp_u64(skip as u64) - EF::ONE) * EF::from_usize(skip).inverse();
        let mut leaves = Vec::with_capacity(folded_height);
        for folded_row in 0..folded_height {
            let mut value = EF::ZERO;
            let mut omega_power = F::ONE;
            for z in 0..skip {
                let denominator = r0 - EF::from(omega_power);
                let barycentric = EF::from(omega_power) * denominator.inverse() * scaling;
                let physical =
                    ((folded_row << l_skip) + z + usize::from(rotated)) & (matrix.height - 1);
                value += EF::from(matrix.value(physical, column)) * barycentric;
                omega_power *= omega;
            }
            leaves.push(value);
        }
        for &challenge in &point[1..] {
            for node in 0..leaves.len() / 2 {
                leaves[node] =
                    leaves[2 * node] + challenge * (leaves[2 * node + 1] - leaves[2 * node]);
            }
            leaves.truncate(leaves.len() / 2);
        }
        leaves[0]
    }

    fn records(
        profile: &FixedSetupOpeningProfileV2,
        point: Vec<EF>,
    ) -> Vec<FixedSetupOpeningProofRecordV2> {
        let claims = profile
            .claims
            .iter()
            .map(|plan| {
                let matrix = &profile.matrices[plan.matrix_index];
                FixedSetupOpeningClaimRecordV2 {
                    setup_index: plan.setup_index,
                    air_id: plan.air_id,
                    sort_idx: plan.sort_idx,
                    part_idx: plan.part_idx,
                    col_idx: plan.col_idx,
                    current: reference_opening(
                        matrix,
                        plan.col_idx as usize,
                        plan.l_skip,
                        &point[..plan.point_len],
                        false,
                    ),
                    rotated: plan.need_rot.then(|| {
                        reference_opening(
                            matrix,
                            plan.col_idx as usize,
                            plan.l_skip,
                            &point[..plan.point_len],
                            true,
                        )
                    }),
                }
            })
            .collect();
        vec![FixedSetupOpeningProofRecordV2 {
            proof_index: 0,
            opening_point: point,
            claims,
        }]
    }

    fn assert_constraints(
        air: &FixedSetupOpeningAirV2,
        main: &RowMajorMatrix<F>,
        preprocessed: &RowMajorMatrix<F>,
    ) {
        check_constraints::<_, NativeSC>(
            air,
            "FixedSetupOpeningAirV2",
            &Some(preprocessed.as_view()),
            &[main.as_view()],
            &[],
        );
    }

    fn rejects_constraints(
        air: &FixedSetupOpeningAirV2,
        main: &RowMajorMatrix<F>,
        preprocessed: &RowMajorMatrix<F>,
    ) -> bool {
        catch_unwind(AssertUnwindSafe(|| {
            assert_constraints(air, main, preprocessed);
        }))
        .is_err()
    }

    #[derive(Clone, Copy, Debug)]
    struct PointSourceAir(FixedSetupOpeningPointBusV2);

    impl BaseAir<F> for PointSourceAir {
        fn width(&self) -> usize {
            4 + D_EF
        }
    }
    impl BaseAirWithPublicValues<F> for PointSourceAir {}
    impl PartitionedBaseAir<F> for PointSourceAir {}

    impl<AB> Air<AB> for PointSourceAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("fixed point source row");
            let active = row[0];
            builder.assert_bool(active);
            self.0.add_key_with_lookups(
                builder,
                FixedSetupOpeningPointMessageV2 {
                    proof_index: row[1].into(),
                    point_index: row[2].into(),
                    value: core::array::from_fn(|limb| row[4 + limb].into()),
                },
                row[3],
            );
            builder
                .when(AB::Expr::ONE - AB::Expr::from(active))
                .assert_zero(row[3]);
        }
    }

    fn point_source_trace(
        profile: &FixedSetupOpeningProfileV2,
        records: &[FixedSetupOpeningProofRecordV2],
    ) -> RowMajorMatrix<F> {
        let width = 4 + D_EF;
        let height = profile
            .point_lookup_demands()
            .len()
            .next_power_of_two()
            .max(2);
        let mut values = F::zero_vec(width * height);
        for (row, &(proof, point_index, count)) in profile.point_lookup_demands().iter().enumerate()
        {
            let record = records
                .iter()
                .find(|record| record.proof_index == proof)
                .expect("point source proof");
            values[row * width] = F::ONE;
            values[row * width + 1] = F::from_u32(proof);
            values[row * width + 2] = F::from_u32(point_index);
            values[row * width + 3] = F::from_u32(count);
            values[row * width + 4..row * width + 4 + D_EF].copy_from_slice(
                record.opening_point[point_index as usize].as_basis_coefficients_slice(),
            );
        }
        RowMajorMatrix::new(values, width)
    }

    fn symbolic_interactions(air: &dyn AnyAir<NativeSC>) -> Vec<SymbolicInteraction<F>> {
        let preprocessed = BaseAir::<F>::preprocessed_trace(air).map(|trace| trace.width());
        get_symbolic_builder(
            air,
            &TraceWidth {
                preprocessed,
                cached_mains: air.cached_main_widths(),
                common_main: air.common_main_width(),
            },
        )
        .constraints()
        .interactions
    }

    fn check_point_bus(
        air: &FixedSetupOpeningAirV2,
        main: &RowMajorMatrix<F>,
        point_source: &RowMajorMatrix<F>,
    ) {
        let source_air = PointSourceAir(air.point_bus);
        let airs: Vec<&dyn AnyAir<NativeSC>> = vec![air, &source_air];
        let preprocessed_owned = vec![air.preprocessed_trace(), None];
        let preprocessed = preprocessed_owned
            .iter()
            .map(|trace| trace.as_ref().map(RowMajorMatrix::as_view))
            .collect::<Vec<_>>();
        let interactions = airs
            .iter()
            .map(|component| {
                symbolic_interactions(*component)
                    .into_iter()
                    .filter(|interaction| interaction.bus_index == air.point_bus.index())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let views = vec![vec![main.as_view()], vec![point_source.as_view()]];
        let public_values = vec![Vec::new(), Vec::new()];
        check_logup(
            &[
                "FixedSetupOpeningAirV2".to_owned(),
                "PointSourceAir".to_owned(),
            ],
            &interactions,
            &preprocessed,
            &views,
            &public_values,
        );
    }

    fn fixture() -> (
        FixedSetupOpeningAirV2,
        Vec<FixedSetupOpeningProofRecordV2>,
        FixedSetupOpeningTraceV2,
        RowMajorMatrix<F>,
    ) {
        let height = 8;
        let width = 2;
        let values = (0..height * width)
            .map(|index| F::from_u32(17 + 13 * index as u32))
            .collect();
        let profile = make_profile(values, 7, digest(100), digest(200), height, width, 1, true);
        let point = vec![EF::from_u32(19), EF::from_u32(23), EF::from_u32(29)];
        let records = records(&profile, point);
        let air = make_air(profile);
        let trace = generate_fixed_setup_opening_trace_v2(&air, &records).expect("honest trace");
        let prep = air.preprocessed_trace().expect("fixed preprocessed trace");
        (air, records, trace, prep)
    }

    #[test]
    fn honest_trace_matches_reference_and_satisfies_air() {
        let (air, records, trace, prep) = fixture();
        assert_eq!(trace.certificates.len(), 1);
        assert_eq!(trace.certificates[0].1, 2);
        assert_eq!(records[0].claims.len(), air.profile.claim_count());
        assert_eq!(
            air.profile.claim_schedule().len(),
            air.profile.claim_count()
        );
        assert_constraints(&air, &trace.matrix, &prep);
    }

    #[test]
    fn tiny_trace_uses_exact_cyclic_ple_wrapping() {
        let values = vec![F::from_u32(5), F::from_u32(41)];
        let profile = make_profile(values, 9, digest(300), digest(400), 2, 1, 4, true);
        let point = vec![EF::from_u32(37)];
        let records = records(&profile, point);
        assert_ne!(
            records[0].claims[0].current,
            records[0].claims[0].rotated.unwrap()
        );
        let air = make_air(profile);
        let trace = generate_fixed_setup_opening_trace_v2(&air, &records).unwrap();
        let prep = air.preprocessed_trace().unwrap();
        assert_constraints(&air, &trace.matrix, &prep);
    }

    #[test]
    fn every_claimed_current_and_rotated_value_is_constrained() {
        let (air, _, trace, prep) = fixture();
        let prep_width = prep.width();
        let main_width = trace.matrix.width();
        let mut roots = Vec::new();
        for row in 0..prep.height() {
            let p: &FixedSetupOpeningPrepColsV2<F> =
                prep.values[row * prep_width..(row + 1) * prep_width].borrow();
            if p.is_root == F::ONE {
                roots.push(row);
            }
        }
        assert_eq!(roots.len(), air.profile.claim_count());
        for &row in &roots {
            for rotated in [false, true] {
                let mut malformed = trace.matrix.clone();
                let cols: &mut FixedSetupOpeningColsV2<F> =
                    malformed.values[row * main_width..(row + 1) * main_width].borrow_mut();
                let target = if rotated {
                    &mut cols.claim_rotated
                } else {
                    &mut cols.claim_current
                };
                target[0] += F::ONE;
                assert!(rejects_constraints(&air, &malformed, &prep));
            }
        }
    }

    #[test]
    fn certified_point_mutation_is_rejected() {
        let (air, _, trace, prep) = fixture();
        let mut malformed = trace.matrix.clone();
        let width = malformed.width();
        let cols: &mut FixedSetupOpeningColsV2<F> = malformed.values[..width].borrow_mut();
        cols.point_a[0] += F::ONE;
        assert!(rejects_constraints(&air, &malformed, &prep));
    }

    #[test]
    fn internally_consistent_wrong_point_fails_certified_point_bus() {
        let (air, honest_records, honest, prep) = fixture();
        let source = point_source_trace(&air.profile, &honest_records);
        check_point_bus(&air, &honest.matrix, &source);

        let mut wrong_point = honest_records[0].opening_point.clone();
        wrong_point[1] += EF::ONE;
        let wrong_records = records(&air.profile, wrong_point);
        let wrong = generate_fixed_setup_opening_trace_v2(&air, &wrong_records).unwrap();
        // This alternative evaluation is algebraically self-consistent.
        assert_constraints(&air, &wrong.matrix, &prep);
        // It nevertheless is not the point exported by the genuine verifier.
        assert!(catch_unwind(AssertUnwindSafe(|| {
            check_point_bus(&air, &wrong.matrix, &source);
        }))
        .is_err());
    }

    #[test]
    fn rotation_selector_air_id_and_vk_digest_are_index_bound() {
        let (air, _, trace, _) = fixture();
        let matrix = air.profile.matrices[0].clone();
        let original_instance = air.profile.instances[0].clone();

        let mut no_rotation = original_instance.clone();
        no_rotation.need_rot = false;
        let bad_rotation = make_air(
            FixedSetupOpeningProfileV2::new(
                air.profile.source_relation_vk_digest,
                vec![matrix.clone()],
                vec![no_rotation],
            )
            .unwrap(),
        );
        assert!(rejects_constraints(
            &bad_rotation,
            &trace.matrix,
            &bad_rotation.preprocessed_trace().unwrap(),
        ));

        let mut wrong_air_matrix = matrix.clone();
        wrong_air_matrix.air_id += 1;
        let wrong_air = make_air(
            FixedSetupOpeningProfileV2::new(
                air.profile.source_relation_vk_digest,
                vec![wrong_air_matrix],
                vec![original_instance.clone()],
            )
            .unwrap(),
        );
        assert!(rejects_constraints(
            &wrong_air,
            &trace.matrix,
            &wrong_air.preprocessed_trace().unwrap(),
        ));

        let wrong_vk = make_air(
            FixedSetupOpeningProfileV2::new(digest(9_000), vec![matrix], vec![original_instance])
                .unwrap(),
        );
        assert!(rejects_constraints(
            &wrong_vk,
            &trace.matrix,
            &wrong_vk.preprocessed_trace().unwrap(),
        ));
    }

    #[test]
    fn equal_shaped_matrix_and_relation_digest_substitution_are_rejected() {
        let (air, _, trace, _) = fixture();
        let mut swapped = air.profile.matrices[0].clone();
        let mut swapped_values = swapped.values.to_vec();
        swapped_values.swap(0, 2);
        swapped.values = swapped_values.into();
        let swapped_air = make_air(
            FixedSetupOpeningProfileV2::new(
                air.profile.source_relation_vk_digest,
                vec![swapped],
                vec![air.profile.instances[0].clone()],
            )
            .unwrap(),
        );
        assert!(rejects_constraints(
            &swapped_air,
            &trace.matrix,
            &swapped_air.preprocessed_trace().unwrap(),
        ));

        let mut wrong_relation = air.profile.matrices[0].clone();
        wrong_relation.relation_digest = digest(8_000);
        let relation_air = make_air(
            FixedSetupOpeningProfileV2::new(
                air.profile.source_relation_vk_digest,
                vec![wrong_relation],
                vec![air.profile.instances[0].clone()],
            )
            .unwrap(),
        );
        assert!(rejects_constraints(
            &relation_air,
            &trace.matrix,
            &relation_air.preprocessed_trace().unwrap(),
        ));
    }

    #[test]
    fn omission_and_duplication_are_rejected_before_witness_authority() {
        let (air, mut records, _, _) = fixture();
        records[0].claims.pop();
        assert_eq!(
            generate_fixed_setup_opening_trace_v2(&air, &records).unwrap_err(),
            FixedSetupOpeningErrorV2::ClaimCount(0),
        );

        let matrix = air.profile.matrices[0].clone();
        let instance = air.profile.instances[0].clone();
        assert_eq!(
            FixedSetupOpeningProfileV2::new(
                air.profile.source_relation_vk_digest,
                vec![matrix],
                vec![instance.clone(), instance],
            )
            .unwrap_err(),
            FixedSetupOpeningErrorV2::InstanceOrder,
        );
    }
}
