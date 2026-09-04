//! Exact protocol-v19 CUDA shared-forest witness adapter for History.
//!
//! CUDA fresh sources are ranges inside one committed forest. They are not
//! ordinary scalar-codeword commitments, so this module deliberately never
//! constructs `MerkleBatchOpeningVerification`. Instead it preserves the full
//! forest descriptor, constrains its exact transcript observation, validates
//! every scalar-to-forest projection, and proves the inner and outer Merkle
//! compression DAGs in a shape-batched dedicated History AIR.

use core::borrow::{Borrow, BorrowMut};
use std::{collections::BTreeMap, sync::Arc};

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit::{
    bus::{Poseidon2PermuteBus, Poseidon2PermuteMessage, TranscriptBus},
    native_warp::{
        generate_native_merkle_leaf_adapter_trace, generate_native_merkle_multiproof_trace,
        NativeAuthenticatedShiftBus, NativeAuthenticatedShiftMessage, NativeInputSlotLayoutBus,
        NativeInputSlotLayoutMessage, NativeLeafValueBus, NativeLeafValueMessage,
        NativeMerkleLeafAdapterAir, NativeMerkleLeafAdapterInput, NativeMerkleMultiproofAir,
        NativeMerkleRootBus, NativeMerkleRootMessage, NativeOpeningLeafBus,
        NativeOpeningLeafMessage, NativeShiftIndexBus, NativeShiftIndexMessage,
        NativeStandardVaccRootBus, NativeStandardVaccRootMessage, NativeVaccTranscriptRoleBus,
        NativeVaccTranscriptRoleMessage, NativeWarpPcdBusInventory, VACC_ROLE_FRESH_ROOT,
    },
    system::BusInventory,
    utils::poseidon2_hash_slice_with_states,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    hasher::MerkleHasher,
    interaction::{InteractionBuilder, LookupBus},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_symmetric::Permutation,
    warp_accum::{
        prune_binary_merkle_paths, verify_binary_merkle_multiproof_recorded,
        BinaryMerkleMultiProof, BinaryMerkleMultiproofRecord, NativeShiftRecord,
        NativeTranscriptPhase, NativeTranscriptPhaseSpan,
    },
    AirRef, AnyAir, BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig,
    TranscriptLog,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_perm, Digest, CHUNK, DIGEST_SIZE, D_EF, EF, F,
};

/// Domain separator used by the CUDA data plane for the logical source-root
/// digest of one physical shared-forest range.
pub const CUDA_SOURCE_ROOT_TAG_V19: u64 = 0x4e57_4355_5254_0013;

/// Backend-neutral copy of the CUDA forest range observed by standard VACC.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaSharedForestCommitmentV19 {
    pub root: Digest,
    pub column_start: u32,
    pub column_width: u32,
    pub forest_width: u32,
    pub rows_per_query: u32,
}

/// Borrowed CUDA opening payload. The rows are full forest rows; retaining
/// only this shard's selected column would not authenticate the shared root.
#[derive(Clone, Copy, Debug)]
pub struct CudaSharedForestOpeningRefV19<'a> {
    pub query_indices: &'a [usize],
    pub opened_rows: &'a [Vec<Vec<EF>>],
    pub authentication_paths: &'a [Vec<Digest>],
}

/// Exact data required to construct a shared-forest History verifier record.
///
/// `shifts` and `transcript_phases` come from the recorded ordinary VACC
/// execution. Consequently, scalar indices and the `FreshCommitments` span are
/// verifier-derived rather than supplied through a second CUDA-local schedule.
pub struct CudaSharedForestVerifierInputV19<'a> {
    pub proof_idx: usize,
    pub log_codeword_len: usize,
    pub commitment: CudaSharedForestCommitmentV19,
    pub shift_answers: Vec<EF>,
    pub opening: CudaSharedForestOpeningRefV19<'a>,
    pub shifts: &'a [NativeShiftRecord<EF>],
    pub transcript_phases: &'a [NativeTranscriptPhaseSpan],
    pub transcript: &'a TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
}

/// One verifier-derived scalar query projected into the shared forest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaSharedForestProjectionV19 {
    pub shift: u32,
    pub scalar_index: u32,
    pub query_index: u32,
    pub local_column: u32,
    pub forest_column: u32,
    pub row_offset: u32,
    /// Base-field position in the full opened forest row.
    pub leaf_position: u32,
    pub value: EF,
}

/// One unique physical forest query and its complete inner row subtree.
pub struct CudaSharedForestUniqueQueryV19<'a> {
    pub query_index: u32,
    pub multiplicity: u32,
    pub opened_rows: &'a [Vec<EF>],
    pub authentication_path: &'a [Digest],
    pub row_digests: Vec<Digest>,
    pub query_digest: Digest,
    /// `None` only when `rows_per_query == 1`, where the row digest is already
    /// the outer leaf.
    pub inner_merkle: Option<BinaryMerkleMultiproofRecord<Digest>>,
}

/// Fully checked adapter record consumed by the dedicated shared-forest AIR.
///
/// This is private witness material. It borrows the exact transcript, full
/// opened rows and paths from the compact SDK receipt and owns only small
/// projection metadata and Merkle compression records.
pub struct CudaSharedForestVerifierRecordV19<'a> {
    pub proof_idx: usize,
    pub log_codeword_len: usize,
    pub codeword_len: usize,
    pub oracle_height: usize,
    pub query_stride: usize,
    pub outer_depth: usize,
    pub commitment: CudaSharedForestCommitmentV19,
    pub fresh_commitments_span: NativeTranscriptPhaseSpan,
    pub projections: Vec<CudaSharedForestProjectionV19>,
    pub unique_queries: Vec<CudaSharedForestUniqueQueryV19<'a>>,
    pub outer_merkle: BinaryMerkleMultiproofRecord<Digest>,
    pub transcript: &'a TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaSharedForestVerifierRecordErrorV19 {
    Empty,
    Shape(&'static str),
    Transcript(&'static str),
    Projection,
    Merkle,
}

/// Authenticate and normalize one CUDA shared-forest opening without changing
/// its commitment semantics.
///
/// The returned Merkle records are generic compression DAGs over the actual
/// full forest leaves. They are not an ordinary scalar-codeword verification.
pub fn prepare_cuda_shared_forest_verifier_record_v19<'a, H>(
    hasher: &H,
    input: CudaSharedForestVerifierInputV19<'a>,
) -> Result<CudaSharedForestVerifierRecordV19<'a>, CudaSharedForestVerifierRecordErrorV19>
where
    H: MerkleHasher<F = F, Digest = Digest>,
{
    if input.shift_answers.is_empty() || input.shifts.is_empty() {
        return Err(CudaSharedForestVerifierRecordErrorV19::Empty);
    }
    if input.proof_idx > u32::MAX as usize {
        return Err(CudaSharedForestVerifierRecordErrorV19::Shape("proof index"));
    }
    let opening = input.opening;
    let query_len = input.shift_answers.len();
    if input.shifts.len() != query_len
        || opening.query_indices.len() != query_len
        || opening.opened_rows.len() != query_len
        || opening.authentication_paths.len() != query_len
    {
        return Err(CudaSharedForestVerifierRecordErrorV19::Shape(
            "query vector lengths",
        ));
    }

    let column_start = input.commitment.column_start as usize;
    let column_width = input.commitment.column_width as usize;
    let forest_width = input.commitment.forest_width as usize;
    let rows_per_query = input.commitment.rows_per_query as usize;
    let codeword_len = 1usize.checked_shl(input.log_codeword_len as u32).ok_or(
        CudaSharedForestVerifierRecordErrorV19::Shape("codeword length"),
    )?;
    if column_width == 0
        || forest_width == 0
        || rows_per_query == 0
        || !rows_per_query.is_power_of_two()
        || !codeword_len.is_multiple_of(column_width)
        || column_start
            .checked_add(column_width)
            .is_none_or(|end| end > forest_width)
    {
        return Err(CudaSharedForestVerifierRecordErrorV19::Shape(
            "forest descriptor",
        ));
    }
    let oracle_height = codeword_len / column_width;
    if !oracle_height.is_power_of_two() || !oracle_height.is_multiple_of(rows_per_query) {
        return Err(CudaSharedForestVerifierRecordErrorV19::Shape(
            "forest oracle height",
        ));
    }
    let query_stride = oracle_height / rows_per_query;
    if !query_stride.is_power_of_two() {
        return Err(CudaSharedForestVerifierRecordErrorV19::Shape(
            "forest query stride",
        ));
    }
    let outer_depth = query_stride.ilog2() as usize;
    let fresh_commitments_span = validate_fresh_commitment_transcript(
        input.transcript,
        input.transcript_phases,
        input.commitment,
    )?;

    let mut projections = Vec::with_capacity(query_len);
    for (shift, ((recorded_shift, &value), &query_index)) in input
        .shifts
        .iter()
        .zip(&input.shift_answers)
        .zip(opening.query_indices)
        .enumerate()
    {
        if recorded_shift.ordinal as usize != shift
            || recorded_shift.fresh_answers.as_slice() != [value]
            || recorded_shift.index as usize >= codeword_len
        {
            return Err(CudaSharedForestVerifierRecordErrorV19::Projection);
        }
        let scalar_index = recorded_shift.index as usize;
        let local_column = scalar_index / oracle_height;
        let row = scalar_index % oracle_height;
        let expected_query = row % query_stride;
        let row_offset = row / query_stride;
        let forest_column = column_start
            .checked_add(local_column)
            .ok_or(CudaSharedForestVerifierRecordErrorV19::Projection)?;
        let rows = &opening.opened_rows[shift];
        if local_column >= column_width
            || forest_column >= forest_width
            || query_index != expected_query
            || rows.len() != rows_per_query
            || rows
                .iter()
                .any(|opened_row| opened_row.len() != forest_width)
            || rows[row_offset][forest_column] != value
            || opening.authentication_paths[shift].len() != outer_depth
        {
            return Err(CudaSharedForestVerifierRecordErrorV19::Projection);
        }
        projections.push(CudaSharedForestProjectionV19 {
            shift: shift
                .try_into()
                .map_err(|_| CudaSharedForestVerifierRecordErrorV19::Projection)?,
            scalar_index: recorded_shift.index,
            query_index: query_index
                .try_into()
                .map_err(|_| CudaSharedForestVerifierRecordErrorV19::Projection)?,
            local_column: local_column
                .try_into()
                .map_err(|_| CudaSharedForestVerifierRecordErrorV19::Projection)?,
            forest_column: forest_column
                .try_into()
                .map_err(|_| CudaSharedForestVerifierRecordErrorV19::Projection)?,
            row_offset: row_offset
                .try_into()
                .map_err(|_| CudaSharedForestVerifierRecordErrorV19::Projection)?,
            leaf_position: forest_column
                .checked_mul(D_EF)
                .and_then(|position| position.try_into().ok())
                .ok_or(CudaSharedForestVerifierRecordErrorV19::Projection)?,
            value,
        });
    }

    let mut unique_by_query = BTreeMap::<usize, usize>::new();
    let mut unique_queries = Vec::<CudaSharedForestUniqueQueryV19<'a>>::new();
    let mut query_digests = Vec::with_capacity(query_len);
    for ordinal in 0..query_len {
        let query_index = opening.query_indices[ordinal];
        let rows = &opening.opened_rows[ordinal];
        let path = &opening.authentication_paths[ordinal];
        let row_digests = rows
            .iter()
            .map(|row| {
                let fields = row
                    .iter()
                    .flat_map(|value| value.as_basis_coefficients_slice().iter().copied())
                    .collect::<Vec<_>>();
                hasher.hash_slice(&fields)
            })
            .collect::<Vec<_>>();
        let (query_digest, inner_merkle) = complete_inner_tree(hasher, &row_digests)?;
        query_digests.push(query_digest);
        if let Some(&unique_ordinal) = unique_by_query.get(&query_index) {
            let unique = &mut unique_queries[unique_ordinal];
            if unique.opened_rows != rows
                || unique.authentication_path != path
                || unique.query_digest != query_digest
            {
                return Err(CudaSharedForestVerifierRecordErrorV19::Merkle);
            }
            unique.multiplicity = unique.multiplicity.checked_add(1).ok_or(
                CudaSharedForestVerifierRecordErrorV19::Shape("query multiplicity"),
            )?;
        } else {
            unique_by_query.insert(query_index, unique_queries.len());
            unique_queries.push(CudaSharedForestUniqueQueryV19 {
                query_index: query_index
                    .try_into()
                    .map_err(|_| CudaSharedForestVerifierRecordErrorV19::Merkle)?,
                multiplicity: 1,
                opened_rows: rows,
                authentication_path: path,
                row_digests,
                query_digest,
                inner_merkle,
            });
        }
    }

    let compact = prune_binary_merkle_paths(
        opening.query_indices,
        opening.authentication_paths,
        outer_depth,
    )
    .map_err(|_| CudaSharedForestVerifierRecordErrorV19::Merkle)?;
    let outer_merkle = verify_binary_merkle_multiproof_recorded(
        hasher,
        input.commitment.root,
        opening.query_indices,
        &query_digests,
        outer_depth,
        &compact,
    )
    .map_err(|_| CudaSharedForestVerifierRecordErrorV19::Merkle)?;

    Ok(CudaSharedForestVerifierRecordV19 {
        proof_idx: input.proof_idx,
        log_codeword_len: input.log_codeword_len,
        codeword_len,
        oracle_height,
        query_stride,
        outer_depth,
        commitment: input.commitment,
        fresh_commitments_span,
        projections,
        unique_queries,
        outer_merkle,
        transcript: input.transcript,
    })
}

fn validate_fresh_commitment_transcript(
    transcript: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    phases: &[NativeTranscriptPhaseSpan],
    commitment: CudaSharedForestCommitmentV19,
) -> Result<NativeTranscriptPhaseSpan, CudaSharedForestVerifierRecordErrorV19> {
    let mut matches = phases
        .iter()
        .filter(|span| span.phase == NativeTranscriptPhase::FreshCommitments);
    let span = matches
        .next()
        .ok_or(CudaSharedForestVerifierRecordErrorV19::Transcript(
            "missing fresh-commitments phase",
        ))?;
    if matches.next().is_some()
        || span.operation_range.end > transcript.len()
        || span.event_range.end > transcript.events().len()
        || span.permutation_range.end > transcript.permutation_transitions().len()
    {
        return Err(CudaSharedForestVerifierRecordErrorV19::Transcript(
            "fresh-commitments phase range",
        ));
    }

    let mut expected = Vec::with_capacity((commitment.root.len() + 4) * D_EF);
    for value in commitment.root {
        push_base_as_extension(&mut expected, value);
    }
    for value in [
        commitment.column_start,
        commitment.column_width,
        commitment.forest_width,
        commitment.rows_per_query,
    ] {
        push_base_as_extension(&mut expected, F::from_u32(value));
    }
    let operations = transcript
        .values()
        .get(span.operation_range.clone())
        .ok_or(CudaSharedForestVerifierRecordErrorV19::Transcript(
            "fresh-commitments operations",
        ))?;
    let samples = transcript
        .samples()
        .get(span.operation_range.clone())
        .ok_or(CudaSharedForestVerifierRecordErrorV19::Transcript(
            "fresh-commitments samples",
        ))?;
    if operations != expected || samples.iter().any(|&is_sample| is_sample) {
        return Err(CudaSharedForestVerifierRecordErrorV19::Transcript(
            "forest descriptor observation",
        ));
    }
    Ok(span.clone())
}

fn push_base_as_extension(output: &mut Vec<F>, value: F) {
    output.push(value);
    output.extend(core::iter::repeat_n(F::ZERO, D_EF - 1));
}

fn complete_inner_tree<H>(
    hasher: &H,
    row_digests: &[Digest],
) -> Result<
    (Digest, Option<BinaryMerkleMultiproofRecord<Digest>>),
    CudaSharedForestVerifierRecordErrorV19,
>
where
    H: MerkleHasher<F = F, Digest = Digest>,
{
    if row_digests.is_empty() || !row_digests.len().is_power_of_two() {
        return Err(CudaSharedForestVerifierRecordErrorV19::Merkle);
    }
    if row_digests.len() == 1 {
        return Ok((row_digests[0], None));
    }
    let mut layer = row_digests.to_vec();
    while layer.len() > 1 {
        layer = layer
            .chunks_exact(2)
            .map(|children| hasher.compress(children[0], children[1]))
            .collect();
    }
    let root = layer[0];
    let indices = (0..row_digests.len()).collect::<Vec<_>>();
    let record = verify_binary_merkle_multiproof_recorded(
        hasher,
        root,
        &indices,
        row_digests,
        row_digests.len().ilog2() as usize,
        &BinaryMerkleMultiProof { siblings: vec![] },
    )
    .map_err(|_| CudaSharedForestVerifierRecordErrorV19::Merkle)?;
    Ok((root, Some(record)))
}

// Keeping descriptor limbs below 2^27 makes every forest-layout identity an
// ordinary integer identity in BabyBear: even `forest_width * D_EF` is below
// the modulus, so modular wraparound cannot fake an in-range forest slice.
const CUDA_FOREST_GEOMETRY_BITS_V19: usize = 27;
// Scalar RS indices need to cover the largest EF4-supported transition class,
// whose rate-1/2 codeword has 2^29 symbols.  These values are never multiplied
// by D_EF; every constrained scalar-index identity remains below 2^29 and thus
// below the BabyBear modulus.  Do not widen descriptor limbs to match this.
const CUDA_FOREST_INDEX_BITS_V19: usize = 29;

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct CudaSharedForestDescriptorColsV19<T> {
    pub root: [T; DIGEST_SIZE],
    pub column_start: T,
    pub column_width: T,
    pub forest_width: T,
    pub rows_per_query: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct CudaSharedForestDescriptorMessageV19<T> {
    pub proof_idx: T,
    pub root: [T; DIGEST_SIZE],
    pub column_start: T,
    pub column_width: T,
    pub forest_width: T,
    pub rows_per_query: T,
}

/// Certified bridge statement emitted only after the physical forest root,
/// range geometry, transcript observation, and Merkle openings have all been
/// checked by the shared-forest verifier.  The descriptor digest is computed
/// in-circuit from the remaining fields; it is not a host-authored authority.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct CudaSharedForestBindingMessageV19<T> {
    pub proof_idx: T,
    pub descriptor_digest: [T; DIGEST_SIZE],
    pub root: [T; DIGEST_SIZE],
    pub column_start: T,
    pub column_width: T,
    pub forest_width: T,
    pub rows_per_query: T,
}

#[derive(Copy, Clone, Debug)]
pub struct CudaSharedForestBindingBusV19(LookupBus);

impl CudaSharedForestBindingBusV19 {
    #[must_use]
    pub fn new(bus_index: openvm_stark_backend::interaction::BusIndex) -> Self {
        Self(LookupBus::new(bus_index))
    }

    pub fn lookup_key<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        key: CudaSharedForestBindingMessageV19<impl Into<AB::Expr> + Clone>,
        enabled: impl Into<AB::Expr>,
    ) {
        self.0.lookup_key(builder, key.to_vec(), enabled);
    }

    pub fn add_key_with_lookups<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        key: CudaSharedForestBindingMessageV19<impl Into<AB::Expr> + Clone>,
        lookups: impl Into<AB::Expr>,
    ) {
        self.0.add_key_with_lookups(builder, key.to_vec(), lookups);
    }
}

/// Internal lookup binding every projection and full-row hash to the one
/// descriptor that is both transcript-observed and Merkle-authenticated.
#[derive(Copy, Clone, Debug)]
pub struct CudaSharedForestDescriptorBusV19(LookupBus);

impl CudaSharedForestDescriptorBusV19 {
    #[must_use]
    pub fn new(bus_index: openvm_stark_backend::interaction::BusIndex) -> Self {
        Self(LookupBus::new(bus_index))
    }

    pub fn lookup_key<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        key: CudaSharedForestDescriptorMessageV19<impl Into<AB::Expr> + Clone>,
        enabled: impl Into<AB::Expr>,
    ) {
        self.0.lookup_key(builder, key.to_vec(), enabled);
    }

    pub fn add_key_with_lookups<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        key: CudaSharedForestDescriptorMessageV19<impl Into<AB::Expr> + Clone>,
        lookups: impl Into<AB::Expr>,
    ) {
        self.0.add_key_with_lookups(builder, key.to_vec(), lookups);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaSharedForestVerifierProfileV19 {
    pub log_codeword_len: usize,
    pub column_width: usize,
    pub rows_per_query: usize,
    pub outer_tree_id: usize,
    pub row_tree_id_offset: usize,
    /// Standard arity-two fresh-slot variant: one without prior, four with it.
    pub input_variant: usize,
}

impl CudaSharedForestVerifierProfileV19 {
    pub fn validate(&self) -> Result<(), &'static str> {
        let codeword_len = 1usize
            .checked_shl(self.log_codeword_len as u32)
            .ok_or("CUDA forest codeword length")?;
        if self.log_codeword_len > CUDA_FOREST_INDEX_BITS_V19
            || self.column_width == 0
            || self.column_width >= (1usize << CUDA_FOREST_GEOMETRY_BITS_V19)
            || self.rows_per_query == 0
            || !self.rows_per_query.is_power_of_two()
            || !codeword_len.is_multiple_of(self.column_width)
            || !matches!(self.input_variant, 1 | 4)
            || self.outer_tree_id > u32::MAX as usize
            || self.row_tree_id_offset > u32::MAX as usize
        {
            return Err("CUDA shared-forest verifier profile");
        }
        let oracle_height = codeword_len / self.column_width;
        let query_stride = oracle_height
            .checked_div(self.rows_per_query)
            .filter(|stride| stride.is_power_of_two())
            .ok_or("CUDA forest query stride")?;
        let row_tree_end = self
            .row_tree_id_offset
            .checked_add(query_stride)
            .ok_or("CUDA forest row-tree range")?;
        if query_stride == 0
            || row_tree_end > u32::MAX as usize + 1
            || (self.outer_tree_id >= self.row_tree_id_offset && self.outer_tree_id < row_tree_end)
        {
            return Err("CUDA forest row-tree range");
        }
        Ok(())
    }

    #[must_use]
    pub fn codeword_len(&self) -> usize {
        1usize << self.log_codeword_len
    }

    #[must_use]
    pub fn oracle_height(&self) -> usize {
        self.codeword_len() / self.column_width
    }

    #[must_use]
    pub fn query_stride(&self) -> usize {
        self.oracle_height() / self.rows_per_query
    }

    #[must_use]
    pub fn outer_depth(&self) -> usize {
        self.query_stride().ilog2() as usize
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct CudaSharedForestRootColsV19<T> {
    pub active: T,
    pub proof_idx: T,
    pub tidx: T,
    pub descriptor: CudaSharedForestDescriptorColsV19<T>,
    pub remaining_width: T,
    pub column_start_bits: [T; CUDA_FOREST_GEOMETRY_BITS_V19],
    pub forest_width_bits: [T; CUDA_FOREST_GEOMETRY_BITS_V19],
    pub remaining_width_bits: [T; CUDA_FOREST_GEOMETRY_BITS_V19],
    pub descriptor_lookup_count: T,
    pub descriptor_lookup_count_inverse: T,
    pub descriptor_digest: [T; DIGEST_SIZE],
    pub descriptor_hash_pre: [[T; POSEIDON2_WIDTH]; 2],
    pub descriptor_hash_post: [[T; POSEIDON2_WIDTH]; 2],
}

#[derive(ColumnsAir)]
#[columns_via(CudaSharedForestRootColsV19<u8>)]
pub struct CudaSharedForestRootAirV19 {
    pub transcript_bus: TranscriptBus,
    /// Present only in the integrated CUDA VACC verifier.  It proves that the
    /// raw transcript operations consumed here are exactly the first standard
    /// VACC role, including all four forest-range limbs.
    pub transcript_role_bus: Option<NativeVaccTranscriptRoleBus>,
    pub opening_leaf_bus: NativeOpeningLeafBus,
    pub merkle_root_bus: NativeMerkleRootBus,
    pub statement_root_bus: NativeStandardVaccRootBus,
    pub descriptor_bus: CudaSharedForestDescriptorBusV19,
    pub binding_bus: CudaSharedForestBindingBusV19,
    pub permute_bus: Poseidon2PermuteBus,
    pub profile: CudaSharedForestVerifierProfileV19,
}

impl BaseAirWithPublicValues<F> for CudaSharedForestRootAirV19 {}
impl PartitionedBaseAir<F> for CudaSharedForestRootAirV19 {}
impl BaseAir<F> for CudaSharedForestRootAirV19 {
    fn width(&self) -> usize {
        CudaSharedForestRootColsV19::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for CudaSharedForestRootAirV19 {
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        let main = builder.main();
        let local_row = main.row_slice(0).expect("CUDA shared-forest root row");
        let next_row = main.row_slice(1).expect("CUDA shared-forest root next row");
        let local: &CudaSharedForestRootColsV19<AB::Var> = (*local_row).borrow();
        let next: &CudaSharedForestRootColsV19<AB::Var> = (*next_row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_zero(local.proof_idx);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .when(next.active)
            .assert_eq(next.proof_idx, local.proof_idx + AB::F::ONE);
        builder.when(local.active).assert_eq(
            local.descriptor.column_width,
            AB::Expr::from_usize(self.profile.column_width),
        );
        builder.when(local.active).assert_eq(
            local.descriptor.rows_per_query,
            AB::Expr::from_usize(self.profile.rows_per_query),
        );
        builder
            .when(local.active)
            .assert_one(local.descriptor_lookup_count * local.descriptor_lookup_count_inverse);
        for bits in [
            local.column_start_bits,
            local.forest_width_bits,
            local.remaining_width_bits,
        ] {
            for bit in bits {
                builder.assert_bool(bit);
            }
        }
        let recompose = |bits: [AB::Var; CUDA_FOREST_GEOMETRY_BITS_V19]| {
            bits.into_iter()
                .enumerate()
                .fold(AB::Expr::ZERO, |value, (bit, flag)| {
                    value + flag * AB::Expr::from_usize(1usize << bit)
                })
        };
        builder.when(local.active).assert_eq(
            local.descriptor.column_start,
            recompose(local.column_start_bits),
        );
        builder.when(local.active).assert_eq(
            local.descriptor.forest_width,
            recompose(local.forest_width_bits),
        );
        builder
            .when(local.active)
            .assert_eq(local.remaining_width, recompose(local.remaining_width_bits));
        builder.when(local.active).assert_eq(
            local.descriptor.column_start + local.descriptor.column_width + local.remaining_width,
            local.descriptor.forest_width,
        );

        let mut tidx: AB::Expr = local.tidx.into();
        let mut ordinal = 0usize;
        for value in local.descriptor.root {
            self.transcript_bus.observe_ext(
                builder,
                local.proof_idx,
                tidx.clone(),
                [value.into(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO],
                local.active,
            );
            if let Some(role_bus) = self.transcript_role_bus {
                role_bus.receive(
                    builder,
                    NativeVaccTranscriptRoleMessage {
                        proof_idx: local.proof_idx.into(),
                        role: AB::Expr::from_usize(VACC_ROLE_FRESH_ROOT),
                        ordinal: AB::Expr::from_usize(ordinal),
                        tidx: tidx.clone(),
                        value: [value.into(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO],
                        is_ext: AB::Expr::ONE,
                        is_sample: AB::Expr::ZERO,
                    },
                    local.active,
                );
            }
            tidx += AB::Expr::from_usize(D_EF);
            ordinal += 1;
        }
        for value in [
            local.descriptor.column_start,
            local.descriptor.column_width,
            local.descriptor.forest_width,
            local.descriptor.rows_per_query,
        ] {
            self.transcript_bus.observe_ext(
                builder,
                local.proof_idx,
                tidx.clone(),
                [value.into(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO],
                local.active,
            );
            if let Some(role_bus) = self.transcript_role_bus {
                role_bus.receive(
                    builder,
                    NativeVaccTranscriptRoleMessage {
                        proof_idx: local.proof_idx.into(),
                        role: AB::Expr::from_usize(VACC_ROLE_FRESH_ROOT),
                        ordinal: AB::Expr::from_usize(ordinal),
                        tidx: tidx.clone(),
                        value: [value.into(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO],
                        is_ext: AB::Expr::ONE,
                        is_sample: AB::Expr::ZERO,
                    },
                    local.active,
                );
            }
            tidx += AB::Expr::from_usize(D_EF);
            ordinal += 1;
        }

        if self.profile.outer_depth() == 0 {
            self.opening_leaf_bus.receive(
                builder,
                NativeOpeningLeafMessage {
                    proof_idx: local.proof_idx.into(),
                    tree_id: AB::Expr::from_usize(self.profile.outer_tree_id),
                    index: AB::Expr::ZERO,
                    digest: local.descriptor.root.map(Into::into),
                },
                local.active,
            );
        } else {
            self.merkle_root_bus.receive(
                builder,
                NativeMerkleRootMessage {
                    proof_idx: local.proof_idx.into(),
                    tree_id: AB::Expr::from_usize(self.profile.outer_tree_id),
                    depth: AB::Expr::from_usize(self.profile.outer_depth()),
                    digest: local.descriptor.root.map(Into::into),
                },
                local.active,
            );
        }
        self.statement_root_bus.add_key_with_lookups(
            builder,
            NativeStandardVaccRootMessage {
                proof_idx: local.proof_idx.into(),
                kind: AB::Expr::ZERO,
                root: local.descriptor.root.map(Into::into),
            },
            local.active,
        );
        self.descriptor_bus.add_key_with_lookups(
            builder,
            descriptor_message(local.proof_idx, &local.descriptor),
            local.descriptor_lookup_count,
        );
        let descriptor_words = [
            AB::Expr::from_u64(CUDA_SOURCE_ROOT_TAG_V19),
            local.descriptor.root[0].into(),
            local.descriptor.root[1].into(),
            local.descriptor.root[2].into(),
            local.descriptor.root[3].into(),
            local.descriptor.root[4].into(),
            local.descriptor.root[5].into(),
            local.descriptor.root[6].into(),
            local.descriptor.root[7].into(),
            local.descriptor.column_start.into(),
            local.descriptor.column_width.into(),
            local.descriptor.forest_width.into(),
            local.descriptor.rows_per_query.into(),
        ];
        let descriptor_word_count = descriptor_words.len();
        for (chunk, word) in descriptor_words.into_iter().enumerate() {
            let round = chunk / CHUNK;
            let lane = chunk % CHUNK;
            builder
                .when(local.active)
                .assert_eq(local.descriptor_hash_pre[round][lane], word);
        }
        for lane in CHUNK..POSEIDON2_WIDTH {
            builder
                .when(local.active)
                .assert_zero(local.descriptor_hash_pre[0][lane]);
        }
        for lane in descriptor_word_count % CHUNK..POSEIDON2_WIDTH {
            builder.when(local.active).assert_eq(
                local.descriptor_hash_pre[1][lane],
                local.descriptor_hash_post[0][lane],
            );
        }
        for round in 0..2 {
            self.permute_bus.lookup_key(
                builder,
                Poseidon2PermuteMessage {
                    input: local.descriptor_hash_pre[round].map(Into::into),
                    output: local.descriptor_hash_post[round].map(Into::into),
                },
                local.active,
            );
        }
        for limb in 0..DIGEST_SIZE {
            builder.when(local.active).assert_eq(
                local.descriptor_digest[limb],
                local.descriptor_hash_post[1][limb],
            );
        }
        self.binding_bus.add_key_with_lookups(
            builder,
            CudaSharedForestBindingMessageV19 {
                proof_idx: local.proof_idx.into(),
                descriptor_digest: local.descriptor_digest.map(Into::into),
                root: local.descriptor.root.map(Into::into),
                column_start: local.descriptor.column_start.into(),
                column_width: local.descriptor.column_width.into(),
                forest_width: local.descriptor.forest_width.into(),
                rows_per_query: local.descriptor.rows_per_query.into(),
            },
            local.active,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct CudaSharedForestProjectionColsV19<T> {
    pub active: T,
    pub proof_idx: T,
    pub shift: T,
    pub scalar_index: T,
    pub local_column: T,
    pub query_index: T,
    pub row_offset: T,
    pub forest_column: T,
    pub leaf_position: T,
    pub local_column_remaining: T,
    pub scalar_index_bits: [T; CUDA_FOREST_INDEX_BITS_V19],
    pub local_column_bits: [T; CUDA_FOREST_INDEX_BITS_V19],
    pub local_column_remaining_bits: [T; CUDA_FOREST_INDEX_BITS_V19],
    pub query_index_bits: [T; CUDA_FOREST_INDEX_BITS_V19],
    pub row_offset_bits: [T; CUDA_FOREST_INDEX_BITS_V19],
    pub value: [T; D_EF],
    pub descriptor: CudaSharedForestDescriptorColsV19<T>,
}

#[derive(ColumnsAir)]
#[columns_via(CudaSharedForestProjectionColsV19<u8>)]
pub struct CudaSharedForestProjectionAirV19 {
    pub shift_index_bus: NativeShiftIndexBus,
    pub leaf_value_bus: NativeLeafValueBus,
    pub authenticated_bus: NativeAuthenticatedShiftBus,
    pub slot_bus: NativeInputSlotLayoutBus,
    pub descriptor_bus: CudaSharedForestDescriptorBusV19,
    pub profile: CudaSharedForestVerifierProfileV19,
}

impl BaseAirWithPublicValues<F> for CudaSharedForestProjectionAirV19 {}
impl PartitionedBaseAir<F> for CudaSharedForestProjectionAirV19 {}
impl BaseAir<F> for CudaSharedForestProjectionAirV19 {
    fn width(&self) -> usize {
        CudaSharedForestProjectionColsV19::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for CudaSharedForestProjectionAirV19 {
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("CUDA shared-forest projection row");
        let next_row = main
            .row_slice(1)
            .expect("CUDA shared-forest projection next row");
        let local: &CudaSharedForestProjectionColsV19<AB::Var> = (*row).borrow();
        let next: &CudaSharedForestProjectionColsV19<AB::Var> = (*next_row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_zero(local.proof_idx);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .when(next.active)
            .assert_bool(next.proof_idx - local.proof_idx);
        builder.when(local.active).assert_eq(
            local.descriptor.column_width,
            AB::Expr::from_usize(self.profile.column_width),
        );
        builder.when(local.active).assert_eq(
            local.descriptor.rows_per_query,
            AB::Expr::from_usize(self.profile.rows_per_query),
        );
        self.descriptor_bus.lookup_key(
            builder,
            descriptor_message(local.proof_idx, &local.descriptor),
            local.active,
        );
        self.shift_index_bus.lookup_key(
            builder,
            NativeShiftIndexMessage {
                proof_idx: local.proof_idx.into(),
                shift: local.shift.into(),
                index: local.scalar_index.into(),
            },
            local.active,
        );
        self.slot_bus.lookup_key(
            builder,
            NativeInputSlotLayoutMessage {
                variant: AB::Expr::from_usize(self.profile.input_variant),
                source: AB::Expr::ZERO,
                kind: [AB::Expr::ONE, AB::Expr::ZERO, AB::Expr::ZERO],
            },
            local.active,
        );
        for bits in [
            &local.scalar_index_bits,
            &local.local_column_bits,
            &local.local_column_remaining_bits,
            &local.query_index_bits,
            &local.row_offset_bits,
        ] {
            for bit in bits {
                builder.assert_bool(bit.clone());
            }
        }
        let recompose = |bits: &[AB::Var; CUDA_FOREST_INDEX_BITS_V19]| {
            bits.iter()
                .enumerate()
                .fold(AB::Expr::ZERO, |value, (bit, flag)| {
                    value + flag.clone() * AB::Expr::from_usize(1usize << bit)
                })
        };
        builder
            .when(local.active)
            .assert_eq(local.scalar_index, recompose(&local.scalar_index_bits));
        builder
            .when(local.active)
            .assert_eq(local.local_column, recompose(&local.local_column_bits));
        builder.when(local.active).assert_eq(
            local.local_column_remaining,
            recompose(&local.local_column_remaining_bits),
        );
        builder
            .when(local.active)
            .assert_eq(local.query_index, recompose(&local.query_index_bits));
        builder
            .when(local.active)
            .assert_eq(local.row_offset, recompose(&local.row_offset_bits));
        for bit in self.profile.log_codeword_len..CUDA_FOREST_INDEX_BITS_V19 {
            builder
                .when(local.active)
                .assert_zero(local.scalar_index_bits[bit]);
        }
        for bit in self.profile.outer_depth()..CUDA_FOREST_INDEX_BITS_V19 {
            builder
                .when(local.active)
                .assert_zero(local.query_index_bits[bit]);
        }
        let log_rows_per_query = self.profile.rows_per_query.ilog2() as usize;
        for bit in log_rows_per_query..CUDA_FOREST_INDEX_BITS_V19 {
            builder
                .when(local.active)
                .assert_zero(local.row_offset_bits[bit]);
        }
        builder.when(local.active).assert_eq(
            local.local_column + local.local_column_remaining + AB::Expr::ONE,
            AB::Expr::from_usize(self.profile.column_width),
        );
        builder.when(local.active).assert_eq(
            local.scalar_index,
            local.local_column * AB::Expr::from_usize(self.profile.oracle_height())
                + local.query_index
                + local.row_offset * AB::Expr::from_usize(self.profile.query_stride()),
        );
        builder.when(local.active).assert_eq(
            local.forest_column,
            local.descriptor.column_start + local.local_column,
        );
        builder.when(local.active).assert_eq(
            local.leaf_position,
            local.forest_column * AB::Expr::from_usize(D_EF),
        );
        let inner_tree_id =
            AB::Expr::from_usize(self.profile.row_tree_id_offset) + local.query_index;
        for limb in 0..D_EF {
            self.leaf_value_bus.lookup_key(
                builder,
                NativeLeafValueMessage {
                    proof_idx: local.proof_idx.into(),
                    tree_id: inner_tree_id.clone(),
                    index: local.row_offset.into(),
                    position: local.leaf_position + AB::Expr::from_usize(limb),
                    value: local.value[limb].into(),
                },
                local.active,
            );
        }
        self.authenticated_bus.send(
            builder,
            NativeAuthenticatedShiftMessage {
                proof_idx: local.proof_idx.into(),
                shift: local.shift.into(),
                source: AB::Expr::ZERO,
                value: local.value.map(Into::into),
            },
            local.active,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct CudaSharedForestLeafHashColsV19<T> {
    pub active: T,
    pub proof_idx: T,
    pub tree_id: T,
    pub leaf_index: T,
    pub block: T,
    pub is_first: T,
    pub is_last: T,
    pub mask: [T; CHUNK],
    pub lookup_count: [T; CHUNK],
    pub before: [T; POSEIDON2_WIDTH],
    pub input: [T; POSEIDON2_WIDTH],
    pub output: [T; POSEIDON2_WIDTH],
    pub descriptor: CudaSharedForestDescriptorColsV19<T>,
}

#[derive(ColumnsAir)]
#[columns_via(CudaSharedForestLeafHashColsV19<u8>)]
pub struct CudaSharedForestLeafHashAirV19 {
    pub permute_bus: Poseidon2PermuteBus,
    pub value_bus: NativeLeafValueBus,
    pub leaf_bus: NativeOpeningLeafBus,
    pub descriptor_bus: CudaSharedForestDescriptorBusV19,
    pub profile: CudaSharedForestVerifierProfileV19,
}

impl BaseAirWithPublicValues<F> for CudaSharedForestLeafHashAirV19 {}
impl PartitionedBaseAir<F> for CudaSharedForestLeafHashAirV19 {}
impl BaseAir<F> for CudaSharedForestLeafHashAirV19 {
    fn width(&self) -> usize {
        CudaSharedForestLeafHashColsV19::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for CudaSharedForestLeafHashAirV19 {
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        let main = builder.main();
        let local_row = main.row_slice(0).expect("CUDA forest leaf hash row");
        let next_row = main.row_slice(1).expect("CUDA forest leaf hash next row");
        let local: &CudaSharedForestLeafHashColsV19<AB::Var> = (*local_row).borrow();
        let next: &CudaSharedForestLeafHashColsV19<AB::Var> = (*next_row).borrow();
        for flag in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        let mut mask_sum = AB::Expr::ZERO;
        for index in 0..CHUNK {
            builder.assert_bool(local.mask[index]);
            mask_sum += local.mask[index];
            if index + 1 < CHUNK {
                builder
                    .when(local.active * (AB::Expr::ONE - local.mask[index]))
                    .assert_zero(local.mask[index + 1]);
            }
            builder
                .when(local.active * (AB::Expr::ONE - local.is_last))
                .assert_one(local.mask[index]);
            builder
                .when(local.active * (AB::Expr::ONE - local.mask[index]))
                .assert_eq(local.input[index], local.before[index]);
            builder
                .when(AB::Expr::ONE - local.mask[index])
                .assert_zero(local.lookup_count[index]);
            self.value_bus.add_key_with_lookups(
                builder,
                NativeLeafValueMessage {
                    proof_idx: local.proof_idx.into(),
                    tree_id: local.tree_id.into(),
                    index: local.leaf_index.into(),
                    position: local.block * AB::Expr::from_usize(CHUNK)
                        + AB::Expr::from_usize(index),
                    value: local.input[index].into(),
                },
                local.lookup_count[index],
            );
        }
        builder
            .when(local.active * local.is_last)
            .assert_one(local.mask[0]);
        builder.when(local.active * local.is_last).assert_eq(
            local.block * AB::Expr::from_usize(CHUNK) + mask_sum,
            local.descriptor.forest_width * AB::Expr::from_usize(D_EF),
        );
        for index in CHUNK..POSEIDON2_WIDTH {
            builder
                .when(local.active)
                .assert_eq(local.input[index], local.before[index]);
        }
        for value in local.before {
            builder
                .when(local.active * local.is_first)
                .assert_zero(value);
        }
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.block);
        builder.when(local.active * local.is_first).assert_eq(
            local.descriptor.column_width,
            AB::Expr::from_usize(self.profile.column_width),
        );
        builder.when(local.active * local.is_first).assert_eq(
            local.descriptor.rows_per_query,
            AB::Expr::from_usize(self.profile.rows_per_query),
        );
        self.descriptor_bus.lookup_key(
            builder,
            descriptor_message(local.proof_idx, &local.descriptor),
            local.active * local.is_first,
        );

        let same_leaf = next.active * (AB::Expr::ONE - next.is_first);
        let mut transition = builder.when_transition();
        let mut same = transition.when(same_leaf);
        same.assert_zero(local.is_last);
        same.assert_eq(next.proof_idx, local.proof_idx);
        same.assert_eq(next.tree_id, local.tree_id);
        same.assert_eq(next.leaf_index, local.leaf_index);
        same.assert_eq(next.block, local.block + AB::F::ONE);
        for index in 0..POSEIDON2_WIDTH {
            same.assert_eq(next.before[index], local.output[index]);
        }
        assert_descriptor_eq(&mut same, &next.descriptor, &local.descriptor);
        let starts_leaf = next.active * next.is_first;
        builder
            .when_transition()
            .when(starts_leaf.clone())
            .assert_one(local.is_last);
        builder
            .when_transition()
            .when(starts_leaf)
            .assert_bool(next.proof_idx - local.proof_idx);
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_one(local.is_last);

        self.permute_bus.lookup_key(
            builder,
            Poseidon2PermuteMessage {
                input: local.input.map(Into::into),
                output: local.output.map(Into::into),
            },
            local.active,
        );
        self.leaf_bus.send(
            builder,
            NativeOpeningLeafMessage {
                proof_idx: local.proof_idx.into(),
                tree_id: local.tree_id.into(),
                index: local.leaf_index.into(),
                digest: core::array::from_fn(|index| local.output[index].into()),
            },
            local.active * local.is_last,
        );
    }
}

fn descriptor_message<T: Clone>(
    proof_idx: T,
    descriptor: &CudaSharedForestDescriptorColsV19<T>,
) -> CudaSharedForestDescriptorMessageV19<T> {
    CudaSharedForestDescriptorMessageV19 {
        proof_idx,
        root: descriptor.root.clone(),
        column_start: descriptor.column_start.clone(),
        column_width: descriptor.column_width.clone(),
        forest_width: descriptor.forest_width.clone(),
        rows_per_query: descriptor.rows_per_query.clone(),
    }
}

fn assert_descriptor_eq<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    left: &CudaSharedForestDescriptorColsV19<AB::Var>,
    right: &CudaSharedForestDescriptorColsV19<AB::Var>,
) {
    for index in 0..DIGEST_SIZE {
        builder.assert_eq(left.root[index].clone(), right.root[index].clone());
    }
    builder.assert_eq(left.column_start.clone(), right.column_start.clone());
    builder.assert_eq(left.column_width.clone(), right.column_width.clone());
    builder.assert_eq(left.forest_width.clone(), right.forest_width.clone());
    builder.assert_eq(left.rows_per_query.clone(), right.rows_per_query.clone());
}

pub struct CudaSharedForestVerifierModuleV19 {
    pub profile: CudaSharedForestVerifierProfileV19,
    pub shared: BusInventory,
    pub buses: NativeWarpPcdBusInventory,
    pub statement_root_bus: NativeStandardVaccRootBus,
    pub descriptor_bus: CudaSharedForestDescriptorBusV19,
    pub binding_bus: CudaSharedForestBindingBusV19,
    integrated_vacc_transcript: bool,
}

impl CudaSharedForestVerifierModuleV19 {
    pub fn new(
        profile: CudaSharedForestVerifierProfileV19,
        shared: BusInventory,
        buses: NativeWarpPcdBusInventory,
        statement_root_bus: NativeStandardVaccRootBus,
        descriptor_bus: CudaSharedForestDescriptorBusV19,
        binding_bus: CudaSharedForestBindingBusV19,
    ) -> Result<Self, &'static str> {
        profile.validate()?;
        Ok(Self {
            profile,
            shared,
            buses,
            statement_root_bus,
            descriptor_bus,
            binding_bus,
            integrated_vacc_transcript: false,
        })
    }

    /// Construct the production shared-forest verifier as the fresh source
    /// lane of one standard VACC verifier.  The forest root AIR consumes the
    /// semantic transcript emitted by the proof-indexed VACC cursor and also
    /// consumes the cursor's exact fresh-commitment role messages.
    pub fn new_vacc_integrated(
        profile: CudaSharedForestVerifierProfileV19,
        shared: BusInventory,
        buses: NativeWarpPcdBusInventory,
        statement_root_bus: NativeStandardVaccRootBus,
        descriptor_bus: CudaSharedForestDescriptorBusV19,
        binding_bus: CudaSharedForestBindingBusV19,
    ) -> Result<Self, &'static str> {
        let mut module = Self::new(
            profile,
            shared,
            buses,
            statement_root_bus,
            descriptor_bus,
            binding_bus,
        )?;
        module.integrated_vacc_transcript = true;
        Ok(module)
    }

    /// AIRs in the exact order emitted by [`Self::generate_traces`].
    #[must_use]
    pub fn airs<PCS: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<PCS>> {
        let mut airs = Vec::with_capacity(5);
        add_cuda_forest_air(
            &mut airs,
            CudaSharedForestRootAirV19 {
                transcript_bus: if self.integrated_vacc_transcript {
                    self.buses.vacc_semantic_transcript
                } else {
                    self.buses.transcript
                },
                transcript_role_bus: self
                    .integrated_vacc_transcript
                    .then_some(self.buses.vacc_transcript_role),
                opening_leaf_bus: self.buses.opening_leaf,
                merkle_root_bus: self.buses.merkle_root,
                statement_root_bus: self.statement_root_bus,
                descriptor_bus: self.descriptor_bus,
                binding_bus: self.binding_bus,
                permute_bus: self.shared.poseidon2_permute_bus,
                profile: self.profile.clone(),
            },
        );
        add_cuda_forest_air(
            &mut airs,
            CudaSharedForestProjectionAirV19 {
                shift_index_bus: self.buses.shift_index,
                leaf_value_bus: self.buses.leaf_value,
                authenticated_bus: self.buses.authenticated_shift,
                slot_bus: self.buses.input_slot_layout,
                descriptor_bus: self.descriptor_bus,
                profile: self.profile.clone(),
            },
        );
        add_cuda_forest_air(
            &mut airs,
            CudaSharedForestLeafHashAirV19 {
                permute_bus: self.shared.poseidon2_permute_bus,
                value_bus: self.buses.leaf_value,
                leaf_bus: self.buses.opening_leaf,
                descriptor_bus: self.descriptor_bus,
                profile: self.profile.clone(),
            },
        );
        add_cuda_forest_air(
            &mut airs,
            NativeMerkleMultiproofAir {
                compress_bus: self.shared.poseidon2_compress_bus,
                leaf_bus: self.buses.opening_leaf,
                node_bus: self.buses.merkle_node,
                root_bus: self.buses.merkle_root,
            },
        );
        add_cuda_forest_air(
            &mut airs,
            NativeMerkleLeafAdapterAir {
                leaf_bus: self.buses.opening_leaf,
                root_bus: self.buses.merkle_root,
                inner_depth: self.profile.rows_per_query.ilog2() as usize,
            },
        );
        airs
    }

    pub fn generate_trace(
        &self,
        record: &CudaSharedForestVerifierRecordV19<'_>,
    ) -> Result<CudaSharedForestVerifierTraceV19, CudaSharedForestVerifierRecordErrorV19> {
        if record.proof_idx != 0 {
            return Err(CudaSharedForestVerifierRecordErrorV19::Shape(
                "single-record proof index",
            ));
        }
        self.generate_traces(core::slice::from_ref(record))
    }

    /// Generate one physical five-AIR witness for all forest records sharing
    /// this numeric shape. Proof identifiers are dense and local to this
    /// shape group, exactly like the shape-batched ordinary VACC verifier.
    ///
    /// Tree identifiers may repeat across proofs because every leaf, node and
    /// root bus key includes `proof_idx`. Within one proof the outer tree and
    /// the complete row-tree range are disjoint by profile validation.
    pub fn generate_traces(
        &self,
        records: &[CudaSharedForestVerifierRecordV19<'_>],
    ) -> Result<CudaSharedForestVerifierTraceV19, CudaSharedForestVerifierRecordErrorV19> {
        let records = records.iter().collect::<Vec<_>>();
        self.generate_traces_from_refs(&records)
    }

    /// Borrowing batch entry point used by the CUDA VACC composite.  It avoids
    /// cloning opened forest rows, Merkle paths, or compression records merely
    /// to join independently prepared continuations-side records.
    pub fn generate_traces_from_refs(
        &self,
        records: &[&CudaSharedForestVerifierRecordV19<'_>],
    ) -> Result<CudaSharedForestVerifierTraceV19, CudaSharedForestVerifierRecordErrorV19> {
        if records.is_empty() {
            return Err(CudaSharedForestVerifierRecordErrorV19::Empty);
        }
        if records.len() >= (1usize << CUDA_FOREST_GEOMETRY_BITS_V19) {
            return Err(CudaSharedForestVerifierRecordErrorV19::Shape(
                "forest proof batch",
            ));
        }
        let mut generated = Vec::with_capacity(records.len());
        for (proof_idx, record) in records.iter().enumerate() {
            if record.proof_idx != proof_idx {
                return Err(CudaSharedForestVerifierRecordErrorV19::Shape(
                    "non-canonical group-local proof index",
                ));
            }
            generated.push(self.generate_record_traces(record)?);
        }

        let widths = [
            CudaSharedForestRootColsV19::<F>::width(),
            CudaSharedForestProjectionColsV19::<F>::width(),
            CudaSharedForestLeafHashColsV19::<F>::width(),
            <NativeMerkleMultiproofAir as BaseAir<F>>::width(&NativeMerkleMultiproofAir {
                compress_bus: self.shared.poseidon2_compress_bus,
                leaf_bus: self.buses.opening_leaf,
                node_bus: self.buses.merkle_node,
                root_bus: self.buses.merkle_root,
            }),
            <NativeMerkleLeafAdapterAir as BaseAir<F>>::width(&NativeMerkleLeafAdapterAir {
                leaf_bus: self.buses.opening_leaf,
                root_bus: self.buses.merkle_root,
                inner_depth: self.profile.rows_per_query.ilog2() as usize,
            }),
        ];
        let mut traces = Vec::with_capacity(widths.len());
        for (matrix_index, width) in widths.into_iter().enumerate() {
            traces.push(merge_cuda_shared_forest_matrices(
                generated.iter().map(|record| &record.traces[matrix_index]),
                width,
            )?);
        }
        Ok(CudaSharedForestVerifierTraceV19 {
            traces,
            poseidon_permutation_inputs: generated
                .iter()
                .flat_map(|record| record.poseidon_permutation_inputs.iter().copied())
                .collect(),
            poseidon_compression_inputs: generated
                .iter()
                .flat_map(|record| record.poseidon_compression_inputs.iter().copied())
                .collect(),
        })
    }

    fn generate_record_traces(
        &self,
        record: &CudaSharedForestVerifierRecordV19<'_>,
    ) -> Result<CudaSharedForestVerifierTraceV19, CudaSharedForestVerifierRecordErrorV19> {
        self.validate_record(record)?;
        let descriptor_lookups = record
            .projections
            .len()
            .checked_add(record.unique_queries.len() * self.profile.rows_per_query)
            .ok_or(CudaSharedForestVerifierRecordErrorV19::Shape(
                "descriptor lookup count",
            ))?;
        let (root, descriptor_permutation_inputs) =
            generate_cuda_shared_forest_root_trace(&self.profile, record, descriptor_lookups)?;
        let projection = generate_cuda_shared_forest_projection_trace(&self.profile, record, None)?;
        let owned_leaves = cuda_shared_forest_owned_leaves(&self.profile, record)?;
        let leaf_hash =
            generate_cuda_shared_forest_leaf_hash_trace(record.commitment, &owned_leaves, None)?;

        let mut records = Vec::<(u32, &BinaryMerkleMultiproofRecord<Digest>)>::new();
        for query in &record.unique_queries {
            if let Some(inner) = &query.inner_merkle {
                records.push((
                    self.profile
                        .row_tree_id_offset
                        .checked_add(query.query_index as usize)
                        .and_then(|tree| tree.try_into().ok())
                        .ok_or(CudaSharedForestVerifierRecordErrorV19::Shape(
                            "inner tree id",
                        ))?,
                    inner,
                ));
            }
        }
        records.push((
            self.profile
                .outer_tree_id
                .try_into()
                .map_err(|_| CudaSharedForestVerifierRecordErrorV19::Shape("outer tree id"))?,
            &record.outer_merkle,
        ));
        let multiproof_air = NativeMerkleMultiproofAir {
            compress_bus: self.shared.poseidon2_compress_bus,
            leaf_bus: self.buses.opening_leaf,
            node_bus: self.buses.merkle_node,
            root_bus: self.buses.merkle_root,
        };
        let merkle = if records
            .iter()
            .any(|(_, proof)| !proof.compressions.is_empty())
        {
            generate_native_merkle_multiproof_trace(record.proof_idx, &records, None)
                .ok_or(CudaSharedForestVerifierRecordErrorV19::Merkle)?
        } else {
            let width = <NativeMerkleMultiproofAir as BaseAir<F>>::width(&multiproof_air);
            RowMajorMatrix::new(F::zero_vec(width), width)
        };
        let adapters = record
            .unique_queries
            .iter()
            .map(|query| {
                Ok(NativeMerkleLeafAdapterInput {
                    proof_idx: record.proof_idx.try_into().map_err(|_| {
                        CudaSharedForestVerifierRecordErrorV19::Shape("proof index")
                    })?,
                    bypass: self.profile.rows_per_query == 1,
                    outer_multiplicity: query.multiplicity,
                    inner_tree_id: self
                        .profile
                        .row_tree_id_offset
                        .checked_add(query.query_index as usize)
                        .and_then(|tree| tree.try_into().ok())
                        .ok_or(CudaSharedForestVerifierRecordErrorV19::Shape(
                            "inner tree id",
                        ))?,
                    outer_tree_id: self.profile.outer_tree_id.try_into().map_err(|_| {
                        CudaSharedForestVerifierRecordErrorV19::Shape("outer tree id")
                    })?,
                    query_index: query.query_index,
                    inner_depth: self.profile.rows_per_query.ilog2(),
                    digest: query.query_digest,
                })
            })
            .collect::<Result<Vec<_>, CudaSharedForestVerifierRecordErrorV19>>()?;
        let leaf_adapter = generate_native_merkle_leaf_adapter_trace(
            &adapters,
            self.profile.rows_per_query.ilog2() as usize,
            None,
        )
        .ok_or(CudaSharedForestVerifierRecordErrorV19::Merkle)?;
        let poseidon_compression_inputs = records
            .iter()
            .flat_map(|(_, record)| {
                record.compressions.iter().map(|compression| {
                    core::array::from_fn(|index| {
                        if index < DIGEST_SIZE {
                            compression.left[index]
                        } else {
                            compression.right[index - DIGEST_SIZE]
                        }
                    })
                })
            })
            .collect();
        Ok(CudaSharedForestVerifierTraceV19 {
            traces: vec![root, projection, leaf_hash.matrix, merkle, leaf_adapter],
            poseidon_permutation_inputs: descriptor_permutation_inputs
                .into_iter()
                .chain(leaf_hash.permutation_inputs)
                .collect(),
            poseidon_compression_inputs,
        })
    }

    fn validate_record(
        &self,
        record: &CudaSharedForestVerifierRecordV19<'_>,
    ) -> Result<(), CudaSharedForestVerifierRecordErrorV19> {
        if record.proof_idx >= (1usize << CUDA_FOREST_GEOMETRY_BITS_V19)
            || record.log_codeword_len != self.profile.log_codeword_len
            || record.commitment.column_width as usize != self.profile.column_width
            || record.commitment.rows_per_query as usize != self.profile.rows_per_query
            || record.codeword_len != self.profile.codeword_len()
            || record.oracle_height != self.profile.oracle_height()
            || record.query_stride != self.profile.query_stride()
            || record.outer_depth != self.profile.outer_depth()
            || record.commitment.column_start >= (1u32 << CUDA_FOREST_GEOMETRY_BITS_V19)
            || record.commitment.forest_width >= (1u32 << CUDA_FOREST_GEOMETRY_BITS_V19)
            || record.outer_merkle.expected_root != record.commitment.root
        {
            return Err(CudaSharedForestVerifierRecordErrorV19::Shape(
                "AIR/profile mismatch",
            ));
        }
        Ok(())
    }
}

fn add_cuda_forest_air<PCS, A>(airs: &mut Vec<AirRef<PCS>>, air: A)
where
    PCS: StarkProtocolConfig,
    A: AnyAir<PCS> + 'static,
{
    airs.push(Arc::new(air));
}

pub struct CudaSharedForestVerifierTraceV19 {
    pub traces: Vec<RowMajorMatrix<F>>,
    pub poseidon_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub poseidon_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

fn merge_cuda_shared_forest_matrices<'a>(
    matrices: impl IntoIterator<Item = &'a RowMajorMatrix<F>>,
    width: usize,
) -> Result<RowMajorMatrix<F>, CudaSharedForestVerifierRecordErrorV19> {
    if width == 0 {
        return Err(CudaSharedForestVerifierRecordErrorV19::Shape(
            "zero trace width",
        ));
    }
    let mut active = Vec::new();
    for matrix in matrices {
        if matrix.width() != width || !matrix.values.len().is_multiple_of(width) {
            return Err(CudaSharedForestVerifierRecordErrorV19::Shape(
                "batched trace width",
            ));
        }
        for row in matrix.values.chunks_exact(width) {
            if row[0] == F::ZERO {
                break;
            }
            active.extend_from_slice(row);
        }
    }
    let active_rows = active.len() / width;
    let height = active_rows.next_power_of_two().max(1);
    active.resize(height * width, F::ZERO);
    Ok(RowMajorMatrix::new(active, width))
}

#[derive(Clone, Debug)]
struct CudaSharedForestOwnedLeafV19 {
    proof_idx: u32,
    tree_id: u32,
    leaf_index: u32,
    values: Vec<F>,
    lookup_counts: Vec<u32>,
}

struct CudaSharedForestLeafHashTraceV19 {
    matrix: RowMajorMatrix<F>,
    permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

fn generate_cuda_shared_forest_root_trace(
    profile: &CudaSharedForestVerifierProfileV19,
    record: &CudaSharedForestVerifierRecordV19<'_>,
    descriptor_lookups: usize,
) -> Result<(RowMajorMatrix<F>, Vec<[F; POSEIDON2_WIDTH]>), CudaSharedForestVerifierRecordErrorV19>
{
    let width = CudaSharedForestRootColsV19::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut CudaSharedForestRootColsV19<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.proof_idx = F::from_usize(record.proof_idx);
    cols.tidx = F::from_usize(record.fresh_commitments_span.operation_range.start);
    set_descriptor(&mut cols.descriptor, record.commitment);
    let remaining = (record.commitment.forest_width as usize)
        .checked_sub(record.commitment.column_start as usize + profile.column_width)
        .ok_or(CudaSharedForestVerifierRecordErrorV19::Shape(
            "forest range remainder",
        ))?;
    cols.remaining_width = F::from_usize(remaining);
    set_geometry_bits(
        &mut cols.column_start_bits,
        record.commitment.column_start as usize,
    )?;
    set_geometry_bits(
        &mut cols.forest_width_bits,
        record.commitment.forest_width as usize,
    )?;
    set_geometry_bits(&mut cols.remaining_width_bits, remaining)?;
    cols.descriptor_lookup_count = F::from_usize(descriptor_lookups);
    cols.descriptor_lookup_count_inverse = cols.descriptor_lookup_count.inverse();
    let mut words = vec![F::from_u64(CUDA_SOURCE_ROOT_TAG_V19)];
    words.extend_from_slice(&record.commitment.root);
    words.extend([
        F::from_u32(record.commitment.column_start),
        F::from_u32(record.commitment.column_width),
        F::from_u32(record.commitment.forest_width),
        F::from_u32(record.commitment.rows_per_query),
    ]);
    let (digest, pre, post) = poseidon2_hash_slice_with_states(&words);
    if pre.len() != 2 || post.len() != 2 {
        return Err(CudaSharedForestVerifierRecordErrorV19::Shape(
            "CUDA source descriptor sponge shape",
        ));
    }
    cols.descriptor_digest = digest;
    cols.descriptor_hash_pre.copy_from_slice(&pre);
    cols.descriptor_hash_post.copy_from_slice(&post);
    Ok((RowMajorMatrix::new(values, width), pre))
}

fn generate_cuda_shared_forest_projection_trace(
    profile: &CudaSharedForestVerifierProfileV19,
    record: &CudaSharedForestVerifierRecordV19<'_>,
    required_height: Option<usize>,
) -> Result<RowMajorMatrix<F>, CudaSharedForestVerifierRecordErrorV19> {
    let height = required_height
        .unwrap_or_else(|| record.projections.len().next_power_of_two())
        .max(1);
    if height < record.projections.len() {
        return Err(CudaSharedForestVerifierRecordErrorV19::Shape(
            "projection trace height",
        ));
    }
    let width = CudaSharedForestProjectionColsV19::<F>::width();
    let mut values = F::zero_vec(height * width);
    for (row, projection) in record.projections.iter().enumerate() {
        let cols: &mut CudaSharedForestProjectionColsV19<F> =
            values[row * width..(row + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_usize(record.proof_idx);
        cols.shift = F::from_u32(projection.shift);
        cols.scalar_index = F::from_u32(projection.scalar_index);
        cols.local_column = F::from_u32(projection.local_column);
        cols.query_index = F::from_u32(projection.query_index);
        cols.row_offset = F::from_u32(projection.row_offset);
        cols.forest_column = F::from_u32(projection.forest_column);
        cols.leaf_position = F::from_u32(projection.leaf_position);
        let local_column_remaining = profile
            .column_width
            .checked_sub(projection.local_column as usize + 1)
            .ok_or(CudaSharedForestVerifierRecordErrorV19::Projection)?;
        cols.local_column_remaining = F::from_usize(local_column_remaining);
        set_geometry_bits(
            &mut cols.scalar_index_bits,
            projection.scalar_index as usize,
        )?;
        set_geometry_bits(
            &mut cols.local_column_bits,
            projection.local_column as usize,
        )?;
        set_geometry_bits(
            &mut cols.local_column_remaining_bits,
            local_column_remaining,
        )?;
        set_geometry_bits(&mut cols.query_index_bits, projection.query_index as usize)?;
        set_geometry_bits(&mut cols.row_offset_bits, projection.row_offset as usize)?;
        cols.value
            .copy_from_slice(projection.value.as_basis_coefficients_slice());
        set_descriptor(&mut cols.descriptor, record.commitment);
    }
    Ok(RowMajorMatrix::new(values, width))
}

fn cuda_shared_forest_owned_leaves(
    profile: &CudaSharedForestVerifierProfileV19,
    record: &CudaSharedForestVerifierRecordV19<'_>,
) -> Result<Vec<CudaSharedForestOwnedLeafV19>, CudaSharedForestVerifierRecordErrorV19> {
    let mut selected = BTreeMap::<(u32, u32, u32), u32>::new();
    for projection in &record.projections {
        let count = selected
            .entry((
                projection.query_index,
                projection.row_offset,
                projection.forest_column,
            ))
            .or_default();
        *count = count
            .checked_add(1)
            .ok_or(CudaSharedForestVerifierRecordErrorV19::Shape(
                "leaf lookup multiplicity",
            ))?;
    }
    let mut leaves = Vec::new();
    for query in &record.unique_queries {
        let tree_id = profile
            .row_tree_id_offset
            .checked_add(query.query_index as usize)
            .and_then(|tree| tree.try_into().ok())
            .ok_or(CudaSharedForestVerifierRecordErrorV19::Shape("row tree id"))?;
        for (row_offset, row) in query.opened_rows.iter().enumerate() {
            let values = row
                .iter()
                .flat_map(|value| value.as_basis_coefficients_slice().iter().copied())
                .collect::<Vec<_>>();
            let mut lookup_counts = vec![0u32; values.len()];
            for column in 0..row.len() {
                let count = selected
                    .get(&(query.query_index, row_offset as u32, column as u32))
                    .copied()
                    .unwrap_or(0);
                for limb in 0..D_EF {
                    lookup_counts[column * D_EF + limb] = count;
                }
            }
            leaves.push(CudaSharedForestOwnedLeafV19 {
                proof_idx: record
                    .proof_idx
                    .try_into()
                    .map_err(|_| CudaSharedForestVerifierRecordErrorV19::Shape("proof index"))?,
                tree_id,
                leaf_index: row_offset
                    .try_into()
                    .map_err(|_| CudaSharedForestVerifierRecordErrorV19::Shape("row offset"))?,
                values,
                lookup_counts,
            });
        }
    }
    Ok(leaves)
}

fn generate_cuda_shared_forest_leaf_hash_trace(
    commitment: CudaSharedForestCommitmentV19,
    leaves: &[CudaSharedForestOwnedLeafV19],
    required_height: Option<usize>,
) -> Result<CudaSharedForestLeafHashTraceV19, CudaSharedForestVerifierRecordErrorV19> {
    if leaves.is_empty()
        || leaves.iter().any(|leaf| {
            leaf.values.len() != commitment.forest_width as usize * D_EF
                || leaf.values.len() != leaf.lookup_counts.len()
        })
    {
        return Err(CudaSharedForestVerifierRecordErrorV19::Shape(
            "full forest row",
        ));
    }
    let valid_rows = leaves
        .iter()
        .map(|leaf| leaf.values.len().div_ceil(CHUNK))
        .sum::<usize>();
    let height = required_height
        .unwrap_or_else(|| valid_rows.next_power_of_two())
        .max(1);
    if height < valid_rows {
        return Err(CudaSharedForestVerifierRecordErrorV19::Shape(
            "leaf hash trace height",
        ));
    }
    let width = CudaSharedForestLeafHashColsV19::<F>::width();
    let mut trace = F::zero_vec(height * width);
    let mut permutation_inputs = Vec::with_capacity(valid_rows);
    let mut trace_row = 0usize;
    for leaf in leaves {
        let blocks = leaf.values.len().div_ceil(CHUNK);
        let mut state = [F::ZERO; POSEIDON2_WIDTH];
        for block in 0..blocks {
            let before = state;
            let start = block * CHUNK;
            let end = leaf.values.len().min(start + CHUNK);
            let fields = &leaf.values[start..end];
            for (target, &field) in state.iter_mut().zip(fields) {
                *target = field;
            }
            let input = state;
            poseidon2_perm().permute_mut(&mut state);
            let cols: &mut CudaSharedForestLeafHashColsV19<F> =
                trace[trace_row * width..(trace_row + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.proof_idx = F::from_u32(leaf.proof_idx);
            cols.tree_id = F::from_u32(leaf.tree_id);
            cols.leaf_index = F::from_u32(leaf.leaf_index);
            cols.block = F::from_usize(block);
            cols.is_first = F::from_bool(block == 0);
            cols.is_last = F::from_bool(block + 1 == blocks);
            for target in cols.mask.iter_mut().take(fields.len()) {
                *target = F::ONE;
            }
            for (target, &count) in cols
                .lookup_count
                .iter_mut()
                .zip(&leaf.lookup_counts[start..end])
            {
                *target = F::from_u32(count);
            }
            cols.before = before;
            cols.input = input;
            cols.output = state;
            set_descriptor(&mut cols.descriptor, commitment);
            permutation_inputs.push(input);
            trace_row += 1;
        }
    }
    Ok(CudaSharedForestLeafHashTraceV19 {
        matrix: RowMajorMatrix::new(trace, width),
        permutation_inputs,
    })
}

fn set_descriptor(
    target: &mut CudaSharedForestDescriptorColsV19<F>,
    commitment: CudaSharedForestCommitmentV19,
) {
    target.root = commitment.root;
    target.column_start = F::from_u32(commitment.column_start);
    target.column_width = F::from_u32(commitment.column_width);
    target.forest_width = F::from_u32(commitment.forest_width);
    target.rows_per_query = F::from_u32(commitment.rows_per_query);
}

fn set_geometry_bits<const N: usize>(
    target: &mut [F; N],
    value: usize,
) -> Result<(), CudaSharedForestVerifierRecordErrorV19> {
    if value >= (1usize << N) {
        return Err(CudaSharedForestVerifierRecordErrorV19::Shape(
            "forest geometry range",
        ));
    }
    for (bit, target) in target.iter_mut().enumerate() {
        *target = F::from_bool(((value >> bit) & 1) == 1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use openvm_recursion_circuit::system::BusIndexManager;
    use openvm_stark_backend::{
        air_builders::debug::check_constraints, p3_matrix::dense::RowMajorMatrixView,
        StarkProtocolConfig, SystemParams,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

    use super::*;

    #[test]
    fn profile_separates_large_scalar_indices_from_bounded_forest_geometry() {
        let largest_ef4_profile = CudaSharedForestVerifierProfileV19 {
            log_codeword_len: 29,
            column_width: 1,
            rows_per_query: 1,
            outer_tree_id: 0,
            row_tree_id_offset: 1,
            input_variant: 1,
        };
        assert!(largest_ef4_profile.validate().is_ok());
        assert_eq!(largest_ef4_profile.outer_depth(), 29);

        let beyond_ef4_profile = CudaSharedForestVerifierProfileV19 {
            log_codeword_len: 30,
            ..largest_ef4_profile
        };
        assert_eq!(
            beyond_ef4_profile.validate(),
            Err("CUDA shared-forest verifier profile")
        );
    }

    struct OwnedForestFixture {
        config: BabyBearPoseidon2Config,
        commitment: CudaSharedForestCommitmentV19,
        transcript: TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
        phases: Vec<NativeTranscriptPhaseSpan>,
        shifts: Vec<NativeShiftRecord<EF>>,
        shift_answers: Vec<EF>,
        query_indices: Vec<usize>,
        opened_rows: Vec<Vec<Vec<EF>>>,
        authentication_paths: Vec<Vec<Digest>>,
    }

    fn owned_forest_fixture() -> OwnedForestFixture {
        let config = BabyBearPoseidon2Config::default_from_params(SystemParams::new_for_testing(6));
        let hasher = config.hasher();
        let all_rows = (0..8)
            .map(|row| {
                (0..2)
                    .map(|column| EF::from(F::from_usize(10 * row + column + 1)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let row_digest = |row: &[EF]| {
            hasher.hash_slice(
                &row.iter()
                    .flat_map(|value| value.as_basis_coefficients_slice().iter().copied())
                    .collect::<Vec<_>>(),
            )
        };
        let leaves = (0..4)
            .map(|query| {
                hasher.compress(
                    row_digest(&all_rows[query]),
                    row_digest(&all_rows[query + 4]),
                )
            })
            .collect::<Vec<_>>();
        let parents = [
            hasher.compress(leaves[0], leaves[1]),
            hasher.compress(leaves[2], leaves[3]),
        ];
        let root = hasher.compress(parents[0], parents[1]);
        let commitment = CudaSharedForestCommitmentV19 {
            root,
            column_start: 1,
            column_width: 1,
            forest_width: 2,
            rows_per_query: 2,
        };
        let mut observed = Vec::new();
        for value in commitment.root {
            push_base_as_extension(&mut observed, value);
        }
        for value in [
            commitment.column_start,
            commitment.column_width,
            commitment.forest_width,
            commitment.rows_per_query,
        ] {
            push_base_as_extension(&mut observed, F::from_u32(value));
        }
        let query_indices = vec![1usize, 2];
        let opened_rows = vec![
            vec![all_rows[1].clone(), all_rows[5].clone()],
            vec![all_rows[2].clone(), all_rows[6].clone()],
        ];
        let authentication_paths = vec![vec![leaves[0], parents[1]], vec![leaves[3], parents[0]]];
        let shift_answers = vec![all_rows[1][1], all_rows[6][1]];
        let shifts = vec![
            NativeShiftRecord {
                ordinal: 0,
                index: 1,
                boolean_point: vec![],
                fresh_answers: vec![shift_answers[0]],
                prior_answer: None,
                merged_answer: EF::ZERO,
            },
            NativeShiftRecord {
                ordinal: 1,
                index: 6,
                boolean_point: vec![],
                fresh_answers: vec![shift_answers[1]],
                prior_answer: None,
                merged_answer: EF::ZERO,
            },
        ];
        OwnedForestFixture {
            config,
            commitment,
            transcript: TranscriptLog::new(observed.clone(), vec![false; observed.len()]),
            phases: vec![NativeTranscriptPhaseSpan {
                phase: NativeTranscriptPhase::FreshCommitments,
                event_range: 0..0,
                operation_range: 0..observed.len(),
                permutation_range: 0..0,
            }],
            shifts,
            shift_answers,
            query_indices,
            opened_rows,
            authentication_paths,
        }
    }

    fn prepare_owned_fixture<'a>(
        fixture: &'a OwnedForestFixture,
        proof_idx: usize,
        commitment: CudaSharedForestCommitmentV19,
    ) -> Result<CudaSharedForestVerifierRecordV19<'a>, CudaSharedForestVerifierRecordErrorV19> {
        prepare_cuda_shared_forest_verifier_record_v19(
            fixture.config.hasher(),
            CudaSharedForestVerifierInputV19 {
                proof_idx,
                log_codeword_len: 3,
                commitment,
                shift_answers: fixture.shift_answers.clone(),
                opening: CudaSharedForestOpeningRefV19 {
                    query_indices: &fixture.query_indices,
                    opened_rows: &fixture.opened_rows,
                    authentication_paths: &fixture.authentication_paths,
                },
                shifts: &fixture.shifts,
                transcript_phases: &fixture.phases,
                transcript: &fixture.transcript,
            },
        )
    }

    #[test]
    fn successful_multi_shard_shared_forest_composition_authenticates_full_rows() {
        let config = BabyBearPoseidon2Config::default_from_params(SystemParams::new_for_testing(6));
        let hasher = config.hasher();
        let all_rows = (0..8)
            .map(|row| {
                (0..2)
                    .map(|column| EF::from(F::from_usize(10 * row + column + 1)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let row_digest = |row: &[EF]| {
            let fields = row
                .iter()
                .flat_map(|value| value.as_basis_coefficients_slice().iter().copied())
                .collect::<Vec<_>>();
            hasher.hash_slice(&fields)
        };
        let leaves = (0..4)
            .map(|query| {
                hasher.compress(
                    row_digest(&all_rows[query]),
                    row_digest(&all_rows[query + 4]),
                )
            })
            .collect::<Vec<_>>();
        let parents = [
            hasher.compress(leaves[0], leaves[1]),
            hasher.compress(leaves[2], leaves[3]),
        ];
        let root = hasher.compress(parents[0], parents[1]);
        let path = |query: usize| vec![leaves[query ^ 1], parents[(query >> 1) ^ 1]];
        let opened_rows = vec![
            vec![all_rows[1].clone(), all_rows[5].clone()],
            vec![all_rows[2].clone(), all_rows[6].clone()],
        ];
        let query_indices = [1usize, 2];
        let authentication_paths = vec![path(1), path(2)];
        let commitment = CudaSharedForestCommitmentV19 {
            root,
            column_start: 1,
            column_width: 1,
            forest_width: 2,
            rows_per_query: 2,
        };
        let mut observed = Vec::new();
        for value in commitment.root {
            push_base_as_extension(&mut observed, value);
        }
        for value in [
            commitment.column_start,
            commitment.column_width,
            commitment.forest_width,
            commitment.rows_per_query,
        ] {
            push_base_as_extension(&mut observed, F::from_u32(value));
        }
        let transcript = TranscriptLog::new(observed.clone(), vec![false; observed.len()]);
        let phases = [NativeTranscriptPhaseSpan {
            phase: NativeTranscriptPhase::FreshCommitments,
            event_range: 0..0,
            operation_range: 0..observed.len(),
            permutation_range: 0..0,
        }];
        let shifts = [
            NativeShiftRecord {
                ordinal: 0,
                index: 1,
                boolean_point: vec![],
                fresh_answers: vec![all_rows[1][1]],
                prior_answer: None,
                merged_answer: EF::ZERO,
            },
            NativeShiftRecord {
                ordinal: 1,
                index: 6,
                boolean_point: vec![],
                fresh_answers: vec![all_rows[6][1]],
                prior_answer: None,
                merged_answer: EF::ZERO,
            },
        ];
        let record = prepare_cuda_shared_forest_verifier_record_v19(
            hasher,
            CudaSharedForestVerifierInputV19 {
                proof_idx: 0,
                log_codeword_len: 3,
                commitment,
                shift_answers: vec![all_rows[1][1], all_rows[6][1]],
                opening: CudaSharedForestOpeningRefV19 {
                    query_indices: &query_indices,
                    opened_rows: &opened_rows,
                    authentication_paths: &authentication_paths,
                },
                shifts: &shifts,
                transcript_phases: &phases,
                transcript: &transcript,
            },
        )
        .expect("valid shared forest record");
        assert_eq!(record.commitment, commitment);
        assert_eq!(record.outer_merkle.expected_root, root);
        assert_eq!(record.projections.len(), 2);
        assert_eq!(record.unique_queries.len(), 2);
        assert_eq!(record.projections[0].forest_column, 1);
        assert_eq!(record.projections[1].row_offset, 1);
        assert!(record
            .unique_queries
            .iter()
            .all(|query| query.inner_merkle.is_some()));

        let mut manager = BusIndexManager::new();
        let shared = BusInventory::new(&mut manager);
        let buses = NativeWarpPcdBusInventory::new(manager.next_bus_idx());
        let mut extra = BusIndexManager::from_next_bus_idx(buses.next_bus_idx());
        let module = CudaSharedForestVerifierModuleV19::new(
            CudaSharedForestVerifierProfileV19 {
                log_codeword_len: 3,
                column_width: 1,
                rows_per_query: 2,
                outer_tree_id: 0,
                row_tree_id_offset: 1,
                input_variant: 1,
            },
            shared,
            buses,
            NativeStandardVaccRootBus::new(extra.new_bus_idx()),
            CudaSharedForestDescriptorBusV19::new(extra.new_bus_idx()),
            CudaSharedForestBindingBusV19::new(extra.new_bus_idx()),
        )
        .expect("valid shared-forest verifier shape");

        let second = prepare_cuda_shared_forest_verifier_record_v19(
            hasher,
            CudaSharedForestVerifierInputV19 {
                proof_idx: 1,
                log_codeword_len: 3,
                commitment,
                shift_answers: vec![all_rows[1][1], all_rows[6][1]],
                opening: CudaSharedForestOpeningRefV19 {
                    query_indices: &query_indices,
                    opened_rows: &opened_rows,
                    authentication_paths: &authentication_paths,
                },
                shifts: &shifts,
                transcript_phases: &phases,
                transcript: &transcript,
            },
        )
        .expect("second valid shared forest record");
        let batch = module
            .generate_traces(&[record, second])
            .expect("shape-batched shared forest traces");
        assert_eq!(batch.traces.len(), 5);
        let root_width = CudaSharedForestRootColsV19::<F>::width();
        let first_root: &CudaSharedForestRootColsV19<F> =
            batch.traces[0].values[..root_width].borrow();
        let second_root: &CudaSharedForestRootColsV19<F> =
            batch.traces[0].values[root_width..2 * root_width].borrow();
        assert_eq!(first_root.proof_idx, F::ZERO);
        assert_eq!(second_root.proof_idx, F::ONE);
        for (air, matrix) in module
            .airs::<BabyBearPoseidon2Config>()
            .iter()
            .zip(&batch.traces)
        {
            check_constraints::<_, BabyBearPoseidon2Config>(
                air.as_ref(),
                &air.name(),
                &None,
                &[RowMajorMatrixView::new(&matrix.values, matrix.width())],
                &[],
            );
        }
    }

    #[test]
    fn rejects_descriptor_root_splice() {
        let fixture = owned_forest_fixture();
        let mut commitment = fixture.commitment;
        commitment.root[0] += F::ONE;
        assert!(matches!(
            prepare_owned_fixture(&fixture, 0, commitment),
            Err(CudaSharedForestVerifierRecordErrorV19::Transcript(
                "forest descriptor observation"
            ))
        ));
    }

    #[test]
    fn rejects_nonzero_descriptor_hash_initial_capacity() {
        let fixture = owned_forest_fixture();
        let record = prepare_owned_fixture(&fixture, 0, fixture.commitment).expect("forest record");
        let mut manager = BusIndexManager::new();
        let shared = BusInventory::new(&mut manager);
        let buses = NativeWarpPcdBusInventory::new(manager.next_bus_idx());
        let mut extra = BusIndexManager::from_next_bus_idx(buses.next_bus_idx());
        let module = CudaSharedForestVerifierModuleV19::new(
            CudaSharedForestVerifierProfileV19 {
                log_codeword_len: 3,
                column_width: 1,
                rows_per_query: 2,
                outer_tree_id: 0,
                row_tree_id_offset: 1,
                input_variant: 1,
            },
            shared,
            buses,
            NativeStandardVaccRootBus::new(extra.new_bus_idx()),
            CudaSharedForestDescriptorBusV19::new(extra.new_bus_idx()),
            CudaSharedForestBindingBusV19::new(extra.new_bus_idx()),
        )
        .expect("module");
        let mut traces = module.generate_trace(&record).expect("forest traces");
        let cols: &mut CudaSharedForestRootColsV19<F> =
            traces.traces[0].values.as_mut_slice().borrow_mut();
        cols.descriptor_hash_pre[0][CHUNK] = F::ONE;
        let root_air = module
            .airs::<BabyBearPoseidon2Config>()
            .into_iter()
            .next()
            .expect("root AIR");
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_constraints::<_, BabyBearPoseidon2Config>(
                root_air.as_ref(),
                &root_air.name(),
                &None,
                &[RowMajorMatrixView::new(
                    &traces.traces[0].values,
                    traces.traces[0].width(),
                )],
                &[],
            );
        }));
        assert!(rejected.is_err(), "a prover-selected sponge IV must fail");
    }

    #[test]
    fn rejects_column_range_width_and_rows_per_query_tampering() {
        let fixture = owned_forest_fixture();
        for commitment in [
            CudaSharedForestCommitmentV19 {
                column_start: 0,
                ..fixture.commitment
            },
            CudaSharedForestCommitmentV19 {
                column_width: 2,
                ..fixture.commitment
            },
            CudaSharedForestCommitmentV19 {
                forest_width: 3,
                ..fixture.commitment
            },
            CudaSharedForestCommitmentV19 {
                rows_per_query: 1,
                ..fixture.commitment
            },
        ] {
            assert!(prepare_owned_fixture(&fixture, 0, commitment).is_err());
        }
    }

    #[test]
    fn rejects_cross_proof_forest_record_splice() {
        let fixture = owned_forest_fixture();
        let first = prepare_owned_fixture(&fixture, 0, fixture.commitment).expect("first record");
        let duplicated =
            prepare_owned_fixture(&fixture, 0, fixture.commitment).expect("duplicated record");
        let mut manager = BusIndexManager::new();
        let shared = BusInventory::new(&mut manager);
        let buses = NativeWarpPcdBusInventory::new(manager.next_bus_idx());
        let mut extra = BusIndexManager::from_next_bus_idx(buses.next_bus_idx());
        let module = CudaSharedForestVerifierModuleV19::new(
            CudaSharedForestVerifierProfileV19 {
                log_codeword_len: 3,
                column_width: 1,
                rows_per_query: 2,
                outer_tree_id: 0,
                row_tree_id_offset: 1,
                input_variant: 1,
            },
            shared,
            buses,
            NativeStandardVaccRootBus::new(extra.new_bus_idx()),
            CudaSharedForestDescriptorBusV19::new(extra.new_bus_idx()),
            CudaSharedForestBindingBusV19::new(extra.new_bus_idx()),
        )
        .expect("module");
        let error = match module.generate_traces_from_refs(&[&first, &duplicated]) {
            Ok(_) => panic!("duplicate local proof id must fail"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            CudaSharedForestVerifierRecordErrorV19::Shape("non-canonical group-local proof index")
        );
    }

    #[test]
    fn rejects_descriptor_not_observed_by_vacc_transcript() {
        let config = BabyBearPoseidon2Config::default_from_params(SystemParams::new_for_testing(6));
        let transcript = TranscriptLog::default();
        let query_indices = [0];
        let opened_rows = [vec![vec![EF::ZERO]]];
        let authentication_paths = [vec![[F::ZERO; DIGEST_SIZE]; 3]];
        let shifts = [NativeShiftRecord {
            ordinal: 0,
            index: 0,
            boolean_point: vec![],
            fresh_answers: vec![EF::ZERO],
            prior_answer: None,
            merged_answer: EF::ZERO,
        }];
        let result = prepare_cuda_shared_forest_verifier_record_v19(
            config.hasher(),
            CudaSharedForestVerifierInputV19 {
                proof_idx: 0,
                log_codeword_len: 3,
                commitment: CudaSharedForestCommitmentV19 {
                    root: [F::ZERO; 8],
                    column_start: 0,
                    column_width: 1,
                    forest_width: 1,
                    rows_per_query: 1,
                },
                shift_answers: vec![EF::ZERO],
                opening: CudaSharedForestOpeningRefV19 {
                    query_indices: &query_indices,
                    opened_rows: &opened_rows,
                    authentication_paths: &authentication_paths,
                },
                shifts: &shifts,
                transcript_phases: &[],
                transcript: &transcript,
            },
        );
        let error = match result {
            Ok(_) => panic!("an unobserved forest descriptor must be rejected"),
            Err(error) => error,
        };
        assert_eq!(
            error,
            CudaSharedForestVerifierRecordErrorV19::Transcript("missing fresh-commitments phase")
        );
    }
}
