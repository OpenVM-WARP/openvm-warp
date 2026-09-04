use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::{
        batching::{OPENING_SECTION_POINT, OPENING_SECTION_TARGET},
        bus::{
            NativeFoldedClaimBus, NativeFoldedClaimMessage, NativeOpeningClaimBus,
            NativeOpeningClaimMessage, NativeTwinScalarBus, NativeTwinScalarMessage,
        },
        twin::{CLAIM_SECTION_ALPHA, TWIN_SCALAR_NU_0},
    },
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeInitialOpeningPointCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeInitialOpeningPointCols<u8>)]
pub struct NativeInitialOpeningPointAir {
    pub folded_bus: NativeFoldedClaimBus,
    pub opening_bus: NativeOpeningClaimBus,
    /// One ordinary consumer plus one fan-out consumer per padded claim.
    pub copies: usize,
}

impl BaseAirWithPublicValues<F> for NativeInitialOpeningPointAir {}
impl PartitionedBaseAir<F> for NativeInitialOpeningPointAir {}
impl<F> BaseAir<F> for NativeInitialOpeningPointAir {
    fn width(&self) -> usize {
        NativeInitialOpeningPointCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeInitialOpeningPointAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native initial opening point row");
        let local: &NativeInitialOpeningPointCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.folded_bus.receive(
            builder,
            NativeFoldedClaimMessage {
                proof_idx: local.proof_idx.into(),
                section: AB::Expr::from_usize(CLAIM_SECTION_ALPHA),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active,
        );
        self.opening_bus.send(
            builder,
            NativeOpeningClaimMessage {
                proof_idx: local.proof_idx.into(),
                claim: AB::Expr::ZERO,
                section: AB::Expr::from_usize(OPENING_SECTION_POINT),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active * AB::Expr::from_usize(self.copies),
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeInitialOpeningTargetCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub value: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeInitialOpeningTargetCols<u8>)]
pub struct NativeInitialOpeningTargetAir {
    pub twin_bus: NativeTwinScalarBus,
    pub opening_bus: NativeOpeningClaimBus,
    /// One ordinary consumer plus one fan-out consumer per padded claim.
    pub copies: usize,
}

impl BaseAirWithPublicValues<F> for NativeInitialOpeningTargetAir {}
impl PartitionedBaseAir<F> for NativeInitialOpeningTargetAir {}
impl<F> BaseAir<F> for NativeInitialOpeningTargetAir {
    fn width(&self) -> usize {
        NativeInitialOpeningTargetCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeInitialOpeningTargetAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("native initial opening target row");
        let local: &NativeInitialOpeningTargetCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.twin_bus.receive(
            builder,
            NativeTwinScalarMessage {
                proof_idx: local.proof_idx.into(),
                kind: AB::Expr::from_usize(TWIN_SCALAR_NU_0),
                value: local.value.map(Into::into),
            },
            local.active,
        );
        self.opening_bus.send(
            builder,
            NativeOpeningClaimMessage {
                proof_idx: local.proof_idx.into(),
                claim: AB::Expr::ZERO,
                section: AB::Expr::from_usize(OPENING_SECTION_TARGET),
                coordinate: AB::Expr::ZERO,
                value: local.value.map(Into::into),
            },
            local.active * AB::Expr::from_usize(self.copies),
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeOpeningPaddingCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub claim: T,
    pub section: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

/// Copies the authenticated output opening (claim zero) into unused selector
/// slots of Construction 8.2's power-of-two batching domain.
///
/// The source claim is received with the same section and coordinate as the
/// destination, so padding cannot introduce a different point or target.
#[derive(ColumnsAir)]
#[columns_via(NativeOpeningPaddingCols<u8>)]
pub struct NativeOpeningPaddingAir {
    pub opening_bus: NativeOpeningClaimBus,
}

impl BaseAirWithPublicValues<F> for NativeOpeningPaddingAir {}
impl PartitionedBaseAir<F> for NativeOpeningPaddingAir {}
impl<F> BaseAir<F> for NativeOpeningPaddingAir {
    fn width(&self) -> usize {
        NativeOpeningPaddingCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeOpeningPaddingAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native opening padding row");
        let local: &NativeOpeningPaddingCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.assert_bool(local.section);
        self.opening_bus.receive(
            builder,
            NativeOpeningClaimMessage {
                proof_idx: local.proof_idx.into(),
                claim: AB::Expr::ZERO,
                section: local.section.into(),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active,
        );
        self.opening_bus.send(
            builder,
            NativeOpeningClaimMessage {
                proof_idx: local.proof_idx.into(),
                claim: local.claim.into(),
                section: local.section.into(),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeOodPointCols<T> {
    pub active: T,
    pub ood: T,
    pub coordinate: T,
    pub proof_idx: T,
    pub tidx: T,
    pub value: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeOodPointCols<u8>)]
pub struct NativeOodPointAir {
    pub transcript_bus: TranscriptBus,
    pub opening_bus: NativeOpeningClaimBus,
}

impl BaseAirWithPublicValues<F> for NativeOodPointAir {}
impl PartitionedBaseAir<F> for NativeOodPointAir {}
impl<F> BaseAir<F> for NativeOodPointAir {
    fn width(&self) -> usize {
        NativeOodPointCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeOodPointAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native OOD point row");
        let local: &NativeOodPointCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.tidx,
            local.value,
            local.active,
        );
        self.opening_bus.send(
            builder,
            NativeOpeningClaimMessage {
                proof_idx: local.proof_idx.into(),
                claim: local.ood + AB::F::ONE,
                section: AB::Expr::from_usize(OPENING_SECTION_POINT),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeOodTargetCols<T> {
    pub active: T,
    pub ood: T,
    pub proof_idx: T,
    pub tidx: T,
    pub value: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeOodTargetCols<u8>)]
pub struct NativeOodTargetAir {
    pub transcript_bus: TranscriptBus,
    pub opening_bus: NativeOpeningClaimBus,
}

impl BaseAirWithPublicValues<F> for NativeOodTargetAir {}
impl PartitionedBaseAir<F> for NativeOodTargetAir {}
impl<F> BaseAir<F> for NativeOodTargetAir {
    fn width(&self) -> usize {
        NativeOodTargetCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeOodTargetAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native OOD target row");
        let local: &NativeOodTargetCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx,
            local.value,
            local.active,
        );
        self.opening_bus.send(
            builder,
            NativeOpeningClaimMessage {
                proof_idx: local.proof_idx.into(),
                claim: local.ood + AB::F::ONE,
                section: AB::Expr::from_usize(OPENING_SECTION_TARGET),
                coordinate: AB::Expr::ZERO,
                value: local.value.map(Into::into),
            },
            local.active,
        );
    }
}

pub fn generate_native_initial_opening_point_trace(
    proof_idx: usize,
    values: &[EF],
) -> Option<RowMajorMatrix<F>> {
    if values.is_empty() {
        return None;
    }
    let height = values.len().next_power_of_two();
    let width = NativeInitialOpeningPointCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    for (coordinate, value) in values.iter().enumerate() {
        let cols: &mut NativeInitialOpeningPointCols<F> =
            trace[coordinate * width..(coordinate + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_usize(proof_idx);
        cols.coordinate = F::from_usize(coordinate);
        cols.value
            .copy_from_slice(value.as_basis_coefficients_slice());
    }
    Some(RowMajorMatrix::new(trace, width))
}

pub fn generate_native_initial_opening_target_trace(
    proof_idx: usize,
    value: EF,
) -> RowMajorMatrix<F> {
    let width = NativeInitialOpeningTargetCols::<F>::width();
    let mut trace = vec![F::ZERO; width];
    let cols: &mut NativeInitialOpeningTargetCols<F> = trace.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.proof_idx = F::from_usize(proof_idx);
    cols.value
        .copy_from_slice(value.as_basis_coefficients_slice());
    RowMajorMatrix::new(trace, width)
}

pub fn generate_native_opening_padding_trace(
    proof_idx: usize,
    first_padding_claim: usize,
    claim_count: usize,
    point: &[EF],
    target: EF,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if first_padding_claim == 0 || first_padding_claim > claim_count || point.is_empty() {
        return None;
    }
    let padding_count = claim_count - first_padding_claim;
    let rows_per_claim = point.len() + 1;
    let valid_rows = padding_count.checked_mul(rows_per_claim)?;
    let height = required_height.unwrap_or_else(|| valid_rows.max(1).next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeOpeningPaddingCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mut row_index = 0;
    for claim in first_padding_claim..claim_count {
        for (coordinate, &value) in point.iter().enumerate() {
            let cols: &mut NativeOpeningPaddingCols<F> =
                trace[row_index * width..(row_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.proof_idx = F::from_usize(proof_idx);
            cols.claim = F::from_usize(claim);
            cols.section = F::from_usize(OPENING_SECTION_POINT);
            cols.coordinate = F::from_usize(coordinate);
            cols.value
                .copy_from_slice(value.as_basis_coefficients_slice());
            row_index += 1;
        }
        let cols: &mut NativeOpeningPaddingCols<F> =
            trace[row_index * width..(row_index + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_usize(proof_idx);
        cols.claim = F::from_usize(claim);
        cols.section = F::from_usize(OPENING_SECTION_TARGET);
        cols.value
            .copy_from_slice(target.as_basis_coefficients_slice());
        row_index += 1;
    }
    Some(RowMajorMatrix::new(trace, width))
}

pub fn generate_native_ood_point_trace(
    proof_idx: usize,
    points: &[Vec<EF>],
    tidx: &[usize],
) -> Option<RowMajorMatrix<F>> {
    if points.is_empty() {
        if !tidx.is_empty() {
            return None;
        }
        let width = NativeOodPointCols::<F>::width();
        return Some(RowMajorMatrix::new(vec![F::ZERO; width], width));
    }
    let dimension = points.first()?.len();
    if points.len() != tidx.len() || points.iter().any(|point| point.len() != dimension) {
        return None;
    }
    let valid_rows = points.len() * dimension;
    let height = valid_rows.next_power_of_two();
    let width = NativeOodPointCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    for (ood, point) in points.iter().enumerate() {
        for (coordinate, value) in point.iter().enumerate() {
            let row_index = ood * dimension + coordinate;
            let cols: &mut NativeOodPointCols<F> =
                trace[row_index * width..(row_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.ood = F::from_usize(ood);
            cols.coordinate = F::from_usize(coordinate);
            cols.proof_idx = F::from_usize(proof_idx);
            cols.tidx = F::from_usize(tidx[ood] + coordinate * D_EF);
            cols.value
                .copy_from_slice(value.as_basis_coefficients_slice());
        }
    }
    Some(RowMajorMatrix::new(trace, width))
}

pub fn generate_native_ood_target_trace(
    proof_idx: usize,
    values: &[EF],
    tidx: &[usize],
) -> Option<RowMajorMatrix<F>> {
    if values.is_empty() {
        if !tidx.is_empty() {
            return None;
        }
        let width = NativeOodTargetCols::<F>::width();
        return Some(RowMajorMatrix::new(vec![F::ZERO; width], width));
    }
    if values.len() != tidx.len() {
        return None;
    }
    let height = values.len().next_power_of_two();
    let width = NativeOodTargetCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    for (ood, value) in values.iter().enumerate() {
        let cols: &mut NativeOodTargetCols<F> = trace[ood * width..(ood + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.ood = F::from_usize(ood);
        cols.proof_idx = F::from_usize(proof_idx);
        cols.tidx = F::from_usize(tidx[ood]);
        cols.value
            .copy_from_slice(value.as_basis_coefficients_slice());
    }
    Some(RowMajorMatrix::new(trace, width))
}

/// One out-of-domain claim row: either a sampled point coordinate or the
/// observed answer at that point.
///
/// The two were separate AIRs differing only in which transcript operation they
/// perform and which opening section they name. Every AIR costs the prover a
/// fixed amount regardless of how little it holds -- measured at ~1.58 ms
/// against the recursive lane's own layers -- and these two hold nine and eight
/// cells. Merging them is a straight saving, and it is what brings a bounded
/// history span at arity four under the child-AIR cap it currently misses by
/// one.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeOodClaimCols<T> {
    pub active: T,
    /// One on an answer row, zero on a point row.
    pub is_target: T,
    pub ood: T,
    pub coordinate: T,
    pub proof_idx: T,
    pub tidx: T,
    pub value: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeOodClaimCols<u8>)]
pub struct NativeOodClaimAir {
    pub transcript_bus: TranscriptBus,
    pub opening_bus: NativeOpeningClaimBus,
}

impl BaseAirWithPublicValues<F> for NativeOodClaimAir {}
impl PartitionedBaseAir<F> for NativeOodClaimAir {}
impl<F> BaseAir<F> for NativeOodClaimAir {
    fn width(&self) -> usize {
        NativeOodClaimCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeOodClaimAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native OOD claim row");
        let local: &NativeOodClaimCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.assert_bool(local.is_target);
        // An answer is a single value, not a coordinate of a point.
        builder.when(local.is_target).assert_zero(local.coordinate);

        // A point is sampled from the transcript; an answer is observed into
        // it. The two multiplicities are complementary, so exactly one happens
        // per active row.
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.tidx,
            local.value,
            local.active * (AB::Expr::ONE - local.is_target),
        );
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx,
            local.value,
            local.active * local.is_target,
        );
        // `is_target` is boolean, so the section can only take the two values
        // the separate AIRs used as constants.
        let section = AB::Expr::from_usize(OPENING_SECTION_POINT)
            + local.is_target
                * (AB::Expr::from_usize(OPENING_SECTION_TARGET)
                    - AB::Expr::from_usize(OPENING_SECTION_POINT));
        self.opening_bus.send(
            builder,
            NativeOpeningClaimMessage {
                proof_idx: local.proof_idx.into(),
                claim: local.ood + AB::F::ONE,
                section,
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active,
        );
    }
}

/// Point rows first, then answer rows, in one table.
pub fn generate_native_ood_claim_trace(
    proof_idx: usize,
    points: &[Vec<EF>],
    point_tidx: &[usize],
    answers: &[EF],
    answer_tidx: &[usize],
) -> Option<RowMajorMatrix<F>> {
    let width = NativeOodClaimCols::<F>::width();
    if points.is_empty() {
        if !point_tidx.is_empty() || !answers.is_empty() || !answer_tidx.is_empty() {
            return None;
        }
        return Some(RowMajorMatrix::new(vec![F::ZERO; width], width));
    }
    let dimension = points.first()?.len();
    if points.len() != point_tidx.len()
        || answers.len() != answer_tidx.len()
        || points.iter().any(|point| point.len() != dimension)
    {
        return None;
    }
    let point_rows = points.len().checked_mul(dimension)?;
    let valid_rows = point_rows.checked_add(answers.len())?;
    let height = valid_rows.next_power_of_two();
    let mut trace = vec![F::ZERO; height * width];
    for (ood, point) in points.iter().enumerate() {
        for (coordinate, value) in point.iter().enumerate() {
            let row_index = ood * dimension + coordinate;
            let cols: &mut NativeOodClaimCols<F> =
                trace[row_index * width..(row_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.ood = F::from_usize(ood);
            cols.coordinate = F::from_usize(coordinate);
            cols.proof_idx = F::from_usize(proof_idx);
            cols.tidx = F::from_usize(point_tidx[ood] + coordinate * D_EF);
            cols.value
                .copy_from_slice(value.as_basis_coefficients_slice());
        }
    }
    for (ood, value) in answers.iter().enumerate() {
        let row_index = point_rows + ood;
        let cols: &mut NativeOodClaimCols<F> =
            trace[row_index * width..(row_index + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.is_target = F::ONE;
        cols.ood = F::from_usize(ood);
        cols.proof_idx = F::from_usize(proof_idx);
        cols.tidx = F::from_usize(answer_tidx[ood]);
        cols.value
            .copy_from_slice(value.as_basis_coefficients_slice());
    }
    Some(RowMajorMatrix::new(trace, width))
}
