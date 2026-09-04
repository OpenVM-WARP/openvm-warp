use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, F};
use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::decompose_usize;
use crate::proof_shape::{
    bus::{ProofShapeMetadataBus, ProofShapeMetadataMessage, PROOF_SHAPE_METADATA_NUM_LIMBS},
    AirMetadata,
};

/// Child-VK metadata carried in the dynamic tail of a large-key
/// [`ProofShapeAir`](super::ProofShapeAir) row.
///
/// These are witness columns, but the complete tuple is authenticated against
/// [`ProofShapeMetadataAir`]'s preprocessed table on one permutation bus.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Debug)]
pub struct ProofShapeMetadataCols<T> {
    pub air_idx: T,
    pub is_required: T,
    pub need_rot: T,
    pub num_public_values: T,
    pub has_public_values: T,
    pub num_interactions: T,
    pub num_interactions_limbs: [T; PROOF_SHAPE_METADATA_NUM_LIMBS],
    pub main_width: T,
    pub is_min_cached: T,
    pub has_preprocessed: T,
    pub preprocessed_log_height: T,
    pub preprocessed_width: T,
    pub preprocessed_commit: [T; DIGEST_SIZE],
}

impl<T: Clone> ProofShapeMetadataCols<T> {
    pub fn message(&self) -> ProofShapeMetadataMessage<T> {
        ProofShapeMetadataMessage {
            air_idx: self.air_idx.clone(),
            is_required: self.is_required.clone(),
            need_rot: self.need_rot.clone(),
            num_public_values: self.num_public_values.clone(),
            has_public_values: self.has_public_values.clone(),
            num_interactions: self.num_interactions.clone(),
            num_interactions_limbs: self.num_interactions_limbs.clone(),
            main_width: self.main_width.clone(),
            is_min_cached: self.is_min_cached.clone(),
            has_preprocessed: self.has_preprocessed.clone(),
            preprocessed_log_height: self.preprocessed_log_height.clone(),
            preprocessed_width: self.preprocessed_width.clone(),
            preprocessed_commit: self.preprocessed_commit.clone(),
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
struct ProofShapeMetadataPrepCols<T> {
    active: T,
    metadata: ProofShapeMetadataCols<T>,
}

/// VK-committed metadata table for the large-child-key proof-shape relation.
///
/// The common main is a single constrained-zero column because MultiSTARK
/// AIRs require a non-empty main trace.  All meaningful values live in the
/// preprocessed trace and are therefore committed by the enclosing verifier
/// circuit's VK.
pub struct ProofShapeMetadataAir {
    pub per_air: Vec<AirMetadata>,
    pub l_skip: usize,
    pub min_cached_idx: usize,
    pub bus: ProofShapeMetadataBus,
}

impl BaseAirWithPublicValues<F> for ProofShapeMetadataAir {}
impl PartitionedBaseAir<F> for ProofShapeMetadataAir {}

impl BaseAir<F> for ProofShapeMetadataAir {
    fn width(&self) -> usize {
        1
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let height = self.per_air.len().max(1).next_power_of_two();
        let width = ProofShapeMetadataPrepCols::<F>::width();
        let mut values = F::zero_vec(height * width);
        for (air_idx, metadata) in self.per_air.iter().enumerate() {
            let row: &mut ProofShapeMetadataPrepCols<F> =
                values[air_idx * width..(air_idx + 1) * width].borrow_mut();
            row.active = F::ONE;
            fill_metadata_cols(
                &mut row.metadata,
                air_idx,
                metadata,
                self.l_skip,
                self.min_cached_idx,
            )?;
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl<AB> Air<AB> for ProofShapeMetadataAir
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::F: PrimeField32,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let main_row = main
            .row_slice(0)
            .expect("proof-shape metadata dummy main row");
        builder.assert_zero(main_row[0]);

        let prep = builder.preprocessed();
        let row = prep
            .row_slice(0)
            .expect("proof-shape metadata preprocessed row");
        let next_row = prep
            .row_slice(1)
            .expect("proof-shape metadata next preprocessed row");
        let local: &ProofShapeMetadataPrepCols<AB::Var> = (*row).borrow();
        let next: &ProofShapeMetadataPrepCols<AB::Var> = (*next_row).borrow();

        for flag in [
            local.active,
            local.metadata.is_required,
            local.metadata.need_rot,
            local.metadata.has_public_values,
            local.metadata.is_min_cached,
            local.metadata.has_preprocessed,
        ] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_zero(local.metadata.air_idx);
        let mut transition = builder.when_transition();
        transition.assert_bool(local.active - next.active);
        transition
            .when(next.active)
            .assert_eq(next.metadata.air_idx, local.metadata.air_idx + AB::F::ONE);
        builder.when_last_row().when(local.active).assert_eq(
            local.metadata.air_idx,
            AB::Expr::from_usize(self.per_air.len() - 1),
        );

        let recomposed = local
            .metadata
            .num_interactions_limbs
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (index, limb)| {
                acc + AB::Expr::from_u32(1 << (8 * index)) * *limb
            });
        builder
            .when(local.active)
            .assert_eq(recomposed, local.metadata.num_interactions);
        let no_preprocessed = AB::Expr::from(local.active)
            * (AB::Expr::ONE - AB::Expr::from(local.metadata.has_preprocessed));
        builder
            .when(no_preprocessed.clone())
            .assert_zero(local.metadata.preprocessed_log_height);
        builder
            .when(no_preprocessed.clone())
            .assert_zero(local.metadata.preprocessed_width);
        for limb in local.metadata.preprocessed_commit {
            builder.when(no_preprocessed.clone()).assert_zero(limb);
        }

        self.bus
            .send(builder, local.metadata.message(), local.active);
    }
}

pub fn fill_metadata_cols(
    cols: &mut ProofShapeMetadataCols<F>,
    air_idx: usize,
    metadata: &AirMetadata,
    l_skip: usize,
    min_cached_idx: usize,
) -> Option<()> {
    cols.air_idx = F::from_usize(air_idx);
    cols.is_required = F::from_bool(metadata.is_required);
    cols.need_rot = F::from_bool(metadata.need_rot);
    cols.num_public_values = F::from_usize(metadata.num_public_values);
    cols.has_public_values = F::from_bool(metadata.num_public_values != 0);
    cols.num_interactions = F::from_usize(metadata.num_interactions);
    cols.num_interactions_limbs =
        decompose_usize::<PROOF_SHAPE_METADATA_NUM_LIMBS, 8>(metadata.num_interactions)
            .map(F::from_usize);
    cols.main_width = F::from_usize(metadata.main_width);
    cols.is_min_cached = F::from_bool(air_idx == min_cached_idx);
    if let Some(preprocessed) = &metadata.preprocessed_data {
        cols.has_preprocessed = F::ONE;
        cols.preprocessed_log_height =
            F::from_usize(l_skip.checked_add_signed(preprocessed.hypercube_dim)?);
        cols.preprocessed_width = F::from_usize(metadata.preprocessed_width?);
        cols.preprocessed_commit = preprocessed.commit;
    }
    Some(())
}

#[must_use]
pub fn generate_metadata_dummy_trace(height: usize) -> RowMajorMatrix<F> {
    RowMajorMatrix::new(F::zero_vec(height), 1)
}

#[cfg(test)]
mod tests {
    use openvm_stark_backend::keygen::types::VerifierSinglePreprocessedData;
    use p3_matrix::Matrix;

    use super::*;

    #[test]
    fn metadata_table_is_dense_and_padded_with_zeroes() {
        let per_air = vec![
            AirMetadata {
                is_required: true,
                need_rot: false,
                num_public_values: 3,
                num_interactions: 257,
                main_width: 11,
                cached_widths: Vec::new(),
                preprocessed_width: None,
                preprocessed_data: None,
            },
            AirMetadata {
                is_required: false,
                need_rot: true,
                num_public_values: 0,
                num_interactions: 9,
                main_width: 7,
                cached_widths: Vec::new(),
                preprocessed_width: Some(5),
                preprocessed_data: Some(VerifierSinglePreprocessedData {
                    commit: [F::ONE; DIGEST_SIZE],
                    hypercube_dim: 2,
                    stacking_width: 5,
                }),
            },
            AirMetadata {
                is_required: false,
                need_rot: false,
                num_public_values: 0,
                num_interactions: 0,
                main_width: 1,
                cached_widths: Vec::new(),
                preprocessed_width: None,
                preprocessed_data: None,
            },
        ];
        let air = ProofShapeMetadataAir {
            per_air,
            l_skip: 4,
            min_cached_idx: 0,
            bus: ProofShapeMetadataBus::new(0),
        };
        let trace = air.preprocessed_trace().unwrap();
        assert_eq!(trace.height(), 4);
        let width = ProofShapeMetadataPrepCols::<F>::width();
        let first: &ProofShapeMetadataPrepCols<F> = trace.values[..width].borrow();
        assert_eq!(first.active, F::ONE);
        assert_eq!(first.metadata.num_interactions, F::from_usize(257));
        assert_eq!(
            first.metadata.num_interactions_limbs,
            [F::ONE, F::ONE, F::ZERO, F::ZERO]
        );
        let second: &ProofShapeMetadataPrepCols<F> = trace.values[width..2 * width].borrow();
        assert_eq!(second.metadata.preprocessed_log_height, F::from_usize(6));
        let padding: &ProofShapeMetadataPrepCols<F> = trace.values[3 * width..4 * width].borrow();
        assert_eq!(padding.active, F::ZERO);
        assert!(padding
            .metadata
            .preprocessed_commit
            .iter()
            .all(|x| *x == F::ZERO));
    }
}
