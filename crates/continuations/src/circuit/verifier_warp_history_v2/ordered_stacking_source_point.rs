//! Circuit-constrained source opening-point handoff for ordered stacking.
//!
//! The genuine SWIRL verifier owns [`FixedSetupOpeningPointBusV2`]. The fixed
//! setup-opening authority already looks up that bus with profile-derived
//! multiplicities. This adapter adds exactly one lookup for every original
//! source-point coordinate and republishes it on the recursion-owned
//! [`OrderedStackingSourcePointBus`]. Ordinary `EqBaseAir` and
//! `SumcheckRoundsAir` then consume those messages directly in their stacking
//! algebra.
//!
//! The genuine point fanout must be configured with
//! [`OrderedStackingSourcePointProfileV2::merged_fixed_setup_point_demands`]:
//! this retains the setup-opening authority's existing lookup counts and adds
//! the adapter's one lookup per coordinate. No host equality is a security
//! boundary.

use core::borrow::{Borrow, BorrowMut};
use std::{collections::BTreeMap, sync::Arc};

use openvm_cpu_backend::CpuBackend;
use openvm_recursion_circuit::stacking::{
    OrderedStackingProfile, OrderedStackingSourcePointBus, OrderedStackingSourcePointMessage,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::{BasedVectorSpace, PrimeCharacteristicRing},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::AirProvingContext,
    BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};

use super::{FixedSetupOpeningPointBusV2, FixedSetupOpeningPointMessageV2};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OrderedStackingSourcePointPlanV2 {
    source_proof_index: u32,
    transition_index: usize,
    coordinate_index: usize,
}

/// Setup-fixed routing from genuine SWIRL proof indices to ordered stacking
/// transitions. Transition indices are positions in `source_proof_indices`.
#[derive(Clone, Debug)]
pub struct OrderedStackingSourcePointProfileV2 {
    plans: Arc<[OrderedStackingSourcePointPlanV2]>,
    source_proof_indices: Arc<[u32]>,
    point_dimensions: Arc<[usize]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderedStackingSourcePointErrorV2 {
    Empty,
    TransitionCount {
        stacking: usize,
        source: usize,
        dimensions: usize,
    },
    DuplicateSourceProof(u32),
    NonCanonicalSourceProof {
        transition: usize,
        source_proof_index: u32,
    },
    PointDimension {
        transition: usize,
        actual: usize,
        minimum: usize,
        maximum: usize,
    },
    InvalidFixedDemand,
    FixedDemandOverflow {
        source_proof_index: u32,
        coordinate_index: u32,
    },
    RecordCount {
        actual: usize,
        expected: usize,
    },
    RecordSourceProof {
        transition: usize,
        actual: u32,
        expected: u32,
    },
    RecordPointDimension {
        transition: usize,
        actual: usize,
        expected: usize,
    },
}

impl core::fmt::Display for OrderedStackingSourcePointErrorV2 {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "invalid ordered stacking source-point handoff: {self:?}"
        )
    }
}

impl std::error::Error for OrderedStackingSourcePointErrorV2 {}

impl OrderedStackingSourcePointProfileV2 {
    /// Create setup-fixed source routing for every ordered reduction.
    ///
    /// `source_proof_indices[t]` identifies the corresponding genuine SWIRL
    /// opening point on `FixedSetupOpeningPointBusV2`; `point_dimensions[t]`
    /// is its exact original dimension. The transition key on the recursion
    /// bus is always `t`, matching the ordinary ordered stacking traces.
    pub fn new(
        stacking_profile: &OrderedStackingProfile,
        source_proof_indices: Vec<u32>,
        point_dimensions: Vec<usize>,
    ) -> Result<Self, OrderedStackingSourcePointErrorV2> {
        let transition_count = stacking_profile.reduction_count();
        if transition_count == 0 {
            return Err(OrderedStackingSourcePointErrorV2::Empty);
        }
        if source_proof_indices.len() != transition_count
            || point_dimensions.len() != transition_count
        {
            return Err(OrderedStackingSourcePointErrorV2::TransitionCount {
                stacking: transition_count,
                source: source_proof_indices.len(),
                dimensions: point_dimensions.len(),
            });
        }
        let mut seen = BTreeMap::new();
        let mut plans = Vec::new();
        for (transition, (&source_proof_index, &point_dimension)) in source_proof_indices
            .iter()
            .zip(&point_dimensions)
            .enumerate()
        {
            // Native V3 statements are contiguous and identity-indexed.  Accepting a different
            // VK-owned permutation here is unnecessary flexibility and could bind transition t's
            // stacking proof to another source transcript point without that mapping appearing in
            // the authority batch digest.
            if source_proof_index != transition as u32 {
                return Err(OrderedStackingSourcePointErrorV2::NonCanonicalSourceProof {
                    transition,
                    source_proof_index,
                });
            }
            if seen.insert(source_proof_index, ()).is_some() {
                return Err(OrderedStackingSourcePointErrorV2::DuplicateSourceProof(
                    source_proof_index,
                ));
            }
            let (minimum, maximum) = stacking_profile
                .source_point_dimension_bounds(transition)
                .ok_or(OrderedStackingSourcePointErrorV2::Empty)?;
            if !(minimum..=maximum).contains(&point_dimension) {
                return Err(OrderedStackingSourcePointErrorV2::PointDimension {
                    transition,
                    actual: point_dimension,
                    minimum,
                    maximum,
                });
            }
            plans.extend((0..point_dimension).map(|coordinate_index| {
                OrderedStackingSourcePointPlanV2 {
                    source_proof_index,
                    transition_index: transition,
                    coordinate_index,
                }
            }));
        }
        Ok(Self {
            plans: plans.into(),
            source_proof_indices: source_proof_indices.into(),
            point_dimensions: point_dimensions.into(),
        })
    }

    #[must_use]
    pub fn transition_count(&self) -> usize {
        self.source_proof_indices.len()
    }

    #[must_use]
    pub fn valid_row_count(&self) -> usize {
        self.plans.len()
    }

    /// Merge the fixed setup-opening authority's existing lookup counts with
    /// this adapter's one additional lookup per original source coordinate.
    /// The returned table is sorted by `(proof_index, point_index)` and is the
    /// exact table to pass to the genuine SWIRL point fanout attachment.
    pub fn merged_fixed_setup_point_demands(
        &self,
        authority_demands: &[(u32, u32, u32)],
    ) -> Result<Vec<(u32, u32, u32)>, OrderedStackingSourcePointErrorV2> {
        let mut merged = BTreeMap::<(u32, u32), u32>::new();
        for &(proof_index, point_index, count) in authority_demands {
            if count == 0 || merged.insert((proof_index, point_index), count).is_some() {
                return Err(OrderedStackingSourcePointErrorV2::InvalidFixedDemand);
            }
        }
        for plan in self.plans.iter() {
            let point_index = u32::try_from(plan.coordinate_index)
                .map_err(|_| OrderedStackingSourcePointErrorV2::InvalidFixedDemand)?;
            let count = merged
                .entry((plan.source_proof_index, point_index))
                .or_default();
            *count = count.checked_add(1).ok_or(
                OrderedStackingSourcePointErrorV2::FixedDemandOverflow {
                    source_proof_index: plan.source_proof_index,
                    coordinate_index: point_index,
                },
            )?;
        }
        Ok(merged
            .into_iter()
            .map(|((proof_index, point_index), count)| (proof_index, point_index, count))
            .collect())
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
struct OrderedStackingSourcePointPrepColsV2<T> {
    active: T,
    source_proof_index: T,
    transition_index: T,
    coordinate_index: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
struct OrderedStackingSourcePointColsV2<T> {
    value: [T; D_EF],
}

/// Authority-owned adapter from the genuine fixed setup source point to the
/// generic recursion ordered-stacking point bus.
#[derive(Clone, Debug)]
pub struct OrderedStackingSourcePointAdapterAirV2 {
    pub profile: OrderedStackingSourcePointProfileV2,
    pub fixed_source_bus: FixedSetupOpeningPointBusV2,
    pub ordered_stacking_bus: OrderedStackingSourcePointBus,
}

impl BaseAir<F> for OrderedStackingSourcePointAdapterAirV2 {
    fn width(&self) -> usize {
        core::mem::size_of::<OrderedStackingSourcePointColsV2<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = core::mem::size_of::<OrderedStackingSourcePointPrepColsV2<u8>>();
        let height = self.profile.plans.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(height * width);
        for (row_idx, plan) in self.profile.plans.iter().enumerate() {
            let cols: &mut OrderedStackingSourcePointPrepColsV2<F> =
                values[row_idx * width..(row_idx + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.source_proof_index = F::from_u32(plan.source_proof_index);
            cols.transition_index = F::from_usize(plan.transition_index);
            cols.coordinate_index = F::from_usize(plan.coordinate_index);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for OrderedStackingSourcePointAdapterAirV2 {}
impl PartitionedBaseAir<F> for OrderedStackingSourcePointAdapterAirV2 {}

impl<AB> Air<AB> for OrderedStackingSourcePointAdapterAirV2
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let prep = builder.preprocessed();
        let prep_row = prep
            .row_slice(0)
            .expect("ordered stacking source-point prep row");
        let prep: &OrderedStackingSourcePointPrepColsV2<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("ordered stacking source-point row");
        let local: &OrderedStackingSourcePointColsV2<AB::Var> = (*row).borrow();

        builder.assert_bool(prep.active);
        let active = AB::Expr::from(prep.active);
        for value in local.value {
            builder
                .when(AB::Expr::ONE - active.clone())
                .assert_zero(value);
        }

        self.fixed_source_bus.lookup_key(
            builder,
            FixedSetupOpeningPointMessageV2 {
                proof_index: prep.source_proof_index.into(),
                point_index: prep.coordinate_index.into(),
                value: local.value.map(Into::into),
            },
            active.clone(),
        );
        self.ordered_stacking_bus.send(
            builder,
            prep.transition_index,
            OrderedStackingSourcePointMessage {
                coordinate_idx: prep.coordinate_index.into(),
                value: local.value.map(Into::into),
            },
            active,
        );
    }
}

/// One genuine SWIRL source point in adapter-profile transition order.
#[derive(Clone, Copy, Debug)]
pub struct OrderedStackingSourcePointRecordV2<'a> {
    pub source_proof_index: u32,
    pub opening_point: &'a [EF],
}

impl OrderedStackingSourcePointAdapterAirV2 {
    pub fn generate_ctx<SC: StarkProtocolConfig<F = F>>(
        &self,
        records: &[OrderedStackingSourcePointRecordV2<'_>],
    ) -> Result<AirProvingContext<CpuBackend<SC>>, OrderedStackingSourcePointErrorV2> {
        if records.len() != self.profile.transition_count() {
            return Err(OrderedStackingSourcePointErrorV2::RecordCount {
                actual: records.len(),
                expected: self.profile.transition_count(),
            });
        }
        for (transition, record) in records.iter().enumerate() {
            let expected_source = self.profile.source_proof_indices[transition];
            if record.source_proof_index != expected_source {
                return Err(OrderedStackingSourcePointErrorV2::RecordSourceProof {
                    transition,
                    actual: record.source_proof_index,
                    expected: expected_source,
                });
            }
            let expected_dimension = self.profile.point_dimensions[transition];
            if record.opening_point.len() != expected_dimension {
                return Err(OrderedStackingSourcePointErrorV2::RecordPointDimension {
                    transition,
                    actual: record.opening_point.len(),
                    expected: expected_dimension,
                });
            }
        }

        let width = core::mem::size_of::<OrderedStackingSourcePointColsV2<u8>>();
        let height = self.profile.plans.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(height * width);
        for (row_idx, plan) in self.profile.plans.iter().enumerate() {
            let cols: &mut OrderedStackingSourcePointColsV2<F> =
                values[row_idx * width..(row_idx + 1) * width].borrow_mut();
            cols.value.copy_from_slice(
                records[plan.transition_index].opening_point[plan.coordinate_index]
                    .as_basis_coefficients_slice(),
            );
        }
        Ok(AirProvingContext::simple_no_pis(RowMajorMatrix::new(
            values, width,
        )))
    }
}
