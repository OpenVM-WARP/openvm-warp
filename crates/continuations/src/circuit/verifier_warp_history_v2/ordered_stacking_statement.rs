//! Constrained handoff from ordered stacking to a multi-constraint WHIR statement.
//!
//! [`StackingModule`](openvm_recursion_circuit::stacking::StackingModule) proves the ordinary
//! stacking reduction and publishes two isolated permutation-bus streams. This AIR receives each
//! stream exactly once and republishes the same values on caller-owned multi-constraint statement
//! lookup buses. Point lookup fanout is fixed in the preprocessed trace because the
//! final-polynomial suffix consumes coordinate `i` `2^(i - num_sumcheck_rounds)` times.
//!
//! For source reduction `t` and the single terminal WHIR proof `p`, the exact messages are:
//!
//! - source point: `(proof_idx=t, WhirOpeningPointMessage { idx=i, value=u_t[i] })`;
//! - WHIR point: `(proof_idx=p, MultiConstraintPointMessage { constraint_idx=t, coordinate_idx=i,
//!   value=u_t[i] })` with profile-fixed multiplicity;
//! - source opening: `(proof_idx=t, OrderedStackingOpeningMessage { opening_idx=j, value=y_t[j]
//!   })`;
//! - WHIR opening: `(proof_idx=p, MultiConstraintOpeningMessage { constraint_idx=t, opening_idx=j,
//!   value=y_t[j] })` exactly once.
//!
//! The caller-prefix transcript message is intentionally not produced here. It is sampled after
//! all statement commitments and remains owned by the terminal multi-WHIR authority assembly.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_cpu_backend::CpuBackend;
use openvm_recursion_circuit::{
    bus::{WhirOpeningPointBus, WhirOpeningPointMessage},
    stacking::{OrderedStackingOpeningBus, OrderedStackingOpeningMessage, OrderedStackingProfile},
    whir::multi_constraint::{
        air::{
            MultiConstraintOpeningMessage, MultiConstraintPointMessage,
            MultiConstraintStatementBuses,
        },
        MultiConstraintWhirProfile,
    },
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct OrderedStackingStatementPlan {
    source_reduction_idx: usize,
    statement_proof_idx: usize,
    constraint_idx: usize,
    item_idx: usize,
    point_lookup_multiplicity: usize,
    is_point: bool,
}

/// VK-owned statement routing and exact lookup multiplicities.
#[derive(Clone, Debug)]
pub struct OrderedStackingMultiConstraintProfile {
    plans: Arc<[OrderedStackingStatementPlan]>,
    statement_count: usize,
    point_dimension: usize,
    commitment_widths: Arc<[Vec<usize>]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderedStackingMultiConstraintError {
    Empty,
    ConstraintCount {
        stacking: usize,
        whir: usize,
    },
    PointDimension {
        reduction: usize,
        stacking: usize,
        whir: usize,
    },
    CommitmentWidths {
        reduction: usize,
    },
    SumcheckRounds,
    PointMultiplicity {
        coordinate: usize,
    },
    TraceSizeOverflow,
    StatementCount {
        actual: usize,
        expected: usize,
    },
    StatementPointDimension {
        reduction: usize,
    },
    StatementCommitmentWidths {
        reduction: usize,
    },
}

impl core::fmt::Display for OrderedStackingMultiConstraintError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "invalid ordered stacking multi-constraint handoff: {self:?}"
        )
    }
}

impl std::error::Error for OrderedStackingMultiConstraintError {}

impl OrderedStackingMultiConstraintProfile {
    /// Map all ordered reductions to consecutive constraints in one terminal multi-WHIR proof.
    ///
    /// `num_whir_sumcheck_rounds` is key material and determines the exact point lookup fanout.
    /// Every reduction must have the same commitment widths required by `whir_profile`.
    pub fn one_whir_class(
        stacking_profile: &OrderedStackingProfile,
        whir_profile: &MultiConstraintWhirProfile,
        num_whir_sumcheck_rounds: usize,
        statement_proof_idx: usize,
    ) -> Result<Self, OrderedStackingMultiConstraintError> {
        let reduction_count = stacking_profile.reduction_count();
        if reduction_count == 0 {
            return Err(OrderedStackingMultiConstraintError::Empty);
        }
        if reduction_count != whir_profile.constraint_count {
            return Err(OrderedStackingMultiConstraintError::ConstraintCount {
                stacking: reduction_count,
                whir: whir_profile.constraint_count,
            });
        }
        if num_whir_sumcheck_rounds > whir_profile.point_dimension {
            return Err(OrderedStackingMultiConstraintError::SumcheckRounds);
        }

        let total_openings = whir_profile
            .total_width()
            .map_err(|_| OrderedStackingMultiConstraintError::TraceSizeOverflow)?;
        let rows_per_statement = whir_profile
            .point_dimension
            .checked_add(total_openings)
            .ok_or(OrderedStackingMultiConstraintError::TraceSizeOverflow)?;
        let mut plans = Vec::with_capacity(
            reduction_count
                .checked_mul(rows_per_statement)
                .ok_or(OrderedStackingMultiConstraintError::TraceSizeOverflow)?,
        );
        let mut commitment_widths = Vec::with_capacity(reduction_count);
        for reduction in 0..reduction_count {
            let shape = stacking_profile
                .statement_shape(reduction)
                .ok_or(OrderedStackingMultiConstraintError::Empty)?;
            if shape.point_dimension != whir_profile.point_dimension {
                return Err(OrderedStackingMultiConstraintError::PointDimension {
                    reduction,
                    stacking: shape.point_dimension,
                    whir: whir_profile.point_dimension,
                });
            }
            if shape.commitment_widths != whir_profile.commitment_widths {
                return Err(OrderedStackingMultiConstraintError::CommitmentWidths { reduction });
            }
            commitment_widths.push(shape.commitment_widths);
            for coordinate in 0..shape.point_dimension {
                let point_lookup_multiplicity = whir_profile
                    .point_lookup_multiplicity(coordinate, num_whir_sumcheck_rounds)
                    .map_err(|_| OrderedStackingMultiConstraintError::PointMultiplicity {
                        coordinate,
                    })?;
                plans.push(OrderedStackingStatementPlan {
                    source_reduction_idx: reduction,
                    statement_proof_idx,
                    constraint_idx: reduction,
                    item_idx: coordinate,
                    point_lookup_multiplicity,
                    is_point: true,
                });
            }
            for opening_idx in 0..total_openings {
                plans.push(OrderedStackingStatementPlan {
                    source_reduction_idx: reduction,
                    statement_proof_idx,
                    constraint_idx: reduction,
                    item_idx: opening_idx,
                    point_lookup_multiplicity: 0,
                    is_point: false,
                });
            }
        }
        Ok(Self {
            plans: plans.into(),
            statement_count: reduction_count,
            point_dimension: whir_profile.point_dimension,
            commitment_widths: commitment_widths.into(),
        })
    }

    #[must_use]
    pub fn statement_count(&self) -> usize {
        self.statement_count
    }

    #[must_use]
    pub fn valid_row_count(&self) -> usize {
        self.plans.len()
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
struct OrderedStackingStatementPrepCols<T> {
    active: T,
    is_point: T,
    source_reduction_idx: T,
    statement_proof_idx: T,
    constraint_idx: T,
    item_idx: T,
    point_lookup_multiplicity: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
struct OrderedStackingStatementCols<T> {
    value: [T; D_EF],
}

/// Authority-owned bus adapter between ordinary stacking and multi-WHIR.
#[derive(Clone, Debug)]
pub struct OrderedStackingMultiConstraintAir {
    pub profile: OrderedStackingMultiConstraintProfile,
    pub source_point_bus: WhirOpeningPointBus,
    pub source_opening_bus: OrderedStackingOpeningBus,
    pub statement_buses: MultiConstraintStatementBuses,
}

impl BaseAir<F> for OrderedStackingMultiConstraintAir {
    fn width(&self) -> usize {
        core::mem::size_of::<OrderedStackingStatementCols<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = core::mem::size_of::<OrderedStackingStatementPrepCols<u8>>();
        let height = self.profile.plans.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(height * width);
        for (row_idx, plan) in self.profile.plans.iter().enumerate() {
            let cols: &mut OrderedStackingStatementPrepCols<F> =
                values[row_idx * width..(row_idx + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.is_point = F::from_bool(plan.is_point);
            cols.source_reduction_idx = F::from_usize(plan.source_reduction_idx);
            cols.statement_proof_idx = F::from_usize(plan.statement_proof_idx);
            cols.constraint_idx = F::from_usize(plan.constraint_idx);
            cols.item_idx = F::from_usize(plan.item_idx);
            cols.point_lookup_multiplicity = F::from_usize(plan.point_lookup_multiplicity);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for OrderedStackingMultiConstraintAir {}
impl PartitionedBaseAir<F> for OrderedStackingMultiConstraintAir {}

impl<AB> Air<AB> for OrderedStackingMultiConstraintAir
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let prep = builder.preprocessed();
        let prep_row = prep
            .row_slice(0)
            .expect("ordered stacking statement prep row");
        let prep: &OrderedStackingStatementPrepCols<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("ordered stacking statement row");
        let local: &OrderedStackingStatementCols<AB::Var> = (*row).borrow();

        builder.assert_bool(prep.active);
        builder.assert_bool(prep.is_point);
        let active = AB::Expr::from(prep.active);
        let is_point = AB::Expr::from(prep.is_point);
        let is_opening = active.clone() - is_point.clone();
        for value in local.value {
            builder
                .when(AB::Expr::ONE - active.clone())
                .assert_zero(value);
        }

        self.source_point_bus.receive(
            builder,
            prep.source_reduction_idx,
            WhirOpeningPointMessage {
                idx: prep.item_idx.into(),
                value: local.value.map(Into::into),
            },
            is_point.clone(),
        );
        self.statement_buses.point.add_key_with_lookups(
            builder,
            prep.statement_proof_idx,
            MultiConstraintPointMessage {
                constraint_idx: prep.constraint_idx.into(),
                coordinate_idx: prep.item_idx.into(),
                value: local.value.map(Into::into),
            },
            is_point * prep.point_lookup_multiplicity,
        );

        self.source_opening_bus.receive(
            builder,
            prep.source_reduction_idx,
            OrderedStackingOpeningMessage {
                opening_idx: prep.item_idx.into(),
                value: local.value.map(Into::into),
            },
            is_opening.clone(),
        );
        self.statement_buses.opening.add_key_with_lookups(
            builder,
            prep.statement_proof_idx,
            MultiConstraintOpeningMessage {
                constraint_idx: prep.constraint_idx.into(),
                opening_idx: prep.item_idx.into(),
                value: local.value.map(Into::into),
            },
            is_opening,
        );
    }
}

/// Values copied into the handoff trace and constrained against stacking's isolated buses.
#[derive(Clone, Copy, Debug)]
pub struct OrderedStackingMultiConstraintStatement<'a> {
    pub reduced_cube_point: &'a [EF],
    pub stacking_openings: &'a [Vec<EF>],
}

impl OrderedStackingMultiConstraintAir {
    pub fn generate_ctx<SC: StarkProtocolConfig<F = F>>(
        &self,
        statements: &[OrderedStackingMultiConstraintStatement<'_>],
    ) -> Result<AirProvingContext<CpuBackend<SC>>, OrderedStackingMultiConstraintError> {
        if statements.len() != self.profile.statement_count {
            return Err(OrderedStackingMultiConstraintError::StatementCount {
                actual: statements.len(),
                expected: self.profile.statement_count,
            });
        }
        for (reduction, statement) in statements.iter().enumerate() {
            if statement.reduced_cube_point.len() != self.profile.point_dimension {
                return Err(
                    OrderedStackingMultiConstraintError::StatementPointDimension { reduction },
                );
            }
            if statement
                .stacking_openings
                .iter()
                .map(Vec::len)
                .ne(self.profile.commitment_widths[reduction].iter().copied())
            {
                return Err(
                    OrderedStackingMultiConstraintError::StatementCommitmentWidths { reduction },
                );
            }
        }
        let width = core::mem::size_of::<OrderedStackingStatementCols<u8>>();
        let height = self.profile.plans.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(height * width);
        for (row_idx, plan) in self.profile.plans.iter().enumerate() {
            let statement = &statements[plan.source_reduction_idx];
            let value = if plan.is_point {
                *statement.reduced_cube_point.get(plan.item_idx).ok_or(
                    OrderedStackingMultiConstraintError::StatementPointDimension {
                        reduction: plan.source_reduction_idx,
                    },
                )?
            } else {
                statement
                    .stacking_openings
                    .iter()
                    .flatten()
                    .nth(plan.item_idx)
                    .copied()
                    .ok_or(
                        OrderedStackingMultiConstraintError::StatementCommitmentWidths {
                            reduction: plan.source_reduction_idx,
                        },
                    )?
            };
            let cols: &mut OrderedStackingStatementCols<F> =
                values[row_idx * width..(row_idx + 1) * width].borrow_mut();
            cols.value
                .copy_from_slice(value.as_basis_coefficients_slice());
        }
        Ok(AirProvingContext::simple_no_pis(RowMajorMatrix::new(
            values, width,
        )))
    }
}
