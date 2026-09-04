//! Typed bridge from the genuine complete-relation accumulator catalog to the
//! relation-independent native terminal RS-adjoint consumers.
//!
//! The row layout and witness generation are deliberately shared with the
//! legacy bridge: they only enumerate the four public accumulator sections and
//! their setup-owned lookup multiplicities.  The AIR is not shared because its
//! source bus is part of the protocol type system.  In particular, this bridge
//! cannot consume a legacy local-only terminal instance.

use core::borrow::Borrow;

use openvm_circuit_primitives::ColumnsAir;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder, interaction::InteractionBuilder, BaseAirWithPublicValues,
    PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use p3_air::{Air, BaseAir};
use p3_matrix::Matrix;

use super::{FixedMultiAirCompleteInstanceValueBus, FixedMultiAirCompleteInstanceValueMessage};
pub use crate::native_warp::terminal::fixed_multi_air::{
    generate_fixed_multi_air_accumulator_bridge_traces as generate_fixed_multi_air_complete_accumulator_bridge_traces,
    FixedMultiAirAccumulatorBridgeCols as FixedMultiAirCompleteAccumulatorBridgeCols,
    FixedMultiAirAccumulatorBridgeScheduleCols as FixedMultiAirCompleteAccumulatorBridgeScheduleCols,
    FixedMultiAirStatementTraceError as FixedMultiAirCompleteAccumulatorBridgeTraceError,
};
use crate::native_warp::terminal::{
    NativeTerminalAccumulatorValueBus, NativeTerminalAccumulatorValueMessage,
};

/// Explicit terminal spelling for the complete relation's canonical instance
/// catalog bus.  This is an alias, not a second protocol namespace.
pub type FixedMultiAirCompleteTerminalInstanceValueBus = FixedMultiAirCompleteInstanceValueBus;

/// Explicit terminal spelling for the complete relation's canonical instance
/// catalog message.  This is an alias, not a second message encoding.
pub type FixedMultiAirCompleteTerminalInstanceValueMessage<T> =
    FixedMultiAirCompleteInstanceValueMessage<T>;

/// Publishes the authenticated complete-relation accumulator values to the
/// generic terminal machinery.
///
/// The cached schedule determines `(section, coordinate, multiplicity)`.  The
/// common trace carries only the value.  Consequently no host assertion and no
/// AIR-name lookup can substitute an accumulator value: it must balance against
/// the complete terminal's typed instance catalog.
pub struct FixedMultiAirCompleteAccumulatorBridgeAir {
    pub fixed_bus: FixedMultiAirCompleteTerminalInstanceValueBus,
    pub native_bus: NativeTerminalAccumulatorValueBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteAccumulatorBridgeAir {}

impl PartitionedBaseAir<F> for FixedMultiAirCompleteAccumulatorBridgeAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteAccumulatorBridgeScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteAccumulatorBridgeCols::<F>::width()
    }
}

impl ColumnsAir for FixedMultiAirCompleteAccumulatorBridgeAir {}

impl BaseAir<F> for FixedMultiAirCompleteAccumulatorBridgeAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteAccumulatorBridgeScheduleCols::<F>::width()
            + FixedMultiAirCompleteAccumulatorBridgeCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteAccumulatorBridgeAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("complete accumulator bridge schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("complete accumulator bridge row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteAccumulatorBridgeScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let local: &FixedMultiAirCompleteAccumulatorBridgeCols<AB::Var> =
            common.as_slice().borrow();

        builder.assert_bool(schedule.active);
        self.fixed_bus.lookup_key(
            builder,
            FixedMultiAirCompleteTerminalInstanceValueMessage {
                section: schedule.section.into(),
                coordinate: schedule.coordinate.into(),
                value: local.value.map(Into::into),
            },
            schedule.active,
        );
        self.native_bus.add_key_with_lookups(
            builder,
            NativeTerminalAccumulatorValueMessage {
                section: schedule.section.into(),
                coordinate: schedule.coordinate.into(),
                value: local.value.map(Into::into),
            },
            schedule.native_lookup_count,
        );
    }
}

#[cfg(test)]
mod tests {
    use core::{any::TypeId, borrow::BorrowMut};
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use openvm_stark_backend::{
        air_builders::{
            debug::{check_constraints, check_logup},
            symbolic::{get_symbolic_builder, SymbolicRapBuilder},
            PartitionedAirBuilder,
        },
        interaction::{InteractionBuilder, SymbolicInteraction},
        keygen::types::TraceWidth,
        warp_pesat::AccumulatorInstance,
        BaseAirWithPublicValues, PartitionedBaseAir,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::{
        BabyBearPoseidon2Config as SC, Digest, DIGEST_SIZE, D_EF, EF, F,
    };
    use p3_air::{Air, BaseAir};
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
    use p3_matrix::{dense::RowMajorMatrix, Matrix};

    use super::*;
    use crate::native_warp::terminal::fixed_multi_air::FixedMultiAirTerminalInstanceValueBus;

    /// Independent test oracle: it owns the complete catalog entries and
    /// consumes exactly the generic values requested by the downstream tail.
    struct BridgeOracleAir {
        complete_bus: FixedMultiAirCompleteTerminalInstanceValueBus,
        native_bus: NativeTerminalAccumulatorValueBus,
    }

    impl BaseAirWithPublicValues<F> for BridgeOracleAir {}

    impl PartitionedBaseAir<F> for BridgeOracleAir {
        fn cached_main_widths(&self) -> Vec<usize> {
            vec![FixedMultiAirCompleteAccumulatorBridgeScheduleCols::<F>::width()]
        }

        fn common_main_width(&self) -> usize {
            FixedMultiAirCompleteAccumulatorBridgeCols::<F>::width()
        }
    }

    impl BaseAir<F> for BridgeOracleAir {
        fn width(&self) -> usize {
            FixedMultiAirCompleteAccumulatorBridgeScheduleCols::<F>::width()
                + FixedMultiAirCompleteAccumulatorBridgeCols::<F>::width()
        }
    }

    impl<AB> Air<AB> for BridgeOracleAir
    where
        AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            let cached = builder.cached_mains()[0]
                .row_slice(0)
                .expect("complete accumulator oracle schedule")
                .to_vec();
            let common = builder
                .common_main()
                .row_slice(0)
                .expect("complete accumulator oracle row")
                .to_vec();
            let schedule: &FixedMultiAirCompleteAccumulatorBridgeScheduleCols<AB::Var> =
                cached.as_slice().borrow();
            let local: &FixedMultiAirCompleteAccumulatorBridgeCols<AB::Var> =
                common.as_slice().borrow();

            builder.assert_bool(schedule.active);
            self.complete_bus.add_key_with_lookups(
                builder,
                FixedMultiAirCompleteTerminalInstanceValueMessage {
                    section: schedule.section.into(),
                    coordinate: schedule.coordinate.into(),
                    value: local.value.map(Into::into),
                },
                schedule.active,
            );
            self.native_bus.lookup_key(
                builder,
                NativeTerminalAccumulatorValueMessage {
                    section: schedule.section.into(),
                    coordinate: schedule.coordinate.into(),
                    value: local.value.map(Into::into),
                },
                schedule.native_lookup_count,
            );
        }
    }

    fn ef4(seed: u64) -> EF {
        EF::from_basis_coefficients_slice(&[
            F::from_u64(seed),
            F::from_u64(seed + 1),
            F::from_u64(seed + 2),
            F::from_u64(seed + 3),
        ])
        .expect("EF4")
    }

    fn instance() -> AccumulatorInstance<EF, Digest> {
        AccumulatorInstance {
            rt: [F::from_u32(91); DIGEST_SIZE],
            alpha: vec![ef4(11), ef4(21), ef4(31)],
            mu: ef4(41),
            beta: vec![ef4(51), ef4(61)],
            eta: ef4(71),
        }
    }

    fn counts() -> [Vec<usize>; 4] {
        [vec![1, 3, 0], vec![2], vec![4, 1], vec![2]]
    }

    fn symbolic_interactions<A>(air: &A) -> Vec<SymbolicInteraction<F>>
    where
        A: Air<SymbolicRapBuilder<F>>
            + BaseAir<F>
            + BaseAirWithPublicValues<F>
            + PartitionedBaseAir<F>,
    {
        get_symbolic_builder(
            air,
            &TraceWidth {
                preprocessed: None,
                cached_mains: air.cached_main_widths(),
                common_main: air.common_main_width(),
            },
        )
        .constraints()
        .interactions
    }

    fn check_balance(
        bridge: &FixedMultiAirCompleteAccumulatorBridgeAir,
        bridge_cached: &RowMajorMatrix<F>,
        bridge_common: &RowMajorMatrix<F>,
        oracle: &BridgeOracleAir,
        oracle_cached: &RowMajorMatrix<F>,
        oracle_common: &RowMajorMatrix<F>,
    ) {
        check_logup(
            &[
                "complete bridge".to_owned(),
                "independent oracle".to_owned(),
            ],
            &[symbolic_interactions(bridge), symbolic_interactions(oracle)],
            &[None, None],
            &[
                vec![bridge_cached.as_view(), bridge_common.as_view()],
                vec![oracle_cached.as_view(), oracle_common.as_view()],
            ],
            &[Vec::new(), Vec::new()],
        );
    }

    fn assert_balance_rejects(
        bridge: &FixedMultiAirCompleteAccumulatorBridgeAir,
        bridge_cached: &RowMajorMatrix<F>,
        bridge_common: &RowMajorMatrix<F>,
        oracle: &BridgeOracleAir,
        oracle_cached: &RowMajorMatrix<F>,
        oracle_common: &RowMajorMatrix<F>,
    ) {
        assert!(catch_unwind(AssertUnwindSafe(|| {
            check_balance(
                bridge,
                bridge_cached,
                bridge_common,
                oracle,
                oracle_cached,
                oracle_common,
            );
        }))
        .is_err());
    }

    #[test]
    fn complete_accumulator_bridge_balances_exact_catalog_and_native_values() {
        assert_ne!(
            TypeId::of::<FixedMultiAirCompleteTerminalInstanceValueBus>(),
            TypeId::of::<FixedMultiAirTerminalInstanceValueBus>(),
            "complete bridge must not accept the legacy catalog bus"
        );

        let complete_bus = FixedMultiAirCompleteTerminalInstanceValueBus::new(1201);
        let native_bus = NativeTerminalAccumulatorValueBus::new(1202);
        let bridge = FixedMultiAirCompleteAccumulatorBridgeAir {
            fixed_bus: complete_bus,
            native_bus,
        };
        let oracle = BridgeOracleAir {
            complete_bus,
            native_bus,
        };
        let (cached, common) = generate_fixed_multi_air_complete_accumulator_bridge_traces(
            &instance(),
            &counts(),
            Some(8),
        )
        .expect("complete accumulator bridge trace");

        check_constraints::<_, SC>(
            &bridge,
            "complete accumulator bridge",
            &None,
            &[cached.as_view(), common.as_view()],
            &[],
        );
        check_constraints::<_, SC>(
            &oracle,
            "complete accumulator bridge oracle",
            &None,
            &[cached.as_view(), common.as_view()],
            &[],
        );
        check_balance(&bridge, &cached, &common, &oracle, &cached, &common);

        let schedule_width = FixedMultiAirCompleteAccumulatorBridgeScheduleCols::<F>::width();
        let value_width = FixedMultiAirCompleteAccumulatorBridgeCols::<F>::width();
        let expected_sections = [0usize, 0, 0, 1, 2, 2, 3];
        let expected_coordinates = [0usize, 1, 2, 0, 0, 1, 0];
        for row in 0..expected_sections.len() {
            let schedule: &FixedMultiAirCompleteAccumulatorBridgeScheduleCols<F> =
                cached.values[row * schedule_width..(row + 1) * schedule_width].borrow();
            let value: &FixedMultiAirCompleteAccumulatorBridgeCols<F> =
                common.values[row * value_width..(row + 1) * value_width].borrow();
            assert_eq!(schedule.active, F::ONE);
            assert_eq!(schedule.section, F::from_usize(expected_sections[row]));
            assert_eq!(
                schedule.coordinate,
                F::from_usize(expected_coordinates[row])
            );
            assert_ne!(value.value, [F::ZERO; D_EF]);
        }
        let padding: &FixedMultiAirCompleteAccumulatorBridgeScheduleCols<F> =
            cached.values[7 * schedule_width..8 * schedule_width].borrow();
        assert_eq!(padding.active, F::ZERO);
    }

    #[test]
    fn complete_accumulator_bridge_rejects_value_section_count_and_active_mutations() {
        let complete_bus = FixedMultiAirCompleteTerminalInstanceValueBus::new(1211);
        let native_bus = NativeTerminalAccumulatorValueBus::new(1212);
        let bridge = FixedMultiAirCompleteAccumulatorBridgeAir {
            fixed_bus: complete_bus,
            native_bus,
        };
        let oracle = BridgeOracleAir {
            complete_bus,
            native_bus,
        };
        let (cached, common) = generate_fixed_multi_air_complete_accumulator_bridge_traces(
            &instance(),
            &counts(),
            Some(8),
        )
        .expect("complete accumulator bridge trace");

        let value_width = FixedMultiAirCompleteAccumulatorBridgeCols::<F>::width();
        let mut bad_value = common.clone();
        let value: &mut FixedMultiAirCompleteAccumulatorBridgeCols<F> =
            bad_value.values[value_width..2 * value_width].borrow_mut();
        value.value[2] += F::ONE;
        assert_balance_rejects(&bridge, &cached, &bad_value, &oracle, &cached, &common);

        let schedule_width = FixedMultiAirCompleteAccumulatorBridgeScheduleCols::<F>::width();
        let mut bad_section = cached.clone();
        let schedule: &mut FixedMultiAirCompleteAccumulatorBridgeScheduleCols<F> =
            bad_section.values[3 * schedule_width..4 * schedule_width].borrow_mut();
        schedule.section += F::ONE;
        assert_balance_rejects(&bridge, &bad_section, &common, &oracle, &cached, &common);

        let mut bad_count = cached.clone();
        let schedule: &mut FixedMultiAirCompleteAccumulatorBridgeScheduleCols<F> =
            bad_count.values[4 * schedule_width..5 * schedule_width].borrow_mut();
        schedule.native_lookup_count += F::ONE;
        assert_balance_rejects(&bridge, &bad_count, &common, &oracle, &cached, &common);

        let mut bad_active = cached.clone();
        let schedule: &mut FixedMultiAirCompleteAccumulatorBridgeScheduleCols<F> =
            bad_active.values[..schedule_width].borrow_mut();
        schedule.active = F::TWO;
        assert!(catch_unwind(AssertUnwindSafe(|| {
            check_constraints::<_, SC>(
                &bridge,
                "mutated complete accumulator bridge active flag",
                &None,
                &[bad_active.as_view(), common.as_view()],
                &[],
            );
        }))
        .is_err());
    }

    #[test]
    fn complete_accumulator_bridge_trace_rejects_bad_shapes_and_short_height() {
        let mut bad_counts = counts();
        bad_counts[2].pop();
        assert!(generate_fixed_multi_air_complete_accumulator_bridge_traces(
            &instance(),
            &bad_counts,
            None,
        )
        .is_err());
        assert!(generate_fixed_multi_air_complete_accumulator_bridge_traces(
            &instance(),
            &counts(),
            Some(6),
        )
        .is_err());
    }
}
