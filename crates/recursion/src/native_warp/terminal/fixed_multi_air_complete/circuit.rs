//! Setup-owned AIR inventory for the genuine complete fixed-multi-AIR PESAT
//! terminal relation.
//!
//! This owner is deliberately limited to the nonlinear relation and its one
//! structured mapped-opening output. It never constructs the legacy
//! `FixedMultiAirPesatIndex` terminal and it does not own the downstream WHIR
//! verifier. The finite-v3 terminal owner connects the typed WHIR cursor and
//! structured buses to the coefficient-two-coset tail.

use std::{collections::BTreeSet, sync::Arc};

use openvm_stark_backend::{
    air_builders::inlined_cached::InlinedCachedAir,
    interaction::BusIndex,
    native_warp::{
        FixedMultiAirCompletePesatIndex, FixedMultiAirCompleteTerminalCircuitComponentKind,
        FixedMultiAirCompleteTerminalCircuitPlan, FixedMultiAirCompleteTerminalEqRole,
        FixedMultiAirCompleteTerminalLinearizer,
    },
    AirRef, AnyAir, PartitionedBaseAir, StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, F};

use super::*;
use crate::{
    bus::TranscriptBus,
    native_warp::terminal::fixed_multi_air::{
        FixedMultiAirMappedTermBus, FixedMultiAirStructuredClaimHeaderBus,
        FixedMultiAirStructuredPointBus,
    },
    system::BusIndexManager,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedMultiAirCompleteTerminalAirRole {
    Instance,
    TranscriptPrefix,
    Beta,
    Decomposition,
    RegionSequence,
    GlobalPoint,
    /// All setup-fixed Eq blocks physically packed into one committed table.
    BlockEqTable {
        table: usize,
    },
    LocalSumcheck {
        region: usize,
    },
    LocalOpenings {
        region: usize,
    },
    LocalFixed {
        region: usize,
    },
    LocalSelectors {
        region: usize,
    },
    LocalNodes {
        region: usize,
    },
    LocalEndpoint {
        region: usize,
    },
    InteractionSumcheck {
        region: usize,
    },
    InteractionOpenings {
        region: usize,
    },
    InteractionFixed {
        region: usize,
    },
    InteractionSelectors {
        region: usize,
    },
    InteractionNodes {
        region: usize,
    },
    InteractionDenominators {
        region: usize,
    },
    InteractionEndpoint {
        region: usize,
    },
    GlobalMapping,
    GlobalMappingPoints,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirCompleteTerminalCircuitError {
    Plan(&'static str),
    Overflow(&'static str),
    Prefix,
    Cursor,
    Local { region: usize },
    Interaction { region: usize },
    BlockEq,
    GlobalMapping,
    Air,
}

/// Private typed namespace. The transcript bus is supplied by the enclosing
/// resumed finite-v3 transcript owner and is therefore intentionally absent.
#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteTerminalBusInventory {
    pub statement_start_cursor: FixedMultiAirCompleteStatementStartCursorBus,
    pub binding: FixedMultiAirCompleteBindingBus,
    pub instance: FixedMultiAirCompleteInstanceValueBus,
    pub local_claim: FixedMultiAirCompleteLocalClaimBus,
    pub interaction_claim: FixedMultiAirCompleteInteractionClaimBus,
    pub region_sequence_start: FixedMultiAirCompleteRegionSequenceStartBus,
    pub beta_claim: FixedMultiAirCompleteBetaClaimBus,
    pub decomposition_receipt: FixedMultiAirCompleteDecompositionReceiptBus,
    pub region_cursor: FixedMultiAirCompleteRegionCursorBus,
    pub local_start: FixedMultiAirCompleteLocalRegionStartBus,
    pub interaction_start: FixedMultiAirCompleteInteractionRegionStartBus,
    pub local_sumcheck_final: FixedMultiAirCompleteLocalRegionSumcheckFinalBus,
    pub interaction_sumcheck_final: FixedMultiAirCompleteInteractionRegionSumcheckFinalBus,
    pub local_point: FixedMultiAirCompleteLocalRegionPointBus,
    pub interaction_point: FixedMultiAirCompleteInteractionRegionPointBus,
    pub local_opening: FixedMultiAirCompleteLocalRegionOpeningBus,
    pub interaction_opening: FixedMultiAirCompleteInteractionRegionOpeningBus,
    pub local_final: FixedMultiAirCompleteLocalRegionFinalEvaluationBus,
    pub interaction_final: FixedMultiAirCompleteInteractionRegionFinalEvaluationBus,
    pub opening_batch_start: FixedMultiAirCompleteOpeningBatchStartBus,
    pub global_point: FixedMultiAirCompleteGlobalPointBus,
    pub regional_point: FixedMultiAirCompleteRegionalPointBus,
    pub eq_weight: FixedMultiAirCompleteEqWeightBus,
    pub local_fixed_state: FixedMultiAirCompleteLocalFixedStateBus,
    pub local_fixed_value: FixedMultiAirCompleteLocalFixedValueBus,
    pub local_node: FixedMultiAirCompleteLocalNodeBus,
    pub interaction_fixed_state: FixedMultiAirCompleteInteractionFixedStateBus,
    pub interaction_fixed_opening: FixedMultiAirCompleteInteractionFixedOpeningBus,
    pub interaction_selector: FixedMultiAirCompleteInteractionSelectorBus,
    pub interaction_node: FixedMultiAirCompleteInteractionNodeBus,
    pub interaction_denominator: FixedMultiAirCompleteInteractionDenominatorBus,
    pub structured_header: FixedMultiAirStructuredClaimHeaderBus,
    pub mapped_term: FixedMultiAirMappedTermBus,
    pub structured_point: FixedMultiAirStructuredPointBus,
    pub whir_prefix_cursor: FixedMultiAirCompleteWhirPrefixCursorBus,
    next_bus_idx: BusIndex,
}

impl FixedMultiAirCompleteTerminalBusInventory {
    #[must_use]
    pub fn new(first_bus_idx: BusIndex) -> Self {
        let mut manager = BusIndexManager::from_next_bus_idx(first_bus_idx);
        macro_rules! bus {
            ($ty:ty) => {
                <$ty>::new(manager.new_bus_idx())
            };
        }
        let inventory = Self {
            statement_start_cursor: bus!(FixedMultiAirCompleteStatementStartCursorBus),
            binding: bus!(FixedMultiAirCompleteBindingBus),
            instance: bus!(FixedMultiAirCompleteInstanceValueBus),
            local_claim: bus!(FixedMultiAirCompleteLocalClaimBus),
            interaction_claim: bus!(FixedMultiAirCompleteInteractionClaimBus),
            region_sequence_start: bus!(FixedMultiAirCompleteRegionSequenceStartBus),
            beta_claim: bus!(FixedMultiAirCompleteBetaClaimBus),
            decomposition_receipt: bus!(FixedMultiAirCompleteDecompositionReceiptBus),
            region_cursor: bus!(FixedMultiAirCompleteRegionCursorBus),
            local_start: bus!(FixedMultiAirCompleteLocalRegionStartBus),
            interaction_start: bus!(FixedMultiAirCompleteInteractionRegionStartBus),
            local_sumcheck_final: bus!(FixedMultiAirCompleteLocalRegionSumcheckFinalBus),
            interaction_sumcheck_final: bus!(
                FixedMultiAirCompleteInteractionRegionSumcheckFinalBus
            ),
            local_point: bus!(FixedMultiAirCompleteLocalRegionPointBus),
            interaction_point: bus!(FixedMultiAirCompleteInteractionRegionPointBus),
            local_opening: bus!(FixedMultiAirCompleteLocalRegionOpeningBus),
            interaction_opening: bus!(FixedMultiAirCompleteInteractionRegionOpeningBus),
            local_final: bus!(FixedMultiAirCompleteLocalRegionFinalEvaluationBus),
            interaction_final: bus!(FixedMultiAirCompleteInteractionRegionFinalEvaluationBus),
            opening_batch_start: bus!(FixedMultiAirCompleteOpeningBatchStartBus),
            global_point: bus!(FixedMultiAirCompleteGlobalPointBus),
            regional_point: bus!(FixedMultiAirCompleteRegionalPointBus),
            eq_weight: bus!(FixedMultiAirCompleteEqWeightBus),
            local_fixed_state: bus!(FixedMultiAirCompleteLocalFixedStateBus),
            local_fixed_value: bus!(FixedMultiAirCompleteLocalFixedValueBus),
            local_node: bus!(FixedMultiAirCompleteLocalNodeBus),
            interaction_fixed_state: bus!(FixedMultiAirCompleteInteractionFixedStateBus),
            interaction_fixed_opening: bus!(FixedMultiAirCompleteInteractionFixedOpeningBus),
            interaction_selector: bus!(FixedMultiAirCompleteInteractionSelectorBus),
            interaction_node: bus!(FixedMultiAirCompleteInteractionNodeBus),
            interaction_denominator: bus!(FixedMultiAirCompleteInteractionDenominatorBus),
            structured_header: bus!(FixedMultiAirStructuredClaimHeaderBus),
            mapped_term: bus!(FixedMultiAirMappedTermBus),
            structured_point: bus!(FixedMultiAirStructuredPointBus),
            whir_prefix_cursor: bus!(FixedMultiAirCompleteWhirPrefixCursorBus),
            next_bus_idx: 0,
        };
        Self {
            next_bus_idx: manager.next_bus_idx(),
            ..inventory
        }
    }

    #[must_use]
    pub const fn next_bus_idx(&self) -> BusIndex {
        self.next_bus_idx
    }

    #[must_use]
    pub const fn prefix(&self, transcript: TranscriptBus) -> FixedMultiAirCompletePrefixBuses {
        FixedMultiAirCompletePrefixBuses {
            transcript,
            statement_start_cursor: self.statement_start_cursor,
            binding: self.binding,
            instance: self.instance,
            local_claim: self.local_claim,
            interaction_claim: self.interaction_claim,
            region_sequence_start: self.region_sequence_start,
            beta_claim: self.beta_claim,
            decomposition_receipt: self.decomposition_receipt,
        }
    }

    #[must_use]
    pub const fn cursor(
        &self,
        transcript: TranscriptBus,
    ) -> FixedMultiAirCompleteRegionCursorOpeningBuses {
        FixedMultiAirCompleteRegionCursorOpeningBuses {
            transcript,
            sequence_start: self.region_sequence_start,
            decomposition_receipt: self.decomposition_receipt,
            cursor: self.region_cursor,
            local_claim: self.local_claim,
            interaction_claim: self.interaction_claim,
            local_start: self.local_start,
            interaction_start: self.interaction_start,
            local_sumcheck_final: self.local_sumcheck_final,
            interaction_sumcheck_final: self.interaction_sumcheck_final,
            authorities: FixedMultiAirCompleteRegionalAuthorityBuses {
                local_point: self.local_point,
                interaction_point: self.interaction_point,
                local_opening: self.local_opening,
                interaction_opening: self.interaction_opening,
                local_final_evaluation: self.local_final,
                interaction_final_evaluation: self.interaction_final,
            },
            batch_start: self.opening_batch_start,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteTerminalCircuitProfile {
    pub relation: Arc<FixedMultiAirCompletePesatIndex<F, Digest>>,
    pub plan: Arc<FixedMultiAirCompleteTerminalCircuitPlan<F, Digest>>,
    pub prefix: Arc<FixedMultiAirCompletePrefixProfile>,
    pub cursor: Arc<FixedMultiAirCompleteRegionCursorOpeningProfile>,
    pub mapping: Arc<FixedMultiAirCompleteGlobalMappingProfile>,
    pub locals: Vec<Arc<FixedMultiAirCompleteLocalEndpointProfile>>,
    pub local_fixed: Vec<FixedMultiAirCompleteLocalFixedValues>,
    pub interactions: Vec<Option<Arc<FixedMultiAirCompleteInteractionEndpointProfile>>>,
    pub interaction_fixed: Vec<Option<Arc<FixedMultiAirCompleteInteractionFixedValues>>>,
    pub local_eq: Vec<Vec<FixedMultiAirCompleteBlockEqProfile>>,
    pub interaction_eq: Vec<Option<Vec<FixedMultiAirCompleteBlockEqProfile>>>,
    pub global_eq: Option<FixedMultiAirCompleteBlockEqProfile>,
    pub packed_eq: Vec<Arc<FixedMultiAirCompletePackedBlockEqProfile>>,
    pub global_point: FixedMultiAirCompleteGlobalPointBridgeProfile,
}

impl FixedMultiAirCompleteTerminalCircuitProfile {
    /// `downstream` names only consumers outside the nonlinear owner (the
    /// terminal package, coefficient-two-coset prefix/adjoint and receipt).
    /// This constructor adds every local, interaction, cursor and Eq consumer
    /// itself and rejects count overflow before AIR allocation.
    pub fn new(
        relation: Arc<FixedMultiAirCompletePesatIndex<F, Digest>>,
        mut downstream: FixedMultiAirCompleteExternalLookupCounts,
    ) -> Result<Self, FixedMultiAirCompleteTerminalCircuitError> {
        let plan = Arc::new(
            FixedMultiAirCompleteTerminalLinearizer::new(relation.as_ref())
                .and_then(|linearizer| linearizer.circuit_plan())
                .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::Plan("circuit plan"))?,
        );
        let region_count = plan.regions.len();
        let beta_len =
            usize::from(plan.metadata.log_constraints)
                .checked_add(usize::try_from(plan.metadata.explicit_len).map_err(|_| {
                    FixedMultiAirCompleteTerminalCircuitError::Overflow("beta length")
                })?)
                .ok_or(FixedMultiAirCompleteTerminalCircuitError::Overflow(
                    "beta length",
                ))?;
        if downstream.alpha.len() != usize::from(plan.metadata.code_class.log_codeword_len)
            || downstream.beta.len() != beta_len
            || downstream.local_claims.len() != region_count
            || downstream.interaction_claims.len() != region_count
        {
            return Err(FixedMultiAirCompleteTerminalCircuitError::Plan(
                "downstream lookup dimensions",
            ));
        }
        for region in 0..region_count {
            add_u32(
                &mut downstream.local_claims[region],
                1,
                "local cursor claim",
            )?;
            add_u32(
                &mut downstream.interaction_claims[region],
                1,
                "interaction cursor claim",
            )?;
        }

        let cursor = Arc::new(
            FixedMultiAirCompleteRegionCursorOpeningProfile::from_plan(plan.clone())
                .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::Cursor)?,
        );
        let mapping = Arc::new(
            FixedMultiAirCompleteGlobalMappingProfile::from_plan(plan.clone())
                .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::GlobalMapping)?,
        );
        if cursor.global_opening_count != mapping.opening_count
            || cursor.protocol_component_count() != mapping.protocol_component_count
        {
            return Err(FixedMultiAirCompleteTerminalCircuitError::Plan(
                "cursor/mapping coverage",
            ));
        }

        let global_weight_block = usize::try_from(plan.scalars.global_constraint.constraint_index)
            .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::Overflow("global block"))?;
        let exact_relation_degree = usize::from(plan.metadata.exact_relation_degree);
        let global_log_constraints = usize::from(plan.metadata.log_constraints);
        let global_explicit_len = usize::try_from(plan.metadata.explicit_len).map_err(|_| {
            FixedMultiAirCompleteTerminalCircuitError::Overflow("global explicit length")
        })?;
        let mut block_ids = BTreeSet::new();
        let mut locals = Vec::with_capacity(region_count);
        let mut local_fixed = Vec::with_capacity(region_count);
        let mut interactions = Vec::with_capacity(region_count);
        let mut interaction_fixed = Vec::with_capacity(region_count);
        let mut local_eq = Vec::with_capacity(region_count);
        let mut interaction_eq = Vec::with_capacity(region_count);
        let mut block_count = 0usize;
        let mut nonempty_interaction_regions = 0usize;

        for region in 0..region_count {
            let direct = relation.region_relation(region).ok_or(
                FixedMultiAirCompleteTerminalCircuitError::Plan("region relation"),
            )?;
            let cached = relation.region_fixed_cached(region).ok_or(
                FixedMultiAirCompleteTerminalCircuitError::Plan("region cached setup"),
            )?;
            let local = Arc::new(
                FixedMultiAirCompleteLocalEndpointProfile::from_circuit_plan(&plan, region)
                    .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::Local { region })?,
            );
            add_counts(
                &mut downstream.beta,
                local.instance_beta_lookup_counts(),
                "local beta lookup",
            )?;
            let fixed = FixedMultiAirCompleteLocalFixedValues::from_relation_setup(
                &local,
                cached,
                direct.fixed_trace(),
            )
            .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::Local { region })?;
            let mut eqs = Vec::with_capacity(local.component.constraints.len());
            for constraint in &local.component.constraints {
                let block = usize::try_from(constraint.constraint_index).map_err(|_| {
                    FixedMultiAirCompleteTerminalCircuitError::Overflow("local Eq block")
                })?;
                if !block_ids.insert(block) {
                    return Err(FixedMultiAirCompleteTerminalCircuitError::Plan(
                        "duplicate Eq block",
                    ));
                }
                eqs.push(
                    FixedMultiAirCompleteBlockEqProfile::from_constraint(block, region, constraint)
                        .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::BlockEq)?,
                );
                block_count = block_count.checked_add(1).ok_or(
                    FixedMultiAirCompleteTerminalCircuitError::Overflow("Eq block count"),
                )?;
            }

            let inverse_blocks = plan.regions[region]
                .interaction
                .interactions
                .iter()
                .map(|interaction| usize::try_from(interaction.inverse_constraint.constraint_index))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| {
                    FixedMultiAirCompleteTerminalCircuitError::Overflow("inverse Eq block")
                })?;
            let interaction = Arc::new(
                FixedMultiAirCompleteInteractionEndpointProfile::new(
                    FixedMultiAirCompleteInteractionEndpointPlanInput {
                        global_log_constraints,
                        global_explicit_len,
                        exact_relation_degree,
                        region: &plan.regions[region],
                        scalars: &plan.scalars,
                        inverse_weight_blocks: &inverse_blocks,
                        global_weight_block,
                    },
                )
                .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::Interaction { region })?,
            );
            let (interaction_profile, fixed_interaction, interaction_eqs) = if interaction
                .is_empty()
            {
                (None, None, None)
            } else {
                nonempty_interaction_regions = nonempty_interaction_regions.checked_add(1).ok_or(
                    FixedMultiAirCompleteTerminalCircuitError::Overflow("interaction region count"),
                )?;
                add_counts(
                    &mut downstream.beta,
                    interaction.instance_beta_lookup_counts(),
                    "interaction beta lookup",
                )?;
                let fixed = FixedMultiAirCompleteInteractionFixedValues::from_relation_setup(
                    &interaction,
                    cached,
                    direct.fixed_trace(),
                )
                .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::Interaction { region })?;
                let mut eqs = Vec::with_capacity(interaction.interaction_count());
                for (ordinal, interaction_plan) in
                    interaction.component.interactions.iter().enumerate()
                {
                    let block = inverse_blocks[ordinal];
                    if !block_ids.insert(block) {
                        return Err(FixedMultiAirCompleteTerminalCircuitError::Plan(
                            "duplicate Eq block",
                        ));
                    }
                    eqs.push(
                        FixedMultiAirCompleteBlockEqProfile::from_constraint(
                            block,
                            region,
                            &interaction_plan.inverse_constraint,
                        )
                        .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::BlockEq)?,
                    );
                    block_count = block_count.checked_add(1).ok_or(
                        FixedMultiAirCompleteTerminalCircuitError::Overflow("Eq block count"),
                    )?;
                }
                (Some(interaction), Some(Arc::new(fixed)), Some(eqs))
            };
            locals.push(local);
            local_fixed.push(fixed);
            local_eq.push(eqs);
            interactions.push(interaction_profile);
            interaction_fixed.push(fixed_interaction);
            interaction_eq.push(interaction_eqs);
        }

        let global_eq = if nonempty_interaction_regions == 0 {
            None
        } else {
            if !matches!(
                plan.scalars.global_constraint.role,
                FixedMultiAirCompleteTerminalEqRole::GlobalConstraint
            ) || !block_ids.insert(global_weight_block)
            {
                return Err(FixedMultiAirCompleteTerminalCircuitError::Plan(
                    "global Eq block",
                ));
            }
            block_count = block_count.checked_add(1).ok_or(
                FixedMultiAirCompleteTerminalCircuitError::Overflow("Eq block count"),
            )?;
            Some(
                FixedMultiAirCompleteBlockEqProfile::new_with_weight_multiplicity(
                    global_weight_block,
                    FixedMultiAirCompleteBlockEqRole::GlobalConstraint,
                    0,
                    plan.scalars.global_constraint.eq.clone(),
                    nonempty_interaction_regions,
                )
                .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::BlockEq)?,
            )
        };
        if block_count == 0 {
            return Err(FixedMultiAirCompleteTerminalCircuitError::Plan(
                "empty Eq inventory",
            ));
        }
        for count in &mut downstream.beta[..global_log_constraints] {
            add_u32(count, 1, "global tau authority")?;
        }
        let global_point =
            FixedMultiAirCompleteGlobalPointBridgeProfile::new(global_log_constraints, block_count)
                .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::BlockEq)?;
        let mut packed_blocks = Vec::with_capacity(block_count);
        for component in &cursor.components {
            match component.kind {
                FixedMultiAirCompleteTerminalCircuitComponentKind::Local => {
                    packed_blocks.extend(local_eq[component.region].iter().cloned());
                }
                FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction => {
                    packed_blocks.extend(
                        interaction_eq[component.region]
                            .as_ref()
                            .ok_or(FixedMultiAirCompleteTerminalCircuitError::Interaction {
                                region: component.region,
                            })?
                            .iter()
                            .cloned(),
                    );
                }
            }
        }
        packed_blocks.extend(global_eq.iter().cloned());
        if packed_blocks.len() != block_count {
            return Err(FixedMultiAirCompleteTerminalCircuitError::Plan(
                "packed Eq coverage",
            ));
        }
        let packed_eq = FixedMultiAirCompletePackedBlockEqProfile::partition(
            packed_blocks,
            FIXED_MULTI_AIR_COMPLETE_PACKED_BLOCK_EQ_MAX_ROWS,
        )
        .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::BlockEq)?
        .into_iter()
        .map(Arc::new)
        .collect();
        let prefix = Arc::new(
            FixedMultiAirCompletePrefixProfile::from_relation(relation.as_ref(), downstream)
                .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::Prefix)?,
        );

        validate_component_multiplicities(
            &cursor,
            &locals,
            &interactions,
            mapping.mapper_opening_lookup_count(),
        )?;
        Ok(Self {
            relation,
            plan,
            prefix,
            cursor,
            mapping,
            locals,
            local_fixed,
            interactions,
            interaction_fixed,
            local_eq,
            interaction_eq,
            global_eq,
            packed_eq,
            global_point,
        })
    }
}

pub struct FixedMultiAirCompleteTerminalCircuit {
    pub profile: Arc<FixedMultiAirCompleteTerminalCircuitProfile>,
    pub buses: FixedMultiAirCompleteTerminalBusInventory,
    pub transcript_bus: TranscriptBus,
    pub proof_idx: usize,
}

impl FixedMultiAirCompleteTerminalCircuit {
    #[must_use]
    pub fn new(
        profile: Arc<FixedMultiAirCompleteTerminalCircuitProfile>,
        transcript_bus: TranscriptBus,
        proof_idx: usize,
        first_bus_idx: BusIndex,
    ) -> Self {
        Self {
            profile,
            buses: FixedMultiAirCompleteTerminalBusInventory::new(first_bus_idx),
            transcript_bus,
            proof_idx,
        }
    }

    #[must_use]
    pub const fn next_bus_idx(&self) -> BusIndex {
        self.buses.next_bus_idx()
    }

    #[must_use]
    pub fn air_roles(&self) -> Vec<FixedMultiAirCompleteTerminalAirRole> {
        let mut roles = vec![
            FixedMultiAirCompleteTerminalAirRole::Instance,
            FixedMultiAirCompleteTerminalAirRole::TranscriptPrefix,
            FixedMultiAirCompleteTerminalAirRole::Beta,
            FixedMultiAirCompleteTerminalAirRole::Decomposition,
            FixedMultiAirCompleteTerminalAirRole::RegionSequence,
            FixedMultiAirCompleteTerminalAirRole::GlobalPoint,
        ];
        roles.extend(
            (0..self.profile.packed_eq.len())
                .map(|table| FixedMultiAirCompleteTerminalAirRole::BlockEqTable { table }),
        );
        for component in &self.profile.cursor.components {
            let region = component.region;
            match component.kind {
                FixedMultiAirCompleteTerminalCircuitComponentKind::Local => {
                    roles.extend([
                        FixedMultiAirCompleteTerminalAirRole::LocalSumcheck { region },
                        FixedMultiAirCompleteTerminalAirRole::LocalOpenings { region },
                        FixedMultiAirCompleteTerminalAirRole::LocalFixed { region },
                        FixedMultiAirCompleteTerminalAirRole::LocalSelectors { region },
                        FixedMultiAirCompleteTerminalAirRole::LocalNodes { region },
                    ]);
                    roles.push(FixedMultiAirCompleteTerminalAirRole::LocalEndpoint { region });
                }
                FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction => {
                    roles.extend([
                        FixedMultiAirCompleteTerminalAirRole::InteractionSumcheck { region },
                        FixedMultiAirCompleteTerminalAirRole::InteractionOpenings { region },
                        FixedMultiAirCompleteTerminalAirRole::InteractionFixed { region },
                        FixedMultiAirCompleteTerminalAirRole::InteractionSelectors { region },
                        FixedMultiAirCompleteTerminalAirRole::InteractionNodes { region },
                        FixedMultiAirCompleteTerminalAirRole::InteractionDenominators { region },
                    ]);
                    roles
                        .push(FixedMultiAirCompleteTerminalAirRole::InteractionEndpoint { region });
                }
            }
        }
        roles.extend([
            FixedMultiAirCompleteTerminalAirRole::GlobalMapping,
            FixedMultiAirCompleteTerminalAirRole::GlobalMappingPoints,
        ]);
        roles
    }

    pub fn airs<SC: StarkProtocolConfig<F = F>>(
        &self,
    ) -> Result<Vec<AirRef<SC>>, FixedMultiAirCompleteTerminalCircuitError> {
        let p = &self.profile;
        let b = &self.buses;
        let prefix = FixedMultiAirCompletePrefixDecompositionAirs::new(
            p.prefix.clone(),
            self.proof_idx,
            b.prefix(self.transcript_bus),
        );
        let cursor = FixedMultiAirCompleteRegionCursorOpeningAirs::new(
            p.cursor.clone(),
            b.cursor(self.transcript_bus),
        )
        .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::Cursor)?;
        let mut opening_airs = cursor.components.into_iter();
        let mut airs = Vec::new();
        add_inlined(&mut airs, prefix.instance);
        add_inlined(&mut airs, prefix.transcript_prefix);
        add_inlined(&mut airs, prefix.beta);
        add_inlined(&mut airs, prefix.decomposition);
        add_inlined(&mut airs, cursor.sequence);
        add_inlined(
            &mut airs,
            FixedMultiAirCompleteGlobalPointBridgeAir {
                profile: p.global_point,
                instance_bus: b.instance,
                global_point_bus: b.global_point,
            },
        );
        for profile in &p.packed_eq {
            add(
                &mut airs,
                FixedMultiAirCompletePackedBlockEqAir {
                    profile: profile.clone(),
                    global_point_bus: b.global_point,
                    regional_point_bus: b.regional_point,
                    weight_bus: b.eq_weight,
                },
            );
        }
        for component in &p.cursor.components {
            let opening = opening_airs
                .next()
                .ok_or(FixedMultiAirCompleteTerminalCircuitError::Air)?;
            let region = component.region;
            match component.kind {
                FixedMultiAirCompleteTerminalCircuitComponentKind::Local => {
                    add(
                        &mut airs,
                        FixedMultiAirCompleteLocalRegionSumcheckAir::new(
                            self.transcript_bus,
                            b.local_start,
                            b.local_point,
                            b.local_sumcheck_final,
                            region,
                            component.round_count,
                            component.opening_count,
                            component.point_fixed_source_count,
                            component.point_common_count,
                        )
                        .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::Air)?,
                    );
                    add_inlined(&mut airs, opening);
                    let profile = p.locals[region].clone();
                    let endpoint = FixedMultiAirCompleteLocalEndpointAirs::new(
                        profile,
                        FixedMultiAirCompleteLocalEndpointBuses {
                            point: b.local_point,
                            block_eq_point: b.regional_point,
                            opening: b.local_opening,
                            final_evaluation: b.local_final,
                            instance: b.instance,
                            eq_weight: b.eq_weight,
                            fixed_state: b.local_fixed_state,
                            fixed_value: b.local_fixed_value,
                            node: b.local_node,
                        },
                    );
                    add_inlined(&mut airs, endpoint.fixed);
                    add_inlined(&mut airs, endpoint.selector);
                    add_inlined(&mut airs, endpoint.nodes);
                    add_inlined(&mut airs, endpoint.fold);
                }
                FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction => {
                    add(
                        &mut airs,
                        FixedMultiAirCompleteInteractionRegionSumcheckAir::new(
                            self.transcript_bus,
                            b.interaction_start,
                            b.interaction_point,
                            b.interaction_sumcheck_final,
                            region,
                            component.round_count,
                            component.opening_count,
                            component.point_fixed_source_count,
                            component.point_common_count,
                        )
                        .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::Air)?,
                    );
                    add_inlined(&mut airs, opening);
                    let profile = p.interactions[region]
                        .clone()
                        .ok_or(FixedMultiAirCompleteTerminalCircuitError::Interaction { region })?;
                    add(
                        &mut airs,
                        FixedMultiAirCompleteInteractionFixedAir {
                            profile: profile.clone(),
                            setup: p.interaction_fixed[region].clone().ok_or(
                                FixedMultiAirCompleteTerminalCircuitError::Interaction { region },
                            )?,
                            point_bus: b.interaction_point,
                            state_bus: b.interaction_fixed_state,
                            opening_bus: b.interaction_fixed_opening,
                        },
                    );
                    add_inlined(
                        &mut airs,
                        FixedMultiAirCompleteInteractionSelectorAir {
                            profile: profile.clone(),
                            point_bus: b.interaction_point,
                            block_eq_point_bus: b.regional_point,
                            selector_bus: b.interaction_selector,
                        },
                    );
                    add_inlined(
                        &mut airs,
                        FixedMultiAirCompleteInteractionNodeAir {
                            profile: profile.clone(),
                            opening_bus: b.interaction_opening,
                            fixed_opening_bus: b.interaction_fixed_opening,
                            instance_bus: b.instance,
                            selector_bus: b.interaction_selector,
                            node_bus: b.interaction_node,
                        },
                    );
                    add_inlined(
                        &mut airs,
                        FixedMultiAirCompleteInteractionDenominatorAir {
                            profile: profile.clone(),
                            instance_bus: b.instance,
                            node_bus: b.interaction_node,
                            denominator_bus: b.interaction_denominator,
                        },
                    );
                    add_inlined(
                        &mut airs,
                        FixedMultiAirCompleteInteractionEndpointAir {
                            profile,
                            opening_bus: b.interaction_opening,
                            final_evaluation_bus: b.interaction_final,
                            instance_bus: b.instance,
                            node_bus: b.interaction_node,
                            denominator_bus: b.interaction_denominator,
                            eq_weight_bus: b.eq_weight,
                        },
                    );
                }
            }
        }
        if opening_airs.next().is_some() {
            return Err(FixedMultiAirCompleteTerminalCircuitError::Air);
        }
        add_inlined(
            &mut airs,
            FixedMultiAirCompleteGlobalMappingAir {
                profile: p.mapping.clone(),
                transcript_bus: self.transcript_bus,
                batch_start_bus: b.opening_batch_start,
                local_opening_bus: b.local_opening,
                interaction_opening_bus: b.interaction_opening,
                header_bus: b.structured_header,
                term_bus: b.mapped_term,
                whir_prefix_cursor_bus: b.whir_prefix_cursor,
            },
        );
        add_inlined(
            &mut airs,
            FixedMultiAirCompleteGlobalMappingPointAir {
                profile: p.mapping.clone(),
                local_point_bus: b.local_point,
                interaction_point_bus: b.interaction_point,
                structured_point_bus: b.structured_point,
            },
        );
        if airs.len() != self.air_roles().len() {
            return Err(FixedMultiAirCompleteTerminalCircuitError::Air);
        }
        Ok(airs)
    }
}

fn validate_component_multiplicities(
    cursor: &FixedMultiAirCompleteRegionCursorOpeningProfile,
    locals: &[Arc<FixedMultiAirCompleteLocalEndpointProfile>],
    interactions: &[Option<Arc<FixedMultiAirCompleteInteractionEndpointProfile>>],
    mapper_count: u32,
) -> Result<(), FixedMultiAirCompleteTerminalCircuitError> {
    for component in &cursor.components {
        let (endpoint_counts, fixed_sources) = match component.kind {
            FixedMultiAirCompleteTerminalCircuitComponentKind::Local => {
                let profile = &locals[component.region];
                (
                    profile.opening_lookup_counts(),
                    profile.point_lookup_count_per_coordinate(),
                )
            }
            FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction => {
                let profile = interactions[component.region].as_ref().ok_or(
                    FixedMultiAirCompleteTerminalCircuitError::Interaction {
                        region: component.region,
                    },
                )?;
                (
                    profile.opening_lookup_counts(),
                    profile.point_fixed_source_count(),
                )
            }
        };
        if endpoint_counts.len() != component.opening_lookup_counts.len()
            || component.point_fixed_source_count != fixed_sources
            || component.point_common_count != 1 + endpoint_counts.len()
            || endpoint_counts
                .iter()
                .zip(&component.opening_lookup_counts)
                .any(|(&endpoint, &published)| {
                    u32::try_from(endpoint)
                        .ok()
                        .and_then(|value| value.checked_add(mapper_count))
                        != Some(published)
                })
        {
            return Err(FixedMultiAirCompleteTerminalCircuitError::Plan(
                "component lookup multiplicity",
            ));
        }
    }
    Ok(())
}

fn add_counts(
    destination: &mut [u32],
    source: &[usize],
    context: &'static str,
) -> Result<(), FixedMultiAirCompleteTerminalCircuitError> {
    if destination.len() != source.len() {
        return Err(FixedMultiAirCompleteTerminalCircuitError::Plan(context));
    }
    for (destination, &source) in destination.iter_mut().zip(source) {
        let source = u32::try_from(source)
            .map_err(|_| FixedMultiAirCompleteTerminalCircuitError::Overflow(context))?;
        add_u32(destination, source, context)?;
    }
    Ok(())
}

fn add_u32(
    destination: &mut u32,
    value: u32,
    context: &'static str,
) -> Result<(), FixedMultiAirCompleteTerminalCircuitError> {
    *destination = destination
        .checked_add(value)
        .ok_or(FixedMultiAirCompleteTerminalCircuitError::Overflow(context))?;
    Ok(())
}

fn add<SC, A>(airs: &mut Vec<AirRef<SC>>, air: A)
where
    SC: StarkProtocolConfig,
    A: AnyAir<SC> + 'static,
{
    airs.push(Arc::new(air));
}

fn add_inlined<SC, A>(airs: &mut Vec<AirRef<SC>>, air: A)
where
    SC: StarkProtocolConfig,
    A: PartitionedBaseAir<SC::F>,
    InlinedCachedAir<A>: AnyAir<SC> + 'static,
{
    airs.push(Arc::new(InlinedCachedAir::new(air)));
}
