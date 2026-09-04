//! Trace assembly for [`FixedMultiAirCompleteTerminalCircuit`].
//!
//! AIR and trace order are both derived from the circuit's semantic role
//! inventory. No AIR-name lookup or legacy terminal proof conversion occurs.

use openvm_stark_backend::{
    native_warp::{
        FixedMultiAirCompleteTerminalCircuitComponentKind, FixedMultiAirCompleteTerminalProof,
    },
    transcript::TranscriptLog,
    warp_pesat::AccumulatorInstance,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, EF, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::*;

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteTerminalPartitionedTrace {
    pub cached_mains: Vec<RowMajorMatrix<F>>,
    pub common_main: RowMajorMatrix<F>,
}

impl FixedMultiAirCompleteTerminalPartitionedTrace {
    #[must_use]
    pub fn simple(common_main: RowMajorMatrix<F>) -> Self {
        Self {
            cached_mains: Vec::new(),
            common_main,
        }
    }

    #[must_use]
    pub fn cached(cached: RowMajorMatrix<F>, common_main: RowMajorMatrix<F>) -> Self {
        Self {
            cached_mains: vec![cached],
            common_main,
        }
    }

    /// Move proof-local cached partitions into the ordinary trace commitment.
    /// Column order is all former cached partitions followed by the former
    /// common partition, matching `InlinedCachedAirBuilder`.
    pub fn inline_cached_mains(&mut self) -> Result<(), FixedMultiAirCompleteTerminalTraceError> {
        if self.cached_mains.is_empty() {
            return Ok(());
        }
        let height = self.common_main.height();
        if height == 0
            || self
                .cached_mains
                .iter()
                .any(|matrix| matrix.height() != height)
        {
            return Err(FixedMultiAirCompleteTerminalTraceError::Layout);
        }
        let total_width = self
            .cached_mains
            .iter()
            .try_fold(self.common_main.width(), |width, matrix| {
                width.checked_add(matrix.width())
            })
            .ok_or(FixedMultiAirCompleteTerminalTraceError::Layout)?;
        let cells = height
            .checked_mul(total_width)
            .ok_or(FixedMultiAirCompleteTerminalTraceError::Layout)?;
        let mut values = Vec::with_capacity(cells);
        for row in 0..height {
            for matrix in &self.cached_mains {
                let width = matrix.width();
                let start = row
                    .checked_mul(width)
                    .ok_or(FixedMultiAirCompleteTerminalTraceError::Layout)?;
                values.extend_from_slice(
                    matrix
                        .values
                        .get(start..start + width)
                        .ok_or(FixedMultiAirCompleteTerminalTraceError::Layout)?,
                );
            }
            let width = self.common_main.width();
            let start = row
                .checked_mul(width)
                .ok_or(FixedMultiAirCompleteTerminalTraceError::Layout)?;
            values.extend_from_slice(
                self.common_main
                    .values
                    .get(start..start + width)
                    .ok_or(FixedMultiAirCompleteTerminalTraceError::Layout)?,
            );
        }
        if values.len() != cells {
            return Err(FixedMultiAirCompleteTerminalTraceError::Layout);
        }
        self.cached_mains.clear();
        self.common_main = RowMajorMatrix::new(values, total_width);
        Ok(())
    }
}

pub struct FixedMultiAirCompleteTerminalTraceWitness<'a> {
    pub authenticated_root: Digest,
    pub instance: &'a AccumulatorInstance<EF, Digest>,
    pub proof: &'a FixedMultiAirCompleteTerminalProof<EF>,
    pub transcript: &'a TranscriptLog<F, [F; 16]>,
    pub statement_start_tidx: usize,
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteTerminalTraceData {
    pub roles: Vec<FixedMultiAirCompleteTerminalAirRole>,
    pub traces: Vec<FixedMultiAirCompleteTerminalPartitionedTrace>,
    pub structured_target: EF,
    pub global_rho: EF,
    /// Transcript cursor consumed by the complete coefficient-two-coset WHIR
    /// prefix. It is also authenticated by `whir_prefix_cursor_bus`.
    pub whir_prefix_start_tidx: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedMultiAirCompleteTerminalTraceError {
    Profile,
    Prefix,
    Cursor,
    Sumcheck,
    Local { region: usize },
    Interaction { region: usize },
    BlockEq,
    Mapping,
    Layout,
    Order,
}

impl FixedMultiAirCompleteTerminalCircuit {
    pub fn generate_traces(
        &self,
        witness: FixedMultiAirCompleteTerminalTraceWitness<'_>,
    ) -> Result<FixedMultiAirCompleteTerminalTraceData, FixedMultiAirCompleteTerminalTraceError>
    {
        let p = &self.profile;
        let prefix_owner = FixedMultiAirCompletePrefixDecompositionAirs::new(
            p.prefix.clone(),
            self.proof_idx,
            self.buses.prefix(self.transcript_bus),
        );
        let prefix = prefix_owner
            .generate_traces(
                FixedMultiAirCompletePrefixWitness {
                    authenticated_root: witness.authenticated_root,
                    instance: witness.instance,
                    proof: witness.proof,
                    transcript: witness.transcript,
                    statement_start_tidx: witness.statement_start_tidx,
                },
                FixedMultiAirCompletePrefixRequiredHeights::default(),
            )
            .map_err(|_| FixedMultiAirCompleteTerminalTraceError::Prefix)?;
        let cursor_owner = FixedMultiAirCompleteRegionCursorOpeningAirs::new(
            p.cursor.clone(),
            self.buses.cursor(self.transcript_bus),
        )
        .map_err(|_| FixedMultiAirCompleteTerminalTraceError::Cursor)?;
        let cursor = cursor_owner
            .generate_traces(
                witness.proof,
                witness.transcript,
                prefix.decomposition_end_tidx,
                witness.instance.eta,
                prefix.beta_claim,
                FixedMultiAirCompleteRegionCursorOpeningRequiredHeights::minimal(
                    p.cursor.protocol_component_count(),
                ),
            )
            .map_err(|_| FixedMultiAirCompleteTerminalTraceError::Cursor)?;

        let log_constraints = p.prefix.log_constraints;
        let tau = witness
            .instance
            .beta
            .get(..log_constraints)
            .ok_or(FixedMultiAirCompleteTerminalTraceError::Profile)?;
        let global_point =
            generate_fixed_multi_air_complete_global_point_bridge_trace(p.global_point, tau, None)
                .map_err(|_| FixedMultiAirCompleteTerminalTraceError::BlockEq)?;
        let global_eq = p
            .global_eq
            .as_ref()
            .map(|profile| {
                generate_fixed_multi_air_complete_block_eq_trace(profile, tau, &[], None)
            })
            .transpose()
            .map_err(|_| FixedMultiAirCompleteTerminalTraceError::BlockEq)?;
        let global_weight = global_eq.as_ref().map_or(EF::ZERO, |trace| trace.weight);
        let mut packed_eq = FixedMultiAirCompletePackedBlockEqForestTraceBuilder::new(&p.packed_eq)
            .map_err(|_| FixedMultiAirCompleteTerminalTraceError::BlockEq)?;

        let mut traces = vec![
            from_prefix(prefix.instance),
            from_prefix(prefix.transcript_prefix),
            from_prefix(prefix.beta),
            from_prefix(prefix.decomposition),
            from_cursor(cursor.sequence.clone()),
            FixedMultiAirCompleteTerminalPartitionedTrace::cached(
                global_point.cached,
                global_point.common,
            ),
        ];
        let mut local_openings = vec![Vec::new(); p.plan.regions.len()];
        let mut local_points = vec![Vec::new(); p.plan.regions.len()];
        let mut interaction_openings = vec![None; p.plan.regions.len()];
        let mut interaction_points = vec![None; p.plan.regions.len()];
        let mut opening_traces = cursor.components.into_iter();

        for (component_ordinal, component) in p.cursor.components.iter().enumerate() {
            let boundary = cursor
                .boundaries
                .get(component_ordinal)
                .ok_or(FixedMultiAirCompleteTerminalTraceError::Order)?;
            if boundary.kind != component.kind || boundary.region != component.region {
                return Err(FixedMultiAirCompleteTerminalTraceError::Order);
            }
            let opening_trace = opening_traces
                .next()
                .ok_or(FixedMultiAirCompleteTerminalTraceError::Order)?;
            let point = fixed_multi_air_complete_region_sumcheck_point(
                witness.transcript,
                boundary.component_start_tidx,
                component.round_count,
            )
            .map_err(|_| FixedMultiAirCompleteTerminalTraceError::Sumcheck)?;
            let region = component.region;
            match component.kind {
                FixedMultiAirCompleteTerminalCircuitComponentKind::Local => {
                    let claim = *witness
                        .proof
                        .local_claims
                        .get(region)
                        .ok_or(FixedMultiAirCompleteTerminalTraceError::Local { region })?;
                    let proof = witness
                        .proof
                        .local_proofs
                        .get(region)
                        .ok_or(FixedMultiAirCompleteTerminalTraceError::Local { region })?;
                    let sumcheck_air = FixedMultiAirCompleteLocalRegionSumcheckAir::new(
                        self.transcript_bus,
                        self.buses.local_start,
                        self.buses.local_point,
                        self.buses.local_sumcheck_final,
                        region,
                        component.round_count,
                        component.opening_count,
                        component.point_fixed_source_count,
                        component.point_common_count,
                    )
                    .map_err(|_| FixedMultiAirCompleteTerminalTraceError::Sumcheck)?;
                    let sumcheck = generate_fixed_multi_air_complete_local_region_sumcheck_trace(
                        &sumcheck_air,
                        claim,
                        proof,
                        witness.transcript,
                        boundary.component_start_tidx,
                        None,
                    )
                    .map_err(|_| FixedMultiAirCompleteTerminalTraceError::Sumcheck)?;
                    let mut eq_weights = Vec::with_capacity(p.local_eq[region].len());
                    for profile in &p.local_eq[region] {
                        let trace = generate_fixed_multi_air_complete_block_eq_trace(
                            profile, tau, &point, None,
                        )
                        .map_err(|_| FixedMultiAirCompleteTerminalTraceError::BlockEq)?;
                        eq_weights.push(trace.weight);
                        packed_eq
                            .push(profile, &trace)
                            .map_err(|_| FixedMultiAirCompleteTerminalTraceError::BlockEq)?;
                    }
                    let endpoint = generate_fixed_multi_air_complete_local_endpoint_trace(
                        &p.locals[region],
                        &p.local_fixed[region],
                        &witness.instance.beta,
                        &point,
                        &proof.opened_columns,
                        &eq_weights,
                        None,
                        None,
                        None,
                        None,
                    )
                    .map_err(|_| FixedMultiAirCompleteTerminalTraceError::Local { region })?;
                    if endpoint.fold.final_claim != boundary.final_evaluation {
                        return Err(FixedMultiAirCompleteTerminalTraceError::Local { region });
                    }
                    traces.push(FixedMultiAirCompleteTerminalPartitionedTrace::simple(
                        sumcheck,
                    ));
                    traces.push(from_cursor(opening_trace));
                    traces.push(cached(endpoint.fixed.cached, endpoint.fixed.common));
                    traces.push(cached(endpoint.selector.cached, endpoint.selector.common));
                    traces.push(cached(endpoint.nodes.cached, endpoint.nodes.common));
                    traces.push(cached(endpoint.fold.cached, endpoint.fold.common));
                    local_openings[region] = proof.opened_columns.clone();
                    local_points[region] = point;
                }
                FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction => {
                    let claim =
                        *witness.proof.interaction_claims.get(region).ok_or(
                            FixedMultiAirCompleteTerminalTraceError::Interaction { region },
                        )?;
                    let proof = witness
                        .proof
                        .interaction_proofs
                        .get(region)
                        .and_then(Option::as_ref)
                        .ok_or(FixedMultiAirCompleteTerminalTraceError::Interaction { region })?;
                    let profile = p.interactions[region]
                        .as_ref()
                        .ok_or(FixedMultiAirCompleteTerminalTraceError::Interaction { region })?;
                    let fixed_setup = p.interaction_fixed[region]
                        .as_ref()
                        .ok_or(FixedMultiAirCompleteTerminalTraceError::Interaction { region })?;
                    let sumcheck_air = FixedMultiAirCompleteInteractionRegionSumcheckAir::new(
                        self.transcript_bus,
                        self.buses.interaction_start,
                        self.buses.interaction_point,
                        self.buses.interaction_sumcheck_final,
                        region,
                        component.round_count,
                        component.opening_count,
                        component.point_fixed_source_count,
                        component.point_common_count,
                    )
                    .map_err(|_| FixedMultiAirCompleteTerminalTraceError::Sumcheck)?;
                    let sumcheck =
                        generate_fixed_multi_air_complete_interaction_region_sumcheck_trace(
                            &sumcheck_air,
                            claim,
                            proof,
                            witness.transcript,
                            boundary.component_start_tidx,
                            None,
                        )
                        .map_err(|_| FixedMultiAirCompleteTerminalTraceError::Sumcheck)?;
                    let fixed = generate_fixed_multi_air_complete_interaction_fixed_trace(
                        profile,
                        fixed_setup.as_ref(),
                        &point,
                        None,
                    )
                    .map_err(|_| FixedMultiAirCompleteTerminalTraceError::Interaction { region })?;
                    let selector = generate_fixed_multi_air_complete_interaction_selector_trace(
                        profile, &point, None,
                    )
                    .map_err(|_| FixedMultiAirCompleteTerminalTraceError::Interaction { region })?;
                    let nodes = generate_fixed_multi_air_complete_interaction_node_trace(
                        profile,
                        &witness.instance.beta,
                        &proof.opened_columns,
                        &fixed.values,
                        selector.selectors,
                        None,
                    )
                    .map_err(|_| FixedMultiAirCompleteTerminalTraceError::Interaction { region })?;
                    let denominators =
                        generate_fixed_multi_air_complete_interaction_denominator_trace(
                            profile,
                            &witness.instance.beta,
                            &nodes.values,
                            None,
                        )
                        .map_err(|_| {
                            FixedMultiAirCompleteTerminalTraceError::Interaction { region }
                        })?;
                    let mut eq_weights = Vec::with_capacity(profile.interaction_count());
                    for eq in p.interaction_eq[region]
                        .as_ref()
                        .ok_or(FixedMultiAirCompleteTerminalTraceError::Interaction { region })?
                    {
                        let trace =
                            generate_fixed_multi_air_complete_block_eq_trace(eq, tau, &point, None)
                                .map_err(|_| FixedMultiAirCompleteTerminalTraceError::BlockEq)?;
                        eq_weights.push(trace.weight);
                        packed_eq
                            .push(eq, &trace)
                            .map_err(|_| FixedMultiAirCompleteTerminalTraceError::BlockEq)?;
                    }
                    let endpoint = generate_fixed_multi_air_complete_interaction_endpoint_trace(
                        profile,
                        &witness.instance.beta,
                        &proof.opened_columns,
                        &nodes.values,
                        &denominators.denominators,
                        &eq_weights,
                        global_weight,
                        boundary.final_evaluation,
                        None,
                    )
                    .map_err(|_| FixedMultiAirCompleteTerminalTraceError::Interaction { region })?;
                    traces.push(FixedMultiAirCompleteTerminalPartitionedTrace::simple(
                        sumcheck,
                    ));
                    traces.push(from_cursor(opening_trace));
                    traces.push(FixedMultiAirCompleteTerminalPartitionedTrace::simple(
                        fixed.common,
                    ));
                    traces.push(cached(selector.cached, selector.common));
                    traces.push(cached(nodes.cached, nodes.common));
                    traces.push(cached(denominators.cached, denominators.common));
                    traces.push(cached(endpoint.cached, endpoint.common));
                    interaction_openings[region] = Some(proof.opened_columns.clone());
                    interaction_points[region] = Some(point);
                }
            }
        }
        if opening_traces.next().is_some() {
            return Err(FixedMultiAirCompleteTerminalTraceError::Order);
        }
        match (p.global_eq.as_ref(), global_eq.as_ref()) {
            (Some(profile), Some(trace)) => packed_eq
                .push(profile, trace)
                .map_err(|_| FixedMultiAirCompleteTerminalTraceError::BlockEq)?,
            (None, None) => {}
            _ => return Err(FixedMultiAirCompleteTerminalTraceError::BlockEq),
        }
        let packed_eq = packed_eq
            .finish()
            .map_err(|_| FixedMultiAirCompleteTerminalTraceError::BlockEq)?;
        // AIR inventory places the packed table immediately after the global
        // point authority. Its LogUp messages remain order independent.
        traces.splice(
            6..6,
            packed_eq
                .into_iter()
                .map(|table| FixedMultiAirCompleteTerminalPartitionedTrace::simple(table.common)),
        );
        let mapping = generate_fixed_multi_air_complete_global_mapping_trace(
            &p.mapping,
            FixedMultiAirCompleteGlobalMappingWitness {
                local_openings: &local_openings,
                interaction_openings: &interaction_openings,
                local_points: &local_points,
                interaction_points: &interaction_points,
            },
            witness.transcript,
            cursor.batch_start_tidx,
            None,
            None,
        )
        .map_err(|_| FixedMultiAirCompleteTerminalTraceError::Mapping)?;
        let structured_target = mapping.target;
        let global_rho = mapping.rho;
        let whir_prefix_start_tidx = mapping.batch_end_tidx;
        traces.push(cached(mapping.mapping_cached, mapping.mapping_common));
        traces.push(cached(mapping.point_cached, mapping.point_common));
        let roles = self.air_roles();
        if traces.len() != roles.len() {
            return Err(FixedMultiAirCompleteTerminalTraceError::Order);
        }
        for trace in &mut traces {
            trace.inline_cached_mains()?;
        }
        Ok(FixedMultiAirCompleteTerminalTraceData {
            roles,
            traces,
            structured_target,
            global_rho,
            whir_prefix_start_tidx,
        })
    }
}

fn from_prefix(
    trace: FixedMultiAirCompletePartitionedTrace,
) -> FixedMultiAirCompleteTerminalPartitionedTrace {
    cached(trace.cached, trace.common)
}

fn from_cursor(
    trace: FixedMultiAirCompleteRegionCursorOpeningPartitionedTrace,
) -> FixedMultiAirCompleteTerminalPartitionedTrace {
    cached(trace.cached, trace.common)
}

fn cached(
    cached: RowMajorMatrix<F>,
    common: RowMajorMatrix<F>,
) -> FixedMultiAirCompleteTerminalPartitionedTrace {
    FixedMultiAirCompleteTerminalPartitionedTrace::cached(cached, common)
}
