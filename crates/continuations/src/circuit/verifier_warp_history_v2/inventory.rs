//! Canonical production AIR order for the History-v2 MultiSTARK.
//!
//! The schedule is data independent. Dynamic submodules contribute a fixed
//! number of AIRs at setup, and all singleton peripherals have stable ids.

use core::ops::Range;

use serde::{Deserialize, Serialize};

/// Maximum supported in-memory History-v2 batch profile. Capacity three needs
/// 143 transitions for the 429-segment production Reth benchmark. The
/// power-of-two bound is transcript/security charged explicitly by the WARP
/// executor.
pub const VERIFIER_WARP_HISTORY_MAX_TRANSITIONS_V2: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerifierWarpHistoryAirRoleV2 {
    /// Full same-message source/LogUp verifier, including transcript exports.
    SourceLogUp,
    /// Appendix-D mixed-alphabet VACC verifier and projection (C3).
    AppendixDVacc,
    FixedMultiAirSourceBoundary,
    LogUpProducer,
    WarpReplayProducer,
    /// Legacy setup-fixed opening evaluator. New production inventories use
    /// [`Self::SetupPcsAuthority`] instead.
    FixedSetupOpening,
    /// Contiguous verifier-owned setup-PCS authority bundle: source
    /// provenance, statement binding, ordered stacking, terminal WHIR, and
    /// completion/certificate adapters.
    SetupPcsAuthority,
    /// Same-message active-child-count functional.
    ActiveCountFunctional,
    /// Sole producer bridge writing authenticated History buses.
    ProducerBridge,
    UserPublicValuesCommit,
    BlockPublicValues,
    AggregateHistory,
    /// Sole physical Poseidon table for every transcript/compression owner.
    SharedPoseidon,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierWarpHistoryAirInventoryEntryV2 {
    pub role: VerifierWarpHistoryAirRoleV2,
    pub range: Range<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierWarpHistoryAirInventoryV2 {
    pub entries: Vec<VerifierWarpHistoryAirInventoryEntryV2>,
    pub history_air_id: usize,
    pub shared_poseidon_air_id: usize,
    pub total_air_count: usize,
}

impl VerifierWarpHistoryAirInventoryV2 {
    /// Legacy compatibility constructor using one fixed-setup opening AIR.
    /// New production callers should use
    /// [`Self::new_with_setup_pcs_authority`].
    pub fn new(source_air_count: usize, appendix_d_air_count: usize) -> Result<Self, &'static str> {
        Self::new_with_replay_producers(source_air_count, appendix_d_air_count, 1)
    }

    /// Legacy compatibility constructor with an explicit replay-producer
    /// count. Production direct-final profiles use one setup-fixed singleton
    /// VACC verifier and replay producer per transition so verifier trace
    /// heights remain independent of History length.
    pub fn new_with_replay_producers(
        source_air_count: usize,
        appendix_d_air_count: usize,
        replay_producer_count: usize,
    ) -> Result<Self, &'static str> {
        Self::new_with_modes(
            source_air_count,
            appendix_d_air_count,
            replay_producer_count,
            1,
        )
    }

    /// Build the canonical production order with one contiguous setup-PCS
    /// authority bundle.
    ///
    /// `setup_authority_air_count` is the complete number of AIRs owned by
    /// that bundle. It must be nonzero when the admitted setup contains fixed
    /// matrices and zero when there is no fixed setup to authenticate. The
    /// inventory deliberately does not infer setup presence from dynamic
    /// proof data; the setup-reconstructed caller owns that decision.
    pub fn new_with_setup_pcs_authority(
        source_air_count: usize,
        appendix_d_air_count: usize,
        replay_producer_count: usize,
        setup_authority_air_count: usize,
    ) -> Result<Self, &'static str> {
        if setup_authority_air_count == 0 {
            return Err("production setup-PCS authority bundle must be nonempty");
        }
        Self::build(
            source_air_count,
            appendix_d_air_count,
            replay_producer_count,
            VerifierWarpHistoryAirRoleV2::SetupPcsAuthority,
            setup_authority_air_count,
            true,
        )
    }

    /// Build the explicit no-fixed-setup production mode.
    ///
    /// Keeping this separate from [`Self::new_with_setup_pcs_authority`]
    /// prevents a caller that has setup commitments from silently selecting
    /// an empty authority range by passing zero as an ordinary count.
    pub fn new_without_setup_pcs_authority(
        source_air_count: usize,
        appendix_d_air_count: usize,
        replay_producer_count: usize,
    ) -> Result<Self, &'static str> {
        Self::build(
            source_air_count,
            appendix_d_air_count,
            replay_producer_count,
            VerifierWarpHistoryAirRoleV2::SetupPcsAuthority,
            0,
            true,
        )
    }

    /// Canonical bounded-History leaf inventory with a nonempty retained
    /// setup-PCS authority. Terminal block public values are deliberately
    /// omitted: the recursive root authenticates them once and binds them to
    /// the unique terminal leaf endpoint.
    pub fn new_chunk_with_setup_pcs_authority_v3(
        source_air_count: usize,
        appendix_d_air_count: usize,
        replay_producer_count: usize,
        setup_authority_air_count: usize,
    ) -> Result<Self, &'static str> {
        if setup_authority_air_count == 0 {
            return Err("chunk setup-PCS authority bundle must be nonempty");
        }
        Self::build(
            source_air_count,
            appendix_d_air_count,
            replay_producer_count,
            VerifierWarpHistoryAirRoleV2::SetupPcsAuthority,
            setup_authority_air_count,
            false,
        )
    }

    /// Canonical bounded-History leaf inventory for a relation without
    /// setup-owned matrices.
    pub fn new_chunk_without_setup_pcs_authority_v3(
        source_air_count: usize,
        appendix_d_air_count: usize,
        replay_producer_count: usize,
    ) -> Result<Self, &'static str> {
        Self::build(
            source_air_count,
            appendix_d_air_count,
            replay_producer_count,
            VerifierWarpHistoryAirRoleV2::SetupPcsAuthority,
            0,
            false,
        )
    }

    /// Legacy compatibility constructor. A count of zero means the admitted
    /// relation has no preprocessed matrices; one selects the old fixed-setup
    /// evaluator.
    pub fn new_with_modes(
        source_air_count: usize,
        appendix_d_air_count: usize,
        replay_producer_count: usize,
        fixed_setup_air_count: usize,
    ) -> Result<Self, &'static str> {
        if fixed_setup_air_count > 1 {
            return Err("invalid legacy fixed-setup AIR count");
        }
        Self::build(
            source_air_count,
            appendix_d_air_count,
            replay_producer_count,
            VerifierWarpHistoryAirRoleV2::FixedSetupOpening,
            fixed_setup_air_count,
            true,
        )
    }

    fn build(
        source_air_count: usize,
        appendix_d_air_count: usize,
        replay_producer_count: usize,
        setup_role: VerifierWarpHistoryAirRoleV2,
        setup_air_count: usize,
        include_terminal_public_values: bool,
    ) -> Result<Self, &'static str> {
        if source_air_count == 0
            || appendix_d_air_count == 0
            || !(1..=VERIFIER_WARP_HISTORY_MAX_TRANSITIONS_V2).contains(&replay_producer_count)
        {
            return Err("empty History-v2 verifier submodule");
        }
        if !matches!(
            setup_role,
            VerifierWarpHistoryAirRoleV2::FixedSetupOpening
                | VerifierWarpHistoryAirRoleV2::SetupPcsAuthority
        ) {
            return Err("invalid History-v2 setup authority role");
        }
        let mut entries = Vec::with_capacity(12);
        let mut cursor = 0usize;
        let mut push = |role, count: usize| -> Result<Range<usize>, &'static str> {
            let end = cursor
                .checked_add(count)
                .ok_or("History-v2 AIR inventory overflow")?;
            let range = cursor..end;
            entries.push(VerifierWarpHistoryAirInventoryEntryV2 {
                role,
                range: range.clone(),
            });
            cursor = end;
            Ok(range)
        };
        push(VerifierWarpHistoryAirRoleV2::SourceLogUp, source_air_count)?;
        push(
            VerifierWarpHistoryAirRoleV2::AppendixDVacc,
            appendix_d_air_count,
        )?;
        for (role, count) in [
            (VerifierWarpHistoryAirRoleV2::FixedMultiAirSourceBoundary, 1),
            (VerifierWarpHistoryAirRoleV2::LogUpProducer, 1),
            (
                VerifierWarpHistoryAirRoleV2::WarpReplayProducer,
                replay_producer_count,
            ),
            (setup_role, setup_air_count),
            (VerifierWarpHistoryAirRoleV2::ActiveCountFunctional, 1),
            (VerifierWarpHistoryAirRoleV2::ProducerBridge, 1),
            (
                VerifierWarpHistoryAirRoleV2::UserPublicValuesCommit,
                usize::from(include_terminal_public_values),
            ),
            (
                VerifierWarpHistoryAirRoleV2::BlockPublicValues,
                usize::from(include_terminal_public_values),
            ),
        ] {
            push(role, count)?;
        }
        let history_air_id = push(VerifierWarpHistoryAirRoleV2::AggregateHistory, 1)?.start;
        let shared_poseidon_air_id = push(VerifierWarpHistoryAirRoleV2::SharedPoseidon, 1)?.start;
        let inventory = Self {
            entries,
            history_air_id,
            shared_poseidon_air_id,
            total_air_count: cursor,
        };
        inventory.validate()?;
        Ok(inventory)
    }

    /// Range occupied by the production setup-PCS authority bundle. Legacy
    /// inventories return `None`, including their empty fixed-setup mode.
    pub fn setup_pcs_authority_range(&self) -> Option<Range<usize>> {
        self.entries
            .iter()
            .find(|entry| entry.role == VerifierWarpHistoryAirRoleV2::SetupPcsAuthority)
            .map(|entry| entry.range.clone())
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if self.entries.len() != 12 || self.total_air_count == 0 {
            return Err("invalid History-v2 AIR role count");
        }
        let setup_role = self.entries[5].role;
        if !matches!(
            setup_role,
            VerifierWarpHistoryAirRoleV2::FixedSetupOpening
                | VerifierWarpHistoryAirRoleV2::SetupPcsAuthority
        ) {
            return Err("invalid History-v2 setup authority role");
        }
        let expected_roles = [
            VerifierWarpHistoryAirRoleV2::SourceLogUp,
            VerifierWarpHistoryAirRoleV2::AppendixDVacc,
            VerifierWarpHistoryAirRoleV2::FixedMultiAirSourceBoundary,
            VerifierWarpHistoryAirRoleV2::LogUpProducer,
            VerifierWarpHistoryAirRoleV2::WarpReplayProducer,
            setup_role,
            VerifierWarpHistoryAirRoleV2::ActiveCountFunctional,
            VerifierWarpHistoryAirRoleV2::ProducerBridge,
            VerifierWarpHistoryAirRoleV2::UserPublicValuesCommit,
            VerifierWarpHistoryAirRoleV2::BlockPublicValues,
            VerifierWarpHistoryAirRoleV2::AggregateHistory,
            VerifierWarpHistoryAirRoleV2::SharedPoseidon,
        ];
        let mut cursor = 0usize;
        for (entry, expected_role) in self.entries.iter().zip(expected_roles) {
            let allowed_empty = entry.range.is_empty()
                && (entry.role == setup_role
                    || matches!(
                        entry.role,
                        VerifierWarpHistoryAirRoleV2::UserPublicValuesCommit
                            | VerifierWarpHistoryAirRoleV2::BlockPublicValues
                    ));
            if entry.role != expected_role
                || entry.range.start != cursor
                || entry.range.end < entry.range.start
                || (entry.range.is_empty() && !allowed_empty)
            {
                return Err("noncanonical History-v2 AIR range");
            }
            cursor = entry.range.end;
        }
        if self.entries[8].range.is_empty() != self.entries[9].range.is_empty() {
            return Err("partial History terminal public-value inventory");
        }
        let history_end = self
            .history_air_id
            .checked_add(1)
            .ok_or("History-v2 AIR inventory overflow")?;
        let shared_poseidon_end = self
            .shared_poseidon_air_id
            .checked_add(1)
            .ok_or("History-v2 AIR inventory overflow")?;
        if cursor != self.total_air_count
            || self.entries[10].range != (self.history_air_id..history_end)
            || self.entries[11].range != (self.shared_poseidon_air_id..shared_poseidon_end)
            || history_end != self.shared_poseidon_air_id
            || shared_poseidon_end != self.total_air_count
        {
            return Err("invalid History-v2 terminal AIR positions");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_inventory_is_dense_and_stable() {
        let inventory = VerifierWarpHistoryAirInventoryV2::new(31, 19).unwrap();
        assert_eq!(inventory.history_air_id, 58);
        assert_eq!(inventory.shared_poseidon_air_id, 59);
        assert_eq!(inventory.total_air_count, 60);
        inventory.validate().unwrap();
    }

    #[test]
    fn empty_c1_mode_is_canonical_and_shifts_terminal_ids() {
        let inventory = VerifierWarpHistoryAirInventoryV2::new_with_modes(31, 19, 1, 0).unwrap();
        let c1 = &inventory.entries[5];
        assert_eq!(c1.role, VerifierWarpHistoryAirRoleV2::FixedSetupOpening);
        assert!(c1.range.is_empty());
        assert_eq!(inventory.history_air_id, 57);
        assert_eq!(inventory.shared_poseidon_air_id, 58);
        assert_eq!(inventory.total_air_count, 59);
        inventory.validate().unwrap();
    }

    #[test]
    fn production_setup_pcs_authority_bundle_is_dense_and_stable() {
        let inventory =
            VerifierWarpHistoryAirInventoryV2::new_with_setup_pcs_authority(31, 19, 2, 17).unwrap();
        let authority = &inventory.entries[5];
        assert_eq!(
            authority.role,
            VerifierWarpHistoryAirRoleV2::SetupPcsAuthority
        );
        assert_eq!(authority.range, 54..71);
        assert_eq!(inventory.setup_pcs_authority_range(), Some(54..71));
        assert_eq!(inventory.history_air_id, 75);
        assert_eq!(inventory.shared_poseidon_air_id, 76);
        assert_eq!(inventory.total_air_count, 77);
        inventory.validate().unwrap();
    }

    #[test]
    fn production_setup_pcs_authority_bundle_cannot_be_empty() {
        assert_eq!(
            VerifierWarpHistoryAirInventoryV2::new_with_setup_pcs_authority(31, 19, 2, 0),
            Err("production setup-PCS authority bundle must be nonempty")
        );
    }

    #[test]
    fn production_empty_setup_authority_mode_is_canonical() {
        let inventory =
            VerifierWarpHistoryAirInventoryV2::new_without_setup_pcs_authority(31, 19, 1).unwrap();
        assert_eq!(inventory.setup_pcs_authority_range(), Some(53..53));
        assert_eq!(inventory.history_air_id, 57);
        assert_eq!(inventory.shared_poseidon_air_id, 58);
        assert_eq!(inventory.total_air_count, 59);
        inventory.validate().unwrap();
    }

    #[test]
    fn production_inventory_rejects_overflow() {
        assert_eq!(
            VerifierWarpHistoryAirInventoryV2::new_with_setup_pcs_authority(usize::MAX, 1, 1, 1,),
            Err("History-v2 AIR inventory overflow")
        );
        assert_eq!(
            VerifierWarpHistoryAirInventoryV2::new_with_setup_pcs_authority(1, 1, 1, usize::MAX,),
            Err("History-v2 AIR inventory overflow")
        );
    }

    #[test]
    fn validation_rejects_noncanonical_order_and_terminal_ids() {
        let inventory =
            VerifierWarpHistoryAirInventoryV2::new_with_setup_pcs_authority(2, 3, 1, 4).unwrap();

        let mut reordered = inventory.clone();
        reordered.entries.swap(6, 7);
        assert_eq!(
            reordered.validate(),
            Err("noncanonical History-v2 AIR range")
        );

        let mut terminal_alias = inventory;
        terminal_alias.shared_poseidon_air_id = terminal_alias.history_air_id;
        assert_eq!(
            terminal_alias.validate(),
            Err("invalid History-v2 terminal AIR positions")
        );
    }
}
