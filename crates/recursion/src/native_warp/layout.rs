use openvm_stark_backend::native_warp::NativeWarpFamilyParams;

/// Collision-free identifiers for vector and equality-evaluation lookup
/// tables. This layout is derived only from fixed family parameters and is
/// therefore identical in bootstrap and recursive transition circuits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeWarpAlgebraLayout {
    pub selector_tau_vector: u32,
    pub gamma_vector: u32,
    pub xi_vector: u32,
    pub alpha_vector: u32,
    pub input_index_vectors: Vec<u32>,
    pub claim_index_vectors: Vec<u32>,
    pub opening_point_vectors: Vec<u32>,
    pub selector_weight_groups: Vec<u32>,
    pub gamma_weight_groups: Vec<u32>,
    pub selector_at_gamma_group: u32,
    pub xi_weight_groups: Vec<u32>,
    pub opening_at_alpha_groups: Vec<u32>,
    pub eq_group_count: usize,
}

impl NativeWarpAlgebraLayout {
    #[must_use]
    pub fn new(family: &NativeWarpFamilyParams) -> Self {
        let claim_count = Self::claim_count(family);
        let mut next_vector = 0u32;
        let mut take_vector = || {
            let value = next_vector;
            next_vector += 1;
            value
        };
        let selector_tau_vector = take_vector();
        let gamma_vector = take_vector();
        let xi_vector = take_vector();
        let alpha_vector = take_vector();
        let input_index_vectors = (0..family.input_arity)
            .map(|_| take_vector())
            .collect::<Vec<_>>();
        let claim_index_vectors = (0..claim_count).map(|_| take_vector()).collect::<Vec<_>>();
        let opening_point_vectors = (0..claim_count).map(|_| take_vector()).collect::<Vec<_>>();

        let selector_weight_groups = (0..family.input_arity as u32).collect::<Vec<_>>();
        let gamma_start = family.input_arity as u32;
        let gamma_weight_groups =
            (gamma_start..gamma_start + family.input_arity as u32).collect::<Vec<_>>();
        let mut next_group = gamma_start + family.input_arity as u32;
        let selector_at_gamma_group = next_group;
        next_group += 1;
        let xi_weight_groups = (next_group..next_group + claim_count as u32).collect::<Vec<_>>();
        next_group += claim_count as u32;
        let opening_at_alpha_groups =
            (next_group..next_group + claim_count as u32).collect::<Vec<_>>();
        next_group += claim_count as u32;
        Self {
            selector_tau_vector,
            gamma_vector,
            xi_vector,
            alpha_vector,
            input_index_vectors,
            claim_index_vectors,
            opening_point_vectors,
            selector_weight_groups,
            gamma_weight_groups,
            selector_at_gamma_group,
            xi_weight_groups,
            opening_at_alpha_groups,
            eq_group_count: next_group as usize,
        }
    }

    #[must_use]
    pub fn claim_count(family: &NativeWarpFamilyParams) -> usize {
        (1 + family.num_ood + family.num_shift_queries).next_power_of_two()
    }
}
