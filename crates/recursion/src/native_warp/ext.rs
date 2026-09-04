//! Native-WARP extension arithmetic represented inside the recursive AIR.
//!
//! The recursive STARK itself still uses its normal degree-four challenge
//! field. These helpers constrain native-WARP EF values as `D_EF` BabyBear
//! limbs, using the configured binomial relation `X^D_EF = W`.

use openvm_stark_sdk::config::baby_bear_poseidon2::D_EF;
use p3_field::{extension::BinomiallyExtendable, PrimeCharacteristicRing};

pub fn ext_field_add<FA>(x: [impl Into<FA>; D_EF], y: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
{
    let x = x.map(Into::into);
    let y = y.map(Into::into);
    core::array::from_fn(|i| x[i].clone() + y[i].clone())
}

pub fn ext_field_subtract<FA>(x: [impl Into<FA>; D_EF], y: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
{
    let x = x.map(Into::into);
    let y = y.map(Into::into);
    core::array::from_fn(|i| x[i].clone() - y[i].clone())
}

pub fn ext_field_multiply_scalar<FA>(x: [impl Into<FA>; D_EF], scalar: impl Into<FA>) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
{
    let x = x.map(Into::into);
    let scalar = scalar.into();
    core::array::from_fn(|i| x[i].clone() * scalar.clone())
}

pub fn ext_field_multiply<FA>(x: [impl Into<FA>; D_EF], y: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
    FA::PrimeSubfield: BinomiallyExtendable<D_EF>,
{
    let x = x.map(Into::into);
    let y = y.map(Into::into);
    let w = FA::from_prime_subfield(FA::PrimeSubfield::W);
    let mut output = core::array::from_fn(|_| FA::ZERO);
    for (i, left) in x.iter().enumerate() {
        for (j, right) in y.iter().enumerate() {
            let degree = i + j;
            let coordinate = degree % D_EF;
            let mut term = left.clone() * right.clone();
            if degree >= D_EF {
                term *= w.clone();
            }
            output[coordinate] = output[coordinate].clone() + term;
        }
    }
    output
}

pub fn eq_1<FA>(x: [impl Into<FA>; D_EF], y: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
    FA::PrimeSubfield: BinomiallyExtendable<D_EF>,
{
    let x = x.map(Into::into);
    let y = y.map(Into::into);
    let xy: [FA; D_EF] = ext_field_multiply(x.clone(), y.clone());
    core::array::from_fn(|i| {
        let one = if i == 0 { FA::ONE } else { FA::ZERO };
        one - x[i].clone() - y[i].clone() + xy[i].clone() * FA::TWO
    })
}

#[cfg(test)]
mod tests {
    use openvm_stark_sdk::config::baby_bear_poseidon2::{EF, F};
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

    use super::*;

    #[test]
    fn limb_arithmetic_matches_configured_extension() {
        let x_coefficients: [F; D_EF] = core::array::from_fn(|i| F::from_usize(2 * i + 3));
        let y_coefficients: [F; D_EF] = core::array::from_fn(|i| F::from_usize(3 * i + 11));
        let x = EF::from_basis_coefficients_slice(&x_coefficients)
            .expect("configured extension coefficients");
        let y = EF::from_basis_coefficients_slice(&y_coefficients)
            .expect("configured extension coefficients");
        let x_coefficients: [F; D_EF] = x
            .as_basis_coefficients_slice()
            .try_into()
            .expect("EF basis width");
        let y_coefficients: [F; D_EF] = y
            .as_basis_coefficients_slice()
            .try_into()
            .expect("EF basis width");
        let product: [F; D_EF] = ext_field_multiply(x_coefficients, y_coefficients);
        let expected_product = x * y;
        assert_eq!(
            product.as_slice(),
            <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(&expected_product)
        );
        let equality: [F; D_EF] = eq_1(x_coefficients, y_coefficients);
        let expected_equality = EF::ONE - x - y + EF::TWO * x * y;
        assert_eq!(
            equality.as_slice(),
            <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(&expected_equality)
        );
    }
}
