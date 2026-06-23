use ark_bls12_381::{Bls12_381, Fq, Fq12, Fq2, Fr, G1Affine, G1Projective, G2Affine, G2Projective};
use ark_ec::{pairing::Pairing, pairing::PairingOutput, AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInteger, One, PrimeField, Zero};
use std::os::raw::{c_int, c_uchar, c_uint};

#[repr(C)]
pub enum Bls12381Result {
    Success = 0,
    InvalidInput = 1,
    InvalidPoint = 2,
    InvalidScalar = 3,
    ComputationFailed = 4,
}

fn parse_g1(bytes: &[u8]) -> Result<G1Affine, Bls12381Result> {
    if bytes.len() != 96 {
        return Err(Bls12381Result::InvalidInput);
    }

    let x = Fq::from_be_bytes_mod_order(&bytes[0..48]);
    let y = Fq::from_be_bytes_mod_order(&bytes[48..96]);
    if x.is_zero() && y.is_zero() {
        return Ok(G1Affine::zero());
    }

    let point = G1Affine::new_unchecked(x, y);
    if point.is_on_curve() && point.is_in_correct_subgroup_assuming_on_curve() {
        Ok(point)
    } else {
        Err(Bls12381Result::InvalidPoint)
    }
}

fn parse_g2(bytes: &[u8]) -> Result<G2Affine, Bls12381Result> {
    if bytes.len() != 192 {
        return Err(Bls12381Result::InvalidInput);
    }

    let x_c0 = Fq::from_be_bytes_mod_order(&bytes[0..48]);
    let x_c1 = Fq::from_be_bytes_mod_order(&bytes[48..96]);
    let y_c0 = Fq::from_be_bytes_mod_order(&bytes[96..144]);
    let y_c1 = Fq::from_be_bytes_mod_order(&bytes[144..192]);
    let x = Fq2::new(x_c0, x_c1);
    let y = Fq2::new(y_c0, y_c1);
    if x.is_zero() && y.is_zero() {
        return Ok(G2Affine::zero());
    }

    let point = G2Affine::new_unchecked(x, y);
    if point.is_on_curve() && point.is_in_correct_subgroup_assuming_on_curve() {
        Ok(point)
    } else {
        Err(Bls12381Result::InvalidPoint)
    }
}

fn parse_scalar(bytes: &[u8]) -> Result<Fr, Bls12381Result> {
    if bytes.len() != 32 {
        return Err(Bls12381Result::InvalidInput);
    }
    Ok(Fr::from_be_bytes_mod_order(bytes))
}

fn write_g1(output: &mut [u8], point: G1Affine) {
    output[..96].fill(0);
    if point.is_zero() {
        return;
    }

    let x_bytes = point.x().expect("affine x").into_bigint().to_bytes_be();
    let y_bytes = point.y().expect("affine y").into_bigint().to_bytes_be();
    output[48 - x_bytes.len()..48].copy_from_slice(&x_bytes);
    output[96 - y_bytes.len()..96].copy_from_slice(&y_bytes);
}

fn write_g2(output: &mut [u8], point: G2Affine) {
    output[..192].fill(0);
    if point.is_zero() {
        return;
    }

    let x = point.x().expect("affine x");
    let y = point.y().expect("affine y");
    let x_c0 = x.c0.into_bigint().to_bytes_be();
    let x_c1 = x.c1.into_bigint().to_bytes_be();
    let y_c0 = y.c0.into_bigint().to_bytes_be();
    let y_c1 = y.c1.into_bigint().to_bytes_be();
    output[48 - x_c0.len()..48].copy_from_slice(&x_c0);
    output[96 - x_c1.len()..96].copy_from_slice(&x_c1);
    output[144 - y_c0.len()..144].copy_from_slice(&y_c0);
    output[192 - y_c1.len()..192].copy_from_slice(&y_c1);
}

unsafe fn input_slice<'a>(input: *const c_uchar, input_len: c_uint) -> Result<&'a [u8], Bls12381Result> {
    if input.is_null() {
        return Err(Bls12381Result::InvalidInput);
    }
    Ok(std::slice::from_raw_parts(input, input_len as usize))
}

unsafe fn output_slice<'a>(output: *mut c_uchar, output_len: c_uint) -> Result<&'a mut [u8], Bls12381Result> {
    if output.is_null() {
        return Err(Bls12381Result::InvalidInput);
    }
    Ok(std::slice::from_raw_parts_mut(output, output_len as usize))
}

fn code(result: Result<(), Bls12381Result>) -> c_int {
    match result {
        Ok(()) => Bls12381Result::Success as c_int,
        Err(err) => err as c_int,
    }
}

#[no_mangle]
pub unsafe extern "C" fn ora_bls12_381_g1_add(
    input: *const c_uchar,
    input_len: c_uint,
    output: *mut c_uchar,
    output_len: c_uint,
) -> c_int {
    code((|| {
        let input = unsafe { input_slice(input, input_len)? };
        let output = unsafe { output_slice(output, output_len)? };
        if input.len() != 192 || output.len() < 96 {
            return Err(Bls12381Result::InvalidInput);
        }

        let p1 = parse_g1(&input[0..96])?;
        let p2 = parse_g1(&input[96..192])?;
        write_g1(output, (p1 + p2).into_affine());
        Ok(())
    })())
}

#[no_mangle]
pub unsafe extern "C" fn ora_bls12_381_g1_mul(
    input: *const c_uchar,
    input_len: c_uint,
    output: *mut c_uchar,
    output_len: c_uint,
) -> c_int {
    code((|| {
        let input = unsafe { input_slice(input, input_len)? };
        let output = unsafe { output_slice(output, output_len)? };
        if input.len() != 128 || output.len() < 96 {
            return Err(Bls12381Result::InvalidInput);
        }

        let point = parse_g1(&input[0..96])?;
        let scalar = parse_scalar(&input[96..128])?;
        write_g1(output, (point * scalar).into_affine());
        Ok(())
    })())
}

#[no_mangle]
pub unsafe extern "C" fn ora_bls12_381_g1_msm(
    input: *const c_uchar,
    input_len: c_uint,
    output: *mut c_uchar,
    output_len: c_uint,
) -> c_int {
    code((|| {
        let input = unsafe { input_slice(input, input_len)? };
        let output = unsafe { output_slice(output, output_len)? };
        if input.is_empty() || input.len() % 128 != 0 || output.len() < 96 {
            return Err(Bls12381Result::InvalidInput);
        }

        let count = input.len() / 128;
        let mut points = Vec::with_capacity(count);
        let mut scalars = Vec::with_capacity(count);
        for i in 0..count {
            let offset = i * 128;
            let point = parse_g1(&input[offset..offset + 96])?;
            let scalar = parse_scalar(&input[offset + 96..offset + 128])?;
            if !point.is_zero() {
                points.push(point);
                scalars.push(scalar);
            }
        }

        let result = if points.is_empty() {
            G1Affine::zero()
        } else {
            G1Projective::msm(&points, &scalars)
                .map_err(|_| Bls12381Result::ComputationFailed)?
                .into_affine()
        };
        write_g1(output, result);
        Ok(())
    })())
}

#[no_mangle]
pub unsafe extern "C" fn ora_bls12_381_g2_add(
    input: *const c_uchar,
    input_len: c_uint,
    output: *mut c_uchar,
    output_len: c_uint,
) -> c_int {
    code((|| {
        let input = unsafe { input_slice(input, input_len)? };
        let output = unsafe { output_slice(output, output_len)? };
        if input.len() != 384 || output.len() < 192 {
            return Err(Bls12381Result::InvalidInput);
        }

        let p1 = parse_g2(&input[0..192])?;
        let p2 = parse_g2(&input[192..384])?;
        write_g2(output, (p1 + p2).into_affine());
        Ok(())
    })())
}

#[no_mangle]
pub unsafe extern "C" fn ora_bls12_381_g2_mul(
    input: *const c_uchar,
    input_len: c_uint,
    output: *mut c_uchar,
    output_len: c_uint,
) -> c_int {
    code((|| {
        let input = unsafe { input_slice(input, input_len)? };
        let output = unsafe { output_slice(output, output_len)? };
        if input.len() != 224 || output.len() < 192 {
            return Err(Bls12381Result::InvalidInput);
        }

        let point = parse_g2(&input[0..192])?;
        let scalar = parse_scalar(&input[192..224])?;
        write_g2(output, (point * scalar).into_affine());
        Ok(())
    })())
}

#[no_mangle]
pub unsafe extern "C" fn ora_bls12_381_g2_msm(
    input: *const c_uchar,
    input_len: c_uint,
    output: *mut c_uchar,
    output_len: c_uint,
) -> c_int {
    code((|| {
        let input = unsafe { input_slice(input, input_len)? };
        let output = unsafe { output_slice(output, output_len)? };
        if input.is_empty() || input.len() % 224 != 0 || output.len() < 192 {
            return Err(Bls12381Result::InvalidInput);
        }

        let count = input.len() / 224;
        let mut points = Vec::with_capacity(count);
        let mut scalars = Vec::with_capacity(count);
        for i in 0..count {
            let offset = i * 224;
            let point = parse_g2(&input[offset..offset + 192])?;
            let scalar = parse_scalar(&input[offset + 192..offset + 224])?;
            if !point.is_zero() {
                points.push(point);
                scalars.push(scalar);
            }
        }

        let result = if points.is_empty() {
            G2Affine::zero()
        } else {
            G2Projective::msm(&points, &scalars)
                .map_err(|_| Bls12381Result::ComputationFailed)?
                .into_affine()
        };
        write_g2(output, result);
        Ok(())
    })())
}

#[no_mangle]
pub unsafe extern "C" fn ora_bls12_381_pairing(
    input: *const c_uchar,
    input_len: c_uint,
    output: *mut c_uchar,
    output_len: c_uint,
) -> c_int {
    code((|| {
        let input = unsafe { input_slice(input, input_len)? };
        let output = unsafe { output_slice(output, output_len)? };
        if input.len() % 288 != 0 || output.len() < 32 {
            return Err(Bls12381Result::InvalidInput);
        }

        output[..32].fill(0);
        if input.is_empty() {
            output[31] = 1;
            return Ok(());
        }

        let count = input.len() / 288;
        let mut g1_points = Vec::with_capacity(count);
        let mut g2_points = Vec::with_capacity(count);
        for i in 0..count {
            let offset = i * 288;
            g1_points.push(parse_g1(&input[offset..offset + 96])?);
            g2_points.push(parse_g2(&input[offset + 96..offset + 288])?);
        }

        let result = Bls12_381::multi_pairing(&g1_points, &g2_points);
        let identity = PairingOutput::<Bls12_381>(Fq12::one());
        if result == identity {
            output[31] = 1;
        }
        Ok(())
    })())
}
