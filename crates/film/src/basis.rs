//! Spectral upsampling: three smooth reflectance spectra b_r, b_g, b_b that sum to 1 and whose colours under D65
//! are the Rec.2020 primaries, so a linear Rec.2020 colour (r, g, b) is the reflectance r·b_r + g·b_g + b·b_b.
//! Linear, so a film's exposure is a 3×3 matrix of the colour. Chosen as the smoothest such set: minimise
//! ½(‖D b_r‖² + ‖D b_g‖² + ‖D b_b‖²) (D = first differences) with b_b = 1 − b_r − b_g, subject to the six
//! colour constraints on b_r and b_g, plus a tiny pull towards grey that keeps the system definite. Solved once
//! (KKT system, 168 unknowns, Gauss–Jordan elimination).

use std::sync::OnceLock;

use lightcraft_color::REC2020;

use crate::cie_table::{CMF_X, CMF_Y, CMF_Z};
use crate::spectrum::{N, Spectrum, daylight, dot};

/// Weight of the pull towards grey (1/3).
const EPS: f64 = 1e-4;

/// The basis; a flat grey basis if the solve ever failed (the tests pin that it doesn't).
pub(crate) fn rec2020() -> &'static [Spectrum; 3] {
    static B: OnceLock<[Spectrum; 3]> = OnceLock::new();
    B.get_or_init(|| solve().unwrap_or([[1.0 / 3.0; N]; 3]))
}

fn solve() -> Option<[Spectrum; 3]> {
    let d65 = daylight(6504.0);
    let norm = dot(&d65, &CMF_Y);
    let cmf = [&CMF_X, &CMF_Y, &CMF_Z];
    // w[c][i]: contribution of reflectance at sample i to X, Y, Z (c) under D65, white Y = 1
    let w: [Spectrum; 3] = std::array::from_fn(|c| std::array::from_fn(|i| d65[i] * cmf[c][i] / norm));
    let m = REC2020.to_xyz().0;
    let n = 2 * N;
    let size = n + 6;
    let mut a = vec![vec![0.0f64; size + 1]; size];
    // H = [[2L, L], [L, 2L]] + EPS·I, L = DᵀD (tridiagonal: 1 or 2 on the diagonal, −1 beside it)
    for (br, bc, f) in [(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)] {
        for i in 0..N {
            let diag = if i == 0 || i == N - 1 { 1.0 } else { 2.0 };
            let (r, c) = (br * N + i, bc * N + i);
            a[r][c] += f * diag;
            if i > 0 {
                a[r][c - 1] -= f;
            }
            if i + 1 < N {
                a[r][c + 1] -= f;
            }
        }
    }
    for (i, row) in a.iter_mut().enumerate().take(n) {
        row[i] += EPS;
        row[size] = EPS / 3.0;
    }
    // constraints: Σ_i w[c][i]·b_k[i] = m[c][k] for k = r, g
    for k in 0..2 {
        for c in 0..3 {
            let r = n + 3 * k + c;
            for i in 0..N {
                a[r][k * N + i] = w[c][i];
                a[k * N + i][r] = w[c][i];
            }
            a[r][size] = m[c][k];
        }
    }
    let x = gauss_jordan(a)?;
    let br: Spectrum = std::array::from_fn(|i| x[i]);
    let bg: Spectrum = std::array::from_fn(|i| x[N + i]);
    let bb: Spectrum = std::array::from_fn(|i| 1.0 - br[i] - bg[i]);
    let out = [br, bg, bb];
    out.iter().all(|s| s.iter().all(|v| v.is_finite())).then_some(out)
}

/// Solve the augmented system `a` (n rows, n + 1 columns) with partial pivoting; `None` if singular.
pub(crate) fn gauss_jordan(mut a: Vec<Vec<f64>>) -> Option<Vec<f64>> {
    let n = a.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-14 {
            return None;
        }
        a.swap(col, piv);
        let pivot = a[col].clone();
        for (r, row) in a.iter_mut().enumerate() {
            if r == col {
                continue;
            }
            let f = row[col] / pivot[col];
            if f != 0.0 {
                for (v, p) in row.iter_mut().zip(&pivot).skip(col) {
                    *v -= f * p;
                }
            }
        }
    }
    Some(a.iter().enumerate().map(|(i, row)| row[n] / row[i]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectrum::{mul, xyz};

    #[test]
    fn the_kkt_system_solves() {
        assert!(solve().is_some());
    }

    #[test]
    fn basis_sums_to_a_flat_white() {
        let b = rec2020();
        for i in 0..N {
            assert!((b[0][i] + b[1][i] + b[2][i] - 1.0).abs() < 1e-9, "{i}");
        }
    }

    #[test]
    fn basis_reproduces_the_rec2020_primaries_under_d65() {
        let b = rec2020();
        let d65 = daylight(6504.0);
        let yw = xyz(&d65)[1];
        let m = REC2020.to_xyz().0;
        for (k, tol) in [(0, 1e-6), (1, 1e-6), (2, 2e-3)] {
            let c = xyz(&mul(&d65, &b[k])).map(|v| v / yw);
            for r in 0..3 {
                assert!((c[r] - m[r][k]).abs() < tol, "primary {k} row {r}: {} vs {}", c[r], m[r][k]);
            }
        }
    }

    #[test]
    fn basis_is_finite_and_bounded() {
        for s in rec2020() {
            assert!(s.iter().all(|v| v.is_finite() && v.abs() < 5.0));
        }
    }
}
