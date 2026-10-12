//! Spectra sampled 380–780 nm every 5 nm (81 samples): the grid of the stock data.

use crate::cie_table::{CMF_X, CMF_Y, CMF_Z, DAYLIGHT_S0, DAYLIGHT_S1, DAYLIGHT_S2};

pub const N: usize = 81;
pub const WL0: f64 = 380.0;
pub const STEP: f64 = 5.0;
pub type Spectrum = [f64; N];

/// Index of 560 nm, where illuminants are normalised to 1.
const I560: usize = 36;

pub fn wavelength(i: usize) -> f64 {
    WL0 + STEP * i as f64
}

pub fn dot(a: &Spectrum, b: &Spectrum) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn mul(a: &Spectrum, b: &Spectrum) -> Spectrum {
    std::array::from_fn(|i| a[i] * b[i])
}

/// Unnormalised CIE 1931 XYZ of a spectral power distribution.
pub fn xyz(s: &Spectrum) -> [f64; 3] {
    [dot(s, &CMF_X), dot(s, &CMF_Y), dot(s, &CMF_Z)]
}

/// Chromaticity of `s`; `None` for a black or non-finite spectrum.
pub fn xy(s: &Spectrum) -> Option<(f64, f64)> {
    let [x, y, z] = xyz(s);
    let t = x + y + z;
    (t.is_finite() && t > 0.0).then(|| (x / t, y / t))
}

/// Planck's law at `kelvin` (clamped to 1000–25000 K; non-finite → 6500 K), normalised to 1 at 560 nm.
pub fn planck(kelvin: f64) -> Spectrum {
    const C2: f64 = 1.4388e-2; // second radiation constant, m·K
    let t = if kelvin.is_finite() { kelvin.clamp(1000.0, 25000.0) } else { 6500.0 };
    let m = |nm: f64| {
        let l = nm * 1e-9;
        l.powi(-5) / ((C2 / (l * t)).exp() - 1.0)
    };
    let k = m(wavelength(I560));
    std::array::from_fn(|i| m(wavelength(i)) / k)
}

/// CIE daylight at correlated colour temperature `cct` (clamped to 4000–25000 K; non-finite → 6504 K), CIE 15
/// method with M1 and M2 rounded to three decimals as the standard prescribes, normalised to 1 at 560 nm.
pub fn daylight(cct: f64) -> Spectrum {
    let t = if cct.is_finite() { cct.clamp(4000.0, 25000.0) } else { 6504.0 };
    let x = if t <= 7000.0 {
        -4.6070e9 / t.powi(3) + 2.9678e6 / t.powi(2) + 0.09911e3 / t + 0.244063
    } else {
        -2.0064e9 / t.powi(3) + 1.9018e6 / t.powi(2) + 0.24748e3 / t + 0.237040
    };
    let y = -3.0 * x * x + 2.870 * x - 0.275;
    let m = 0.0241 + 0.2562 * x - 0.7341 * y;
    let round3 = |v: f64| (v * 1000.0).round() / 1000.0;
    let m1 = round3((-1.3515 - 1.7703 * x + 5.9114 * y) / m);
    let m2 = round3((0.0300 - 31.4424 * x + 30.0717 * y) / m);
    let s: Spectrum = std::array::from_fn(|i| DAYLIGHT_S0[i] + m1 * DAYLIGHT_S1[i] + m2 * DAYLIGHT_S2[i]);
    let k = s[I560];
    s.map(|v| v / k)
}

/// The light a stock's data name: CIE `D50`/`D55`/`D65`/`D75` (5003/5503/6504/7504 K, the CIE's ×1.4388/1.4380
/// temperatures), anything else (tungsten enlarger lamps such as `TH-KG3`) our [`enlarger_lamp`].
pub fn illuminant(name: &str) -> Spectrum {
    match name {
        "D50" => daylight(5003.0),
        "D55" => daylight(5503.0),
        "D65" => daylight(6504.0),
        "D75" => daylight(7504.0),
        _ => enlarger_lamp(),
    }
}

/// Our enlarger lamp: a 3200 K tungsten source with a smooth heat-absorbing roll-off above 700 nm. Our own
/// approximation, not measured filter data.
pub fn enlarger_lamp() -> Spectrum {
    let p = planck(3200.0);
    std::array::from_fn(|i| p[i] / (1.0 + ((wavelength(i) - 700.0) / 12.0).exp()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: (f64, f64), b: (f64, f64), tol: f64) -> bool {
        (a.0 - b.0).abs() < tol && (a.1 - b.1).abs() < tol
    }

    #[test]
    fn grid_is_380_to_780_nm() {
        assert_eq!(wavelength(0), 380.0);
        assert_eq!(wavelength(I560), 560.0);
        assert_eq!(wavelength(N - 1), 780.0);
    }

    #[test]
    fn cmf_y_peaks_at_555_nm() {
        let i = (0..N).max_by(|&a, &b| CMF_Y[a].total_cmp(&CMF_Y[b])).unwrap();
        assert_eq!(wavelength(i), 555.0);
        assert!((CMF_Y[i] - 1.0).abs() < 0.01, "{}", CMF_Y[i]);
    }

    #[test]
    fn daylight_has_the_cie_chromaticities() {
        assert!(near(xy(&daylight(6504.0)).unwrap(), (0.3127, 0.3290), 0.001), "D65 {:?}", xy(&daylight(6504.0)));
        assert!(near(xy(&daylight(5003.0)).unwrap(), (0.3457, 0.3585), 0.001), "D50 {:?}", xy(&daylight(5003.0)));
    }

    #[test]
    fn planck_2856_is_illuminant_a() {
        assert!(near(xy(&planck(2856.0)).unwrap(), (0.4476, 0.4074), 0.001), "{:?}", xy(&planck(2856.0)));
    }

    #[test]
    fn enlarger_lamp_is_warm_and_cut_in_the_infrared() {
        let l = enlarger_lamp();
        assert!(l[N - 1] < 0.05 * l[I560 + 20]);
        assert!(xy(&l).unwrap().0 > 0.40);
    }

    #[test]
    fn hostile_temperatures_give_finite_spectra() {
        for t in [f64::NAN, -1.0, 0.0, 1e12, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(planck(t).iter().all(|v| v.is_finite()), "planck {t}");
            assert!(daylight(t).iter().all(|v| v.is_finite()), "daylight {t}");
        }
        assert!(xy(&[0.0; N]).is_none());
    }
}
