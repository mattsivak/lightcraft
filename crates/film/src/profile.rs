//! Film and paper stock data (spektrafilm profile JSON: `info` + `data`), validated on load. Missing samples
//! (`null`) in dye and base spectra are zero density; anything else malformed is an error.

use serde::{Deserialize, Serialize};

use crate::FilmError;
use crate::spectrum::{N, Spectrum, wavelength};

/// Largest profile accepted (the shipped ones are ~210 KB).
const MAX_JSON: usize = 4 << 20;
/// Largest |density| or |log sensitivity| accepted.
const MAX_VALUE: f64 = 40.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// Colour negative film: printed onto a paper.
    Negative,
    /// Slide (reversal) film: viewed directly.
    Positive,
    /// Print paper.
    Paper,
}

/// A characteristic curve: density above base against log10 exposure (`x` strictly increasing).
#[derive(Clone, Debug)]
pub struct Curve {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

impl Curve {
    /// Linear interpolation, clamped to the end values; NaN reads as the start.
    pub fn eval(&self, x: f64) -> f64 {
        let (Some(&x0), Some(&x1), Some(&y0), Some(&y1)) = (self.x.first(), self.x.last(), self.y.first(), self.y.last()) else {
            return 0.0;
        };
        if x.is_nan() || x <= x0 {
            return y0;
        }
        if x >= x1 {
            return y1;
        }
        let i = self.x.partition_point(|&v| v <= x).max(1);
        match (self.x.get(i - 1), self.x.get(i), self.y.get(i - 1), self.y.get(i)) {
            (Some(&xa), Some(&xb), Some(&ya), Some(&yb)) if xb > xa => ya + (yb - ya) * (x - xa) / (xb - xa),
            _ => y1,
        }
    }

    /// Smallest and largest density.
    pub fn range(&self) -> (f64, f64) {
        self.y.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)))
    }

    /// First and last log exposure.
    pub fn domain(&self) -> (f64, f64) {
        (self.x.first().copied().unwrap_or(0.0), self.x.last().copied().unwrap_or(0.0))
    }
}

/// A film or paper stock, ready for the model.
#[derive(Clone, Debug)]
pub struct Stock {
    pub id: String,
    pub name: String,
    pub kind: Kind,
    /// The light the stock is balanced for (`D55`, `TH-KG3`, …; see [`crate::spectrum::illuminant`]).
    pub reference_illuminant: String,
    /// A negative's intended paper.
    pub target_print: Option<String>,
    /// Linear spectral sensitivity of the red-, green- and blue-sensitive layers.
    pub sensitivity: [Spectrum; 3],
    /// Spectral density of the cyan, magenta and yellow dyes per unit layer density.
    pub dye: [Spectrum; 3],
    /// Spectral density of the unexposed, processed stock.
    pub base: Spectrum,
    /// Density above base against log10 exposure, per layer (R, G, B).
    pub curves: [Curve; 3],
}

#[derive(Deserialize)]
struct File {
    info: Info,
    data: Data,
}

#[derive(Deserialize)]
struct Info {
    stock: String,
    name: String,
    #[serde(rename = "type")]
    kind: String,
    support: String,
    #[serde(default)]
    target_print: Option<String>,
    #[serde(default)]
    reference_illuminant: Option<String>,
}

#[derive(Deserialize)]
struct Data {
    wavelengths: Vec<f64>,
    log_sensitivity: Vec<[Option<f64>; 3]>,
    channel_density: Vec<[Option<f64>; 3]>,
    base_density: Vec<Option<f64>>,
    log_exposure: Vec<f64>,
    density_curves: Vec<[Option<f64>; 3]>,
}

fn bad(id: &str, why: impl std::fmt::Display) -> FilmError {
    FilmError::Profile(format!("{id}: {why}"))
}

fn checked(id: &str, what: &str, v: f64) -> Result<f64, FilmError> {
    if v.is_finite() && v.abs() <= MAX_VALUE { Ok(v) } else { Err(bad(id, format!("{what}: value {v} out of range"))) }
}

/// Three spectra from 81 rows of three optional samples; `None` becomes `fill`.
fn spectra3(id: &str, what: &str, rows: &[[Option<f64>; 3]], fill: f64) -> Result<[Spectrum; 3], FilmError> {
    if rows.len() != N {
        return Err(bad(id, format!("{what}: {} samples, expected {N}", rows.len())));
    }
    let mut out = [[0.0; N]; 3];
    for (i, row) in rows.iter().enumerate() {
        for (spectrum, v) in out.iter_mut().zip(row) {
            if let Some(slot) = spectrum.get_mut(i) {
                *slot = checked(id, what, v.unwrap_or(fill))?;
            }
        }
    }
    Ok(out)
}

impl Stock {
    pub fn parse(json: &str) -> Result<Stock, FilmError> {
        if json.len() > MAX_JSON {
            return Err(bad("profile", "file too large"));
        }
        let f: File = serde_json::from_str(json).map_err(|e| bad("profile", e))?;
        let id = f.info.stock;
        if id.is_empty() || id.len() > 64 || !id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_') {
            return Err(bad("profile", format!("bad stock id {id:?}")));
        }
        let kind = match (f.info.support.as_str(), f.info.kind.as_str()) {
            ("paper", _) => Kind::Paper,
            (_, "negative") => Kind::Negative,
            (_, "positive") => Kind::Positive,
            (s, k) => return Err(bad(&id, format!("unsupported stock {s}/{k}"))),
        };
        let d = f.data;
        if d.wavelengths.len() != N || d.wavelengths.iter().enumerate().any(|(i, w)| (w - wavelength(i)).abs() > 1e-6) {
            return Err(bad(&id, "wavelengths are not 380..780 nm in 5 nm steps"));
        }
        // a missing log sensitivity is no sensitivity: 10^-40
        let log_sens = spectra3(&id, "log_sensitivity", &d.log_sensitivity, -MAX_VALUE)?;
        let sensitivity = log_sens.map(|s| s.map(|v| 10f64.powf(v)));
        let dye = spectra3(&id, "channel_density", &d.channel_density, 0.0)?;
        if d.base_density.len() != N {
            return Err(bad(&id, format!("base_density: {} samples, expected {N}", d.base_density.len())));
        }
        let mut base = [0.0; N];
        for (b, v) in base.iter_mut().zip(&d.base_density) {
            *b = checked(&id, "base_density", v.unwrap_or(0.0))?;
        }
        let n = d.log_exposure.len();
        if !(16..=4096).contains(&n) || d.density_curves.len() != n {
            return Err(bad(&id, "log_exposure and density_curves need the same 16..4096 samples"));
        }
        for v in &d.log_exposure {
            checked(&id, "log_exposure", *v)?;
        }
        if d.log_exposure.windows(2).any(|w| matches!(w, [a, b] if b <= a)) {
            return Err(bad(&id, "log_exposure must increase"));
        }
        let mut ys: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::with_capacity(n));
        for row in &d.density_curves {
            for (y, v) in ys.iter_mut().zip(row) {
                let v = v.ok_or_else(|| bad(&id, "density_curves has a gap"))?;
                y.push(checked(&id, "density_curves", v)?);
            }
        }
        let curves = ys.map(|y| Curve { x: d.log_exposure.clone(), y });
        Ok(Stock {
            name: f.info.name,
            kind,
            reference_illuminant: f.info.reference_illuminant.unwrap_or_else(|| "D55".into()),
            target_print: f.info.target_print,
            sensitivity,
            dye,
            base,
            curves,
            id,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const PORTRA: &str = include_str!("../../../assets/film/spektrafilm/kodak_portra_400.json");

    fn mutated(f: impl FnOnce(&mut Value)) -> Result<Stock, FilmError> {
        let mut v: Value = serde_json::from_str(PORTRA).unwrap();
        f(&mut v);
        Stock::parse(&v.to_string())
    }

    fn argmax(s: &Spectrum) -> f64 {
        wavelength((0..N).max_by(|&a, &b| s[a].total_cmp(&s[b])).unwrap())
    }

    #[test]
    fn portra_400_parses_with_rgb_layers_and_cmy_dyes() {
        let s = Stock::parse(PORTRA).unwrap();
        assert_eq!(s.id, "kodak_portra_400");
        assert_eq!(s.kind, Kind::Negative);
        assert_eq!(s.reference_illuminant, "D55");
        assert_eq!(s.target_print.as_deref(), Some("kodak_portra_endura"));
        assert!(argmax(&s.sensitivity[0]) > 600.0 && argmax(&s.sensitivity[2]) < 480.0);
        assert!(argmax(&s.dye[0]) > 620.0 && argmax(&s.dye[2]) < 480.0);
        let (lo, hi) = s.curves[1].range();
        assert!(lo < 0.05 && hi > 1.5, "{lo} {hi}");
        assert!((s.curves[0].eval(0.0) - 0.683).abs() < 0.02, "{}", s.curves[0].eval(0.0));
    }

    #[test]
    fn curve_eval_clamps_and_survives_nan() {
        let c = Curve { x: vec![0.0, 1.0, 2.0], y: vec![0.0, 1.0, 4.0] };
        assert_eq!(c.eval(-5.0), 0.0);
        assert_eq!(c.eval(0.5), 0.5);
        assert_eq!(c.eval(1.5), 2.5);
        assert_eq!(c.eval(9.0), 4.0);
        assert_eq!(c.eval(f64::NAN), 0.0);
        assert_eq!(c.eval(f64::INFINITY), 4.0);
    }

    #[test]
    fn malformed_profiles_are_errors_not_panics() {
        assert!(Stock::parse("").is_err());
        assert!(Stock::parse("{}").is_err());
        assert!(Stock::parse(&PORTRA[..PORTRA.len() / 2]).is_err());
        assert!(Stock::parse(&" ".repeat(MAX_JSON + 1)).is_err());
        assert!(mutated(|v| v["data"]["wavelengths"].as_array_mut().unwrap().truncate(80)).is_err());
        assert!(mutated(|v| v["data"]["log_exposure"][10] = 9.0.into()).is_err(), "not increasing");
        assert!(mutated(|v| v["data"]["density_curves"][5][1] = 1e300.into()).is_err());
        assert!(mutated(|v| v["data"]["density_curves"][5][1] = Value::Null).is_err(), "gap in a curve");
        assert!(mutated(|v| v["info"]["type"] = "instant".into()).is_err());
        assert!(mutated(|v| v["info"]["stock"] = "../../etc".into()).is_err(), "id charset");
        assert!(mutated(|v| v["data"]["log_exposure"].as_array_mut().unwrap().truncate(3)).is_err());
    }
}
