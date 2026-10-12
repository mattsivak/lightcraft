//! The stocks LightCraft ships: spektrafilm profiles by Andrea Volpato (CC BY-SA 4.0), unmodified, in
//! `assets/film/spektrafilm/` (README and licence there; `assets/ATTRIBUTION.md`).

use std::sync::OnceLock;

use serde::Serialize;

use crate::profile::{Kind, Stock};

const BUILTIN: &[&str] = &[
    include_str!("../../../assets/film/spektrafilm/kodak_portra_400.json"),
    include_str!("../../../assets/film/spektrafilm/kodak_ektar_100.json"),
    include_str!("../../../assets/film/spektrafilm/kodak_gold_200.json"),
    include_str!("../../../assets/film/spektrafilm/fujifilm_velvia_100.json"),
    include_str!("../../../assets/film/spektrafilm/kodak_portra_endura.json"),
];

/// The paper a negative is printed on when neither the edit nor the stock names one that ships.
pub const DEFAULT_PAPER: &str = "kodak_portra_endura";

/// Every built-in stock that parses (one that doesn't is left out, never a crash).
fn all() -> &'static [Stock] {
    static ALL: OnceLock<Vec<Stock>> = OnceLock::new();
    ALL.get_or_init(|| BUILTIN.iter().filter_map(|j| Stock::parse(j).ok()).collect())
}

/// The built-in stock `id`.
pub fn stock(id: &str) -> Option<&'static Stock> {
    all().iter().find(|s| s.id == id)
}

/// A stock as menus and agents list it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct StockInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub kind: Kind,
}

fn infos(keep: impl Fn(Kind) -> bool) -> Vec<StockInfo> {
    all().iter().filter(|s| keep(s.kind)).map(|s| StockInfo { id: &s.id, name: &s.name, kind: s.kind }).collect()
}

/// Films (negative and slide), in shipping order.
pub fn films() -> Vec<StockInfo> {
    infos(|k| k != Kind::Paper)
}

/// Print papers.
pub fn papers() -> Vec<StockInfo> {
    infos(|k| k == Kind::Paper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spectrum::{N, wavelength};

    #[test]
    fn every_builtin_stock_parses() {
        for j in BUILTIN {
            Stock::parse(j).unwrap();
        }
        assert_eq!(all().len(), BUILTIN.len());
        let ids: Vec<_> = films().iter().map(|f| f.id).collect();
        assert_eq!(ids, ["kodak_portra_400", "kodak_ektar_100", "kodak_gold_200", "fujifilm_velvia_100"]);
        assert_eq!(papers().iter().map(|p| p.id).collect::<Vec<_>>(), [DEFAULT_PAPER]);
        assert_eq!(stock("fujifilm_velvia_100").unwrap().kind, Kind::Positive);
        assert!(stock("nope").is_none());
    }

    #[test]
    fn every_stock_has_rgb_layers_and_cmy_dyes() {
        let peak = |s: &crate::spectrum::Spectrum| wavelength((0..N).max_by(|&a, &b| s[a].total_cmp(&s[b])).unwrap());
        for s in all() {
            let (r, g, b) = (peak(&s.sensitivity[0]), peak(&s.sensitivity[1]), peak(&s.sensitivity[2]));
            assert!(r > 580.0 && (500.0..600.0).contains(&g) && b < 500.0, "{} sensitivity peaks {r} {g} {b}", s.id);
            let (c, m, y) = (peak(&s.dye[0]), peak(&s.dye[1]), peak(&s.dye[2]));
            assert!(c > 600.0 && (500.0..600.0).contains(&m) && y < 500.0, "{} dye peaks {c} {m} {y}", s.id);
        }
    }

    /// The model's assumption: dye spectra are per unit layer density, so base + Σ D(x)·dye, with the layer
    /// densities read off the curves at one log exposure x near 0, is the mid-scale neutral the data sheet measured.
    ///
    /// The data sheets' mid-scale neutral is not exactly at log exposure 0: the best fit is at x = +0.15 (Portra
    /// 400), +0.28 (Ektar 100), −0.11 (Gold 200), 0 (Velvia 100, Portra Endura), with RMS 0.011 / 0.007 / 0.010 /
    /// 0.005 / 0.022. At x = 0 Ektar's RMS is 0.156 because of that offset (a near-constant −0.15 across the
    /// spectrum), not because its dyes are scaled differently.
    #[test]
    fn dyes_reproduce_the_midscale_neutral() {
        for j in BUILTIN {
            let s = Stock::parse(j).unwrap();
            let v: serde_json::Value = serde_json::from_str(j).unwrap();
            let mid = v["data"]["midscale_neutral_density"].as_array().unwrap();
            let fit = |x: f64| {
                let d: Vec<f64> = s.curves.iter().map(|c| c.eval(x)).collect();
                let errs: Vec<f64> = (0..N)
                    .filter_map(|i| {
                        let m = mid[i].as_f64()?;
                        Some(s.base[i] + d[0] * s.dye[0][i] + d[1] * s.dye[1][i] + d[2] * s.dye[2][i] - m)
                    })
                    .collect();
                ((errs.iter().map(|e| e * e).sum::<f64>() / errs.len() as f64).sqrt(), errs.len())
            };
            let (rms, x) = (-50..=50).map(|k| (fit(f64::from(k) / 100.0).0, f64::from(k) / 100.0)).min_by(|a, b| a.0.total_cmp(&b.0)).unwrap();
            let samples = fit(x).1;
            assert!(samples > 40 && rms < 0.03, "{}: rms {rms} at log exposure {x} over {samples}", s.id);
            assert!(x.abs() <= 0.35, "{}: mid-scale neutral at log exposure {x}, far from 0", s.id);
        }
    }
}
