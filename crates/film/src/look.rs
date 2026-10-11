//! A film look: our film → print → scan model (see `docs/film-simulation.md`) for one stock, paper and set of
//! print settings. The spectral work is baked per (film, paper) into two 3-D grids and cached, so per pixel a look
//! costs a 3×3 matrix, six 1-D tables and two trilinear lookups, and moving a film slider never rebakes.

use std::sync::{Arc, Mutex, PoisonError};

use lightcraft_color::{D65, Mat3, REC2020, Xy, bradford};

use crate::FilmError;
use crate::basis;
use crate::grid::{Grid3, Table1};
use crate::profile::{Curve, Kind, Stock};
use crate::spectrum::{self, Spectrum, mul, xyz};
use crate::stocks::{DEFAULT_PAPER, stock};

/// The scene grey that exposes a film to log10 exposure 0 (mid-scale on every shipped curve).
const MID_GREY: f64 = 0.184;
/// Side of the baked 3-D grids.
const GRID: usize = 33;
/// Samples of each baked characteristic curve.
const CURVE_SAMPLES: usize = 1024;
/// (film, paper) pairs kept baked.
const CACHE: usize = 6;

/// What a look is built from (the `film.*` develop settings).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LookParams {
    /// Film stock id ([`crate::films`]).
    pub stock: String,
    /// Paper id for a negative; empty = the stock's own paper if it ships, else [`DEFAULT_PAPER`].
    pub paper: String,
    /// Film exposure, EV.
    pub exposure_ev: f64,
    /// Print exposure, EV (more is a darker print).
    pub print_exposure_ev: f64,
    /// Enlarger filtration on top of the neutral balance, CC (1 CC = 0.01 density).
    pub print_magenta: f64,
    pub print_yellow: f64,
}

/// What depends on the stocks only.
struct Baked {
    to_exposure: [[f32; 3]; 3],
    film_curves: [Table1; 3],
    path: Path,
}

enum Path {
    /// Negative: densities → paper log exposure, balance, paper curves, paper densities → colour.
    Print { to_paper: Grid3, balance: [f32; 3], paper_curves: [Table1; 3], scan: Grid3 },
    /// Slide: densities → colour on a light box.
    Direct { view: Grid3 },
}

/// A ready film look.
#[derive(Clone)]
pub struct Look {
    baked: Arc<Baked>,
    gain: f32,
    print_offset: [f32; 3],
}

impl std::fmt::Debug for Look {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Look").field("gain", &self.gain).field("print_offset", &self.print_offset).finish_non_exhaustive()
    }
}

impl Look {
    /// Linear Rec.2020 (scene) → display-linear Rec.2020 in 0..1 (the paper or slide base is white).
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let b = &*self.baked;
        let c = rgb.map(|v| if v.is_finite() { v.max(0.0) } else { 0.0 });
        let m = &b.to_exposure;
        let le: [f32; 3] = std::array::from_fn(|l| ((m[l][0] * c[0] + m[l][1] * c[1] + m[l][2] * c[2]) * self.gain).max(1e-7).log10());
        let d: [f32; 3] = std::array::from_fn(|l| b.film_curves[l].eval(le[l]));
        let out = match &b.path {
            Path::Direct { view } => view.sample(d),
            Path::Print { to_paper, balance, paper_curves, scan } => {
                let lh = to_paper.sample(d);
                let dp: [f32; 3] = std::array::from_fn(|j| paper_curves[j].eval(lh[j] + balance[j] + self.print_offset[j]));
                scan.sample(dp)
            }
        };
        out.map(|v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 })
    }
}

/// The look for `p`, baking its stocks on first use.
pub fn look(p: &LookParams) -> Result<Look, FilmError> {
    let film = stock(&p.stock).ok_or_else(|| FilmError::UnknownStock(p.stock.clone()))?;
    let paper = match film.kind {
        Kind::Paper => return Err(FilmError::NotAFilm(film.id.clone())),
        Kind::Positive => None,
        Kind::Negative => {
            let id = if p.paper.is_empty() {
                film.target_print.as_deref().filter(|t| stock(t).is_some()).unwrap_or(DEFAULT_PAPER)
            } else {
                p.paper.as_str()
            };
            let paper = stock(id).ok_or_else(|| FilmError::UnknownStock(id.to_string()))?;
            if paper.kind != Kind::Paper {
                return Err(FilmError::NotAPaper(paper.id.clone()));
            }
            Some(paper)
        }
    };
    let baked = cached(&film.id, paper.map_or("", |q| q.id.as_str()), || bake(film, paper))?;
    let ev = |v: f64| if v.is_finite() { v.clamp(-10.0, 10.0) } else { 0.0 };
    let cc = |v: f64| if v.is_finite() { v.clamp(-200.0, 200.0) / 100.0 } else { 0.0 };
    let pe = ev(p.print_exposure_ev) * std::f64::consts::LOG10_2;
    Ok(Look {
        baked,
        gain: 2f64.powf(ev(p.exposure_ev)) as f32,
        // red-sensitive layer: exposure only; green: less light through more magenta; blue: through more yellow
        print_offset: [pe as f32, (pe - cc(p.print_magenta)) as f32, (pe - cc(p.print_yellow)) as f32],
    })
}

fn cached(film: &str, paper: &str, make: impl FnOnce() -> Result<Baked, FilmError>) -> Result<Arc<Baked>, FilmError> {
    static BAKED: Mutex<Vec<((String, String), Arc<Baked>)>> = Mutex::new(Vec::new());
    let key = (film.to_string(), paper.to_string());
    let hit = BAKED.lock().unwrap_or_else(PoisonError::into_inner).iter().find(|(k, _)| *k == key).map(|(_, b)| b.clone());
    if let Some(b) = hit {
        return Ok(b);
    }
    // baked outside the lock: two threads may bake the same pair once each, never block each other
    let b = Arc::new(make()?);
    let mut c = BAKED.lock().unwrap_or_else(PoisonError::into_inner);
    if c.len() >= CACHE {
        c.remove(0);
    }
    c.push((key, b.clone()));
    Ok(b)
}

fn bake(film: &Stock, paper: Option<&Stock>) -> Result<Baked, FilmError> {
    let to_exposure = exposure_matrix(film)?;
    let film_curves = film.curves.each_ref().map(table);
    let (lo, hi) = density_box(&film.curves);
    let path = match paper {
        None => {
            let v = Viewer::new(film)?;
            Path::Direct { view: Grid3::bake(GRID, lo, hi, |d| v.view(film, d))? }
        }
        Some(p) => {
            let lamp = spectrum::enlarger_lamp();
            let to_paper = Grid3::bake(GRID, lo, hi, |d| print_log_exposure(film, p, &lamp, d))?;
            let (plo, phi) = density_box(&p.curves);
            let v = Viewer::new(p)?;
            let scan = Grid3::bake(GRID, plo, phi, |d| v.view(p, d))?;
            let mid = film.curves.each_ref().map(|c| c.eval(0.0));
            let balance = neutral_balance(print_log_exposure(film, p, &lamp, mid), p, &v);
            Path::Print { to_paper, balance, paper_curves: p.curves.each_ref().map(table), scan }
        }
    };
    Ok(Baked { to_exposure, film_curves, path })
}

fn table(c: &Curve) -> Table1 {
    let (lo, hi) = c.domain();
    Table1::from_fn(lo, hi, CURVE_SAMPLES, |x| c.eval(x))
}

fn density_box(curves: &[Curve; 3]) -> ([f64; 3], [f64; 3]) {
    let r = curves.each_ref().map(Curve::range);
    (r.map(|(lo, _)| lo), r.map(|(lo, hi)| hi.max(lo + 1e-6)))
}

/// Linear Rec.2020 → layer exposure, rows scaled so [`MID_GREY`] exposes every layer to 1.
fn exposure_matrix(s: &Stock) -> Result<[[f32; 3]; 3], FilmError> {
    let b = basis::rec2020();
    let light = spectrum::illuminant(&s.reference_illuminant);
    let mut out = [[0.0f32; 3]; 3];
    for (l, row) in out.iter_mut().enumerate() {
        let lit = mul(&light, &s.sensitivity[l]);
        let m: [f64; 3] = std::array::from_fn(|k| spectrum::dot(&lit, &b[k]));
        let grey = MID_GREY * (m[0] + m[1] + m[2]);
        if !(grey.is_finite() && grey > 0.0) {
            return Err(FilmError::Profile(format!("{}: layer {l} sees no light", s.id)));
        }
        *row = m.map(|v| (v / grey) as f32);
    }
    Ok(out)
}

/// Spectral transmittance (or reflectance) of `s` with layer densities `d` above base.
fn transmittance(s: &Stock, d: [f64; 3]) -> Spectrum {
    std::array::from_fn(|i| 10f64.powf(-(s.base[i] + d[0] * s.dye[0][i] + d[1] * s.dye[1][i] + d[2] * s.dye[2][i])))
}

/// log10 exposure of each paper layer under the enlarger lamp through a negative with densities `d`.
fn print_log_exposure(neg: &Stock, paper: &Stock, lamp: &Spectrum, d: [f64; 3]) -> [f64; 3] {
    let light = mul(lamp, &transmittance(neg, d));
    std::array::from_fn(|j| spectrum::dot(&light, &paper.sensitivity[j]).max(1e-30).log10())
}

/// A dyed stock seen under D50 (a print, or a slide on a light box) → linear Rec.2020, adapted so the unexposed
/// base is white (1, 1, 1).
struct Viewer {
    d50: Spectrum,
    to_2020: Mat3,
}

impl Viewer {
    fn new(s: &Stock) -> Result<Viewer, FilmError> {
        let d50 = spectrum::daylight(5003.0);
        let white = xyz(&mul(&d50, &transmittance(s, [0.0; 3])));
        if !(white[1].is_finite() && white[1] > 0.0) {
            return Err(FilmError::Profile(format!("{}: the base passes no light", s.id)));
        }
        let k = 1.0 / white[1];
        let to_2020 = REC2020.from_xyz().mul(&bradford(Xy::from_xyz(white), D65)).mul(&Mat3::diag(k, k, k));
        Ok(Viewer { d50, to_2020 })
    }

    fn view(&self, s: &Stock, d: [f64; 3]) -> [f64; 3] {
        self.to_2020.apply(xyz(&mul(&self.d50, &transmittance(s, d))))
    }
}

/// Per-layer log offsets so the print of a mid-grey negative (`lh`: its paper log exposures) is neutral, at the
/// brightness the plain offsets `-lh` give. Newton's method with a numeric Jacobian; `-lh` if it can't converge.
fn neutral_balance(lh: [f64; 3], paper: &Stock, v: &Viewer) -> [f32; 3] {
    let out = |o: [f64; 3]| v.view(paper, std::array::from_fn(|j| paper.curves[j].eval(lh[j] + o[j])));
    let start = lh.map(|x| -x);
    let target = {
        let c = out(start);
        0.2627 * c[0] + 0.6780 * c[1] + 0.0593 * c[2]
    };
    let resid = |o: [f64; 3]| out(o).map(|c| c - target);
    let mut o = start;
    for _ in 0..30 {
        let f = resid(o);
        if f.iter().all(|r| r.abs() < 1e-6) {
            break;
        }
        let h = 1e-4;
        let cols: [[f64; 3]; 3] = std::array::from_fn(|c| {
            let mut oc = o;
            oc[c] += h;
            let fc = resid(oc);
            std::array::from_fn(|r| (fc[r] - f[r]) / h)
        });
        let Some(inv) = Mat3(cols).transpose().inverse() else { break };
        let step = inv.apply(f);
        o = std::array::from_fn(|c| o[c] - step[c].clamp(-0.5, 0.5));
    }
    let o = if o.iter().all(|v| v.is_finite()) { o } else { start };
    o.map(|v| v as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(stock: &str) -> LookParams {
        LookParams { stock: stock.into(), ..Default::default() }
    }

    fn luma(c: [f32; 3]) -> f32 {
        0.2627 * c[0] + 0.6780 * c[1] + 0.0593 * c[2]
    }

    fn spread(c: [f32; 3]) -> f32 {
        c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
    }

    #[test]
    fn mid_grey_prints_neutral_on_every_negative() {
        for id in ["kodak_portra_400", "kodak_ektar_100", "kodak_gold_200"] {
            let l = look(&params(id)).unwrap();
            let g = l.apply([MID_GREY as f32; 3]);
            assert!(spread(g) < 0.005, "{id}: {g:?}");
            assert!((0.08..0.6).contains(&luma(g)), "{id}: {g:?}");
            for v in [0.03f32, 0.6] {
                assert!(spread(l.apply([v; 3])) < 0.1, "{id} grey {v}: {:?}", l.apply([v; 3]));
            }
        }
    }

    #[test]
    fn brighter_scenes_make_brighter_prints_and_slides() {
        for id in ["kodak_portra_400", "fujifilm_velvia_100"] {
            let l = look(&params(id)).unwrap();
            let ys: Vec<f32> = [0.005f32, 0.02, 0.08, 0.184, 0.5, 1.5, 6.0].iter().map(|&v| luma(l.apply([v; 3]))).collect();
            assert!(ys.windows(2).all(|w| w[1] >= w[0] - 1e-4), "{id}: {ys:?}");
            assert!(ys[ys.len() - 1] > ys[0] + 0.3, "{id}: {ys:?}");
        }
    }

    #[test]
    fn saturated_colours_keep_their_hue() {
        let l = look(&params("kodak_portra_400")).unwrap();
        let r = l.apply([0.4, 0.05, 0.05]);
        assert!(r[0] > r[1] && r[0] > r[2], "{r:?}");
        let b = l.apply([0.03, 0.06, 0.4]);
        assert!(b[2] > b[0] && b[2] > b[1], "{b:?}");
    }

    #[test]
    fn hostile_pixels_give_finite_output_in_range() {
        let l = look(&params("kodak_portra_400")).unwrap();
        let v = look(&params("fujifilm_velvia_100")).unwrap();
        for p in [[f32::NAN; 3], [f32::INFINITY; 3], [f32::NEG_INFINITY; 3], [-1.0, 0.5, 2.0], [1e6; 3], [0.0; 3], [f32::MAX; 3]] {
            for o in [l.apply(p), v.apply(p)] {
                assert!(o.iter().all(|c| c.is_finite() && (0.0..=1.0).contains(c)), "{p:?} -> {o:?}");
            }
        }
    }

    #[test]
    fn print_controls_move_the_print() {
        let base = look(&params("kodak_portra_400")).unwrap().apply([MID_GREY as f32; 3]);
        let darker = look(&LookParams { print_exposure_ev: 1.0, ..params("kodak_portra_400") }).unwrap().apply([MID_GREY as f32; 3]);
        assert!(luma(darker) < luma(base) - 0.02, "{base:?} {darker:?}");
        let magenta = look(&LookParams { print_magenta: 20.0, ..params("kodak_portra_400") }).unwrap().apply([MID_GREY as f32; 3]);
        assert!(magenta[1] / luma(magenta) > base[1] / luma(base), "more magenta filtration makes a greener print: {magenta:?}");
        let film_plus = look(&LookParams { exposure_ev: 1.0, ..params("kodak_portra_400") }).unwrap().apply([MID_GREY as f32; 3]);
        assert!(luma(film_plus) > luma(base), "{film_plus:?}");
    }

    #[test]
    fn hostile_parameters_are_clamped() {
        let p = LookParams { exposure_ev: f64::NAN, print_exposure_ev: 1e300, print_magenta: f64::INFINITY, ..params("kodak_portra_400") };
        let o = look(&p).unwrap().apply([0.2; 3]);
        assert!(o.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn wrong_ids_are_errors() {
        assert_eq!(look(&params("nope")).unwrap_err(), FilmError::UnknownStock("nope".into()));
        assert_eq!(look(&params("kodak_portra_endura")).unwrap_err(), FilmError::NotAFilm("kodak_portra_endura".into()));
        let p = LookParams { paper: "kodak_ektar_100".into(), ..params("kodak_portra_400") };
        assert_eq!(look(&p).unwrap_err(), FilmError::NotAPaper("kodak_ektar_100".into()));
        let p = LookParams { paper: "nope".into(), ..params("kodak_portra_400") };
        assert_eq!(look(&p).unwrap_err(), FilmError::UnknownStock("nope".into()));
    }

    #[test]
    fn slider_changes_reuse_the_baked_stages() {
        let a = look(&params("kodak_ektar_100")).unwrap();
        let b = look(&LookParams { print_exposure_ev: 0.3, print_yellow: -10.0, exposure_ev: 0.5, ..params("kodak_ektar_100") }).unwrap();
        assert!(Arc::ptr_eq(&a.baked, &b.baked));
    }

    #[test]
    #[ignore = "timing; run with --release -- --ignored --nocapture"]
    fn bake_and_apply_timing() {
        let t = std::time::Instant::now();
        let l = look(&params("kodak_gold_200")).unwrap();
        let bake = t.elapsed();
        let t = std::time::Instant::now();
        let mut acc = 0.0f32;
        for i in 0..2_500_000u32 {
            let v = (i % 1000) as f32 / 400.0;
            acc += l.apply([v, v * 0.8, v * 0.6])[1];
        }
        eprintln!("bake {bake:?}, 2.5 MP apply {:?} ({acc})", t.elapsed());
    }
}
