//! A film look: our film → lab scan or film → print model (see `docs/film-simulation.md`) for one stock, output and
//! set of settings. The spectral work is baked per (film, paper or scan) into 3-D grids and cached, so per pixel a
//! look costs a 3×3 matrix, a few 1-D tables and one or two trilinear lookups, and moving a film slider never
//! rebakes.

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
/// (film, paper or scan) pairs kept baked.
const CACHE: usize = 6;
/// Cache key of a negative's lab scan (paper ids are lower-case words, never this).
const SCAN_KEY: &str = "@scan";
/// Density per decade of exposure a lab scanner's fixed curve expects: a typical colour negative's gamma. A stock with a
/// steeper negative scans contrastier, a flatter one softer, as on a real minilab (it is not normalised away).
const SCAN_GAMMA: f64 = 0.62;
/// Stops over mid grey a lab scan takes to white (soft shoulder).
const SCAN_WHITE_STOPS: f64 = 4.0;
/// Strength of the lab scan's colour correction (0 = none, 1 = fully fitted to measured surfaces' true colours).
const SCAN_COLOUR: f64 = 1.0;
/// Push/pull development: negative contrast grows by this much per stop pushed (shrinks per stop pulled).
const PUSH_GAMMA_PER_STOP: f32 = 0.12;
/// Print filtration range, CC (1 CC = 0.01 density).
const MAX_CC: f64 = 30.0;
/// Print exposure range, EV.
const MAX_PRINT_EV: f64 = 1.0;
/// Push/pull range, stops.
const MAX_PUSH: f64 = 3.0;

/// How a negative becomes the picture (slides are always viewed directly).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Output {
    /// Scanned the way a minilab scanner does: each channel inverted, the orange mask taken out, soft contrast. Most
    /// film photos seen today are such scans.
    #[default]
    Scan,
    /// Printed on photo paper with an enlarger and viewed under D50: punchier, the darkroom look.
    Print,
}

/// What a look is built from (the `film.*` develop settings and Light ▸ Contrast, Whites, Blacks).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LookParams {
    /// Film stock id ([`crate::films`]).
    pub stock: String,
    /// Lab scan or print, for a negative.
    pub output: Output,
    /// Paper id for a printed negative; empty = the stock's own paper if it ships, else [`DEFAULT_PAPER`].
    pub paper: String,
    /// Push (+) or pull (−) processing, stops (±3): longer or shorter development steepens or flattens the film's
    /// densities around mid grey ([`PUSH_GAMMA_PER_STOP`] per stop), so contrast changes and mid grey keeps its
    /// brightness. (Brightness is Light ▸ Exposure.)
    pub exposure_ev: f64,
    /// Print exposure, EV (±1; more is a darker print).
    pub print_exposure_ev: f64,
    /// Enlarger filtration on top of the neutral balance, CC (±30; 1 CC = 0.01 density); the print exposure is
    /// compensated so mid grey keeps its brightness.
    pub print_magenta: f64,
    pub print_yellow: f64,
    /// Contrast, Whites and Blacks (−100..100) on the scan or print, after the film ([`post_tone`]): contrast like a
    /// paper grade around mid grey, Whites and Blacks the white and black points.
    pub contrast: f64,
    pub whites: f64,
    pub blacks: f64,
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
    /// Slide on a light box, or a negative's lab scan: densities → colour.
    Direct { view: Grid3 },
}

/// A ready film look.
#[derive(Clone)]
pub struct Look {
    baked: Arc<Baked>,
    /// Push/pull: the development contrast, and the layers' mid-grey densities it pivots around (so mid grey keeps
    /// its density, and the scan or print its brightness).
    push_gamma: f32,
    mid_density: [f32; 3],
    print_offset: [f32; 3],
    /// Contrast, Whites, Blacks after the film, −1..1.
    post: [f32; 3],
    /// Where scene mid grey comes out (luminance): the pivot of Contrast, Whites and Blacks.
    pivot: f32,
}

impl std::fmt::Debug for Look {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Look")
            .field("push_gamma", &self.push_gamma)
            .field("print_offset", &self.print_offset)
            .field("post", &self.post)
            .finish_non_exhaustive()
    }
}

impl Look {
    /// Linear Rec.2020 (scene) → display-linear Rec.2020 in 0..1 (the paper or slide base is white).
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let out = post_tone(self.film(rgb), self.post, self.pivot);
        out.map(|v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 })
    }

    /// The scan, print or slide before Contrast, Whites and Blacks.
    fn film(&self, rgb: [f32; 3]) -> [f32; 3] {
        let b = &*self.baked;
        let c = rgb.map(|v| if v.is_finite() { v.max(0.0) } else { 0.0 });
        let m = &b.to_exposure;
        let le: [f32; 3] = std::array::from_fn(|l| (m[l][0] * c[0] + m[l][1] * c[1] + m[l][2] * c[2]).max(1e-7).log10());
        // push/pull: longer or shorter development steepens or flattens the densities around mid grey
        let d: [f32; 3] = std::array::from_fn(|l| {
            let mid = self.mid_density[l];
            mid + (b.film_curves[l].eval(le[l]) - mid) * self.push_gamma
        });
        match &b.path {
            Path::Direct { view } => view.sample(d),
            Path::Print { to_paper, balance, paper_curves, scan } => {
                let lh = to_paper.sample(d);
                let dp: [f32; 3] = std::array::from_fn(|j| paper_curves[j].eval(lh[j] + balance[j] + self.print_offset[j]));
                scan.sample(dp)
            }
        }
    }
}

fn smooth(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn luma(c: [f32; 3]) -> f32 {
    0.2627 * c[0] + 0.6780 * c[1] + 0.0593 * c[2]
}

/// Light ▸ Contrast, Whites and Blacks (`post`, −1..1 each) on the scan or print (display-linear), as darkroom and lab
/// tools place them (paper grade, white and black points):
/// - Contrast: a power around mid grey on luminance (×1.5 slope at +1, ÷1.5 at −1), hue kept; mid grey stays.
/// - Whites: brightens (+, up to ×4) or dims (−, down to ×½) from mid grey up, fully from 0.8, so +100 clips from
///   about 2.5 stops over grey; shadows untouched.
/// - Blacks: lifts (+, up to 0.05 linear at black) or crushes (−, 0.03 linear to black) below mid grey, neutrally;
///   highlights untouched.
fn post_tone(o: [f32; 3], post: [f32; 3], pivot: f32) -> [f32; 3] {
    let [con, wh, bl] = post;
    let m = if pivot.is_finite() { pivot.clamp(0.05, 0.6) } else { MID_GREY as f32 };
    let mut o = o.map(|v| if v.is_finite() { v.max(0.0) } else { 0.0 });
    if con != 0.0 {
        let y = luma(o);
        if y > 1e-7 {
            let k = (m * (y / m).powf(1.5f32.powf(con)) / y).min(1e4);
            o = o.map(|v| v * k);
        }
    }
    if wh != 0.0 {
        // stronger up than down: +100 clips from about 2.5 stops over grey, −100 leaves white at a half
        let k = if wh > 0.0 { 2.0 } else { 1.0 };
        let g = (wh * k * smooth(m, 0.8, luma(o))).exp2();
        o = o.map(|v| v * g);
    }
    if bl != 0.0 {
        let w = 1.0 - smooth(0.0, m, luma(o));
        let off = if bl > 0.0 { 0.05 * bl } else { 0.03 * bl } * w;
        o = o.map(|v| (v + off).max(0.0));
    }
    o
}

/// The look for `p`, baking its stocks on first use.
pub fn look(p: &LookParams) -> Result<Look, FilmError> {
    let film = stock(&p.stock).ok_or_else(|| FilmError::UnknownStock(p.stock.clone()))?;
    let paper = match (film.kind, p.output) {
        (Kind::Paper, _) => return Err(FilmError::NotAFilm(film.id.clone())),
        (Kind::Positive, _) | (Kind::Negative, Output::Scan) => None,
        (Kind::Negative, Output::Print) => {
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
    let key = match (film.kind, paper) {
        (_, Some(q)) => q.id.as_str(),
        (Kind::Negative, None) => SCAN_KEY,
        _ => "",
    };
    let baked = cached(&film.id, key, || bake(film, paper))?;
    let finite = |v: f64, max: f64| if v.is_finite() { v.clamp(-max, max) } else { 0.0 };
    let unit = |v: f64| (finite(v, 100.0) / 100.0) as f32;
    let push = finite(p.exposure_ev, MAX_PUSH) as f32;
    let mid_density: [f32; 3] = std::array::from_fn(|l| baked.film_curves[l].eval(0.0));
    let pe = finite(p.print_exposure_ev, MAX_PRINT_EV) * std::f64::consts::LOG10_2;
    let (cm, cy) = (finite(p.print_magenta, MAX_CC) / 100.0, finite(p.print_yellow, MAX_CC) / 100.0);
    let mut look = Look {
        baked,
        push_gamma: (1.0 + PUSH_GAMMA_PER_STOP * push).clamp(0.5, 1.6),
        mid_density,
        print_offset: [pe as f32; 3],
        post: [unit(p.contrast), unit(p.whites), unit(p.blacks)],
        pivot: MID_GREY as f32,
    };
    if matches!(look.baked.path, Path::Print { .. }) && (cm != 0.0 || cy != 0.0) {
        // filtration changes colour, not brightness: find the overall print exposure that keeps mid grey's
        // brightness with the filters in (an enlarger's filter factor, compensated)
        let grey = [MID_GREY as f32; 3];
        let target = luma(look.film(grey));
        // red-sensitive layer: exposure only; green: less light through more magenta; blue: through more yellow
        let filtered = |e: f64| [(pe + e) as f32, (pe + e - cm) as f32, (pe + e - cy) as f32];
        let (mut lo, mut hi) = (-1.0f64, 1.0f64);
        for _ in 0..40 {
            let e = 0.5 * (lo + hi);
            look.print_offset = filtered(e);
            // more print exposure darkens the print
            if luma(look.film(grey)) > target {
                lo = e;
            } else {
                hi = e;
            }
        }
        look.print_offset = filtered(0.5 * (lo + hi));
    }
    look.pivot = luma(look.film([MID_GREY as f32; 3]));
    Ok(look)
}

fn cached(film: &str, paper: &str, make: impl FnOnce() -> Result<Baked, FilmError>) -> Result<Arc<Baked>, FilmError> {
    static BAKED: Mutex<Vec<((String, String), Arc<Baked>)>> = Mutex::new(Vec::new());
    let key = (film.to_string(), paper.to_string());
    let hit = BAKED.lock().unwrap_or_else(PoisonError::into_inner).iter().find(|(k, _)| *k == key).map(|(_, b)| b.clone());
    if let Some(b) = hit {
        return Ok(b);
    }
    // baked outside the lock: two threads may bake the same pair once each, never block each other; the first to
    // finish wins and the other drops its copy, so every look of a pair shares one bake
    let b = Arc::new(make()?);
    let mut c = BAKED.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, won)) = c.iter().find(|(k, _)| *k == key) {
        return Ok(won.clone());
    }
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
        None if film.kind == Kind::Negative => {
            let s = Scanner::new(film)?;
            Path::Direct { view: Grid3::bake(GRID, lo, hi, |d| s.view(film, d))? }
        }
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

/// A minilab scanner reading a negative: three narrow bands (σ 18 nm) at the stock's cyan, magenta and yellow dye
/// peaks (our own approximation of scanner LEDs, not measured data). Like a lab scanner's fixed setup:
/// - black point at the film base: each channel's density above the base (the orange mask divided out) is inverted,
///   and clear film is black, in every channel (a neutral floor);
/// - one fixed curve for every stock ([`SCAN_GAMMA`] on green), so a contrastier negative scans contrastier; red and
///   blue follow their own slope relative to green (as a scanner's channel balance does), so greys stay neutral
///   from black to white; balanced per channel at mid grey;
/// - white [`SCAN_WHITE_STOPS`] over mid grey, through a soft shoulder;
/// - a colour-correction matrix ([`Scanner::fit_colour`]), as a lab scanner applies: the film's layers are not display
///   primaries, so without one skies rotate towards cyan.
struct Scanner {
    channels: [Spectrum; 3],
    base: [f64; 3],
    mid: [f64; 3],
    /// Scan density per decade of scene exposure each channel is read at.
    gamma: [f64; 3],
    /// The shoulder's white, in units where mid grey is `white · 2^-SCAN_WHITE_STOPS`.
    white: f64,
    /// Colour correction on the linear scan (rows sum to 1: greys untouched).
    colour: [[f64; 3]; 3],
}

impl Scanner {
    fn new(neg: &Stock) -> Result<Scanner, FilmError> {
        // each band at its dye's absorption peak (kept inside the visible range the bands fit in)
        let peak = |dye: &Spectrum| {
            let i = (0..dye.len()).max_by(|&a, &b| dye[a].total_cmp(&dye[b])).unwrap_or(0);
            spectrum::wavelength(i).clamp(420.0, 700.0)
        };
        let channels = [peak(&neg.dye[0]), peak(&neg.dye[1]), peak(&neg.dye[2])]
            .map(|c: f64| -> Spectrum { std::array::from_fn(|i| (-0.5 * ((spectrum::wavelength(i) - c) / 18.0).powi(2)).exp()) });
        let t0 = transmittance(neg, [0.0; 3]);
        let base: [f64; 3] = std::array::from_fn(|k| spectrum::dot(&channels[k], &t0));
        if base.iter().any(|b| !(b.is_finite() && *b > 0.0)) {
            return Err(FilmError::Profile(format!("{}: the base passes no light to the scanner", neg.id)));
        }
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut s = Scanner { channels, base, mid: [0.0; 3], gamma: [SCAN_GAMMA; 3], white: 1.0, colour: identity };
        let at = |le: f64| neg.curves.each_ref().map(|c| c.eval(le));
        s.mid = s.density(neg, at(0.0));
        if s.mid.iter().any(|m| !(m.is_finite() && *m > 0.05)) {
            return Err(FilmError::Profile(format!("{}: mid grey makes no density on the negative", neg.id)));
        }
        // each channel's slope over ±1 stop around grey, relative to green's
        let (lo, hi) = (s.density(neg, at(-0.3)), s.density(neg, at(0.3)));
        let slope: [f64; 3] = std::array::from_fn(|k| ((hi[k] - lo[k]) / 0.6).max(0.05));
        s.gamma = slope.map(|v| SCAN_GAMMA * v / slope[1]);
        // the shoulder's white such that mid grey (white · 2^-stops) comes out at mid grey: shoulder(w / 16) = grey
        let ratio = (-SCAN_WHITE_STOPS).exp2();
        let (mut a, mut b) = (1.0, 64.0);
        for _ in 0..60 {
            let w = 0.5 * (a + b);
            if shoulder(w * ratio, w) < MID_GREY {
                a = w;
            } else {
                b = w;
            }
        }
        s.white = 0.5 * (a + b);
        if let Some(m) = s.fit_colour(neg) {
            s.colour = std::array::from_fn(|r| std::array::from_fn(|c| identity[r][c] + SCAN_COLOUR * (m[r][c] - identity[r][c])));
        }
        Ok(s)
    }

    /// The scanner's colour matrix for `neg`: the 3×3 with rows summing to 1 (greys untouched) that best maps the
    /// linear scans of 190 measured surfaces (rawtoaces-data's training reflectances, lit by the stock's reference
    /// light) to their true colours (Rec.2020, adapted to D65, mid grey = 1), by relative least squares. `None` if the
    /// fit is degenerate (the scan then has no colour correction).
    fn fit_colour(&self, neg: &Stock) -> Option<[[f64; 3]; 3]> {
        let samples: Vec<([f64; 3], [f64; 3])> = training_patches(neg)?
            .into_iter()
            .map(|(d, t)| (self.linear(neg, d), t))
            .filter(|(x, t)| x.iter().chain(t).all(|v| v.is_finite()) && 0.2627 * t[0] + 0.6780 * t[1] + 0.0593 * t[2] > 0.01)
            .collect();
        let start = least_squares_colour(&samples)?;
        Some(refine_colour(&samples, start))
    }

    /// Scan densities above the film base (the orange mask divided out).
    fn density(&self, neg: &Stock, d: [f64; 3]) -> [f64; 3] {
        let t = transmittance(neg, d);
        std::array::from_fn(|k| -(spectrum::dot(&self.channels[k], &t) / self.base[k]).max(1e-30).log10())
    }

    /// The linear scan before colour correction: scene-relative light per channel, 1 at mid grey, 0 at the film base
    /// (clear film is the black point).
    fn linear(&self, neg: &Stock, d: [f64; 3]) -> [f64; 3] {
        let dn = self.density(neg, d);
        std::array::from_fn(|k| {
            let x = 10f64.powf((dn[k] - self.mid[k]) / self.gamma[k]);
            let black = 10f64.powf(-self.mid[k] / self.gamma[k]);
            ((x - black) / (1.0 - black)).max(0.0)
        })
    }

    /// The scanned, inverted colour of a negative with layer densities `d` (linear Rec.2020).
    fn view(&self, neg: &Stock, d: [f64; 3]) -> [f64; 3] {
        let lin = self.linear(neg, d);
        let ratio = (-SCAN_WHITE_STOPS).exp2();
        std::array::from_fn(|r| {
            let c = &self.colour[r];
            let v = c[0] * lin[0] + c[1] * lin[1] + c[2] * lin[2];
            shoulder(v * self.white * ratio, self.white)
        })
    }
}

/// The rows-sum-to-1 3×3 minimising relative squared error from linear scans `x` to true colours `t` (each sample
/// weighted by 1/luminance², KKT per row): the start of [`refine_colour`]. `None` if degenerate.
fn least_squares_colour(samples: &[([f64; 3], [f64; 3])]) -> Option<[[f64; 3]; 3]> {
    // normal equations per output row: A = Σ w·x·xᵀ, b_r = Σ w·x·t_r
    let mut a = [[0.0f64; 3]; 3];
    let mut b = [[0.0f64; 3]; 3];
    for (x, t) in samples {
        let y = 0.2627 * t[0] + 0.6780 * t[1] + 0.0593 * t[2];
        let w = 1.0 / (y * y);
        for (i, (ai, xi)) in a.iter_mut().zip(x).enumerate() {
            for (aij, xj) in ai.iter_mut().zip(x) {
                *aij += w * xi * xj;
            }
            for (bj, tj) in b.iter_mut().zip(t) {
                bj[i] += w * xi * tj;
            }
        }
    }
    // minimise mᵀ A m − 2 b_rᵀ m subject to m₀ + m₁ + m₂ = 1 (KKT, 4 unknowns per row)
    let mut out = [[0.0f64; 3]; 3];
    for (row, br) in out.iter_mut().zip(&b) {
        let mut kkt = vec![vec![0.0f64; 5]; 4];
        for (i, (k, ai)) in kkt.iter_mut().zip(&a).enumerate() {
            for (kij, aij) in k.iter_mut().zip(ai) {
                *kij = 2.0 * aij;
            }
            k[3] = 1.0;
            k[4] = 2.0 * br[i];
        }
        if let Some(last) = kkt.get_mut(3) {
            *last = vec![1.0, 1.0, 1.0, 0.0, 1.0];
        }
        let m = basis::gauss_jordan(kkt)?;
        *row = [*m.first()?, *m.get(1)?, *m.get(2)?];
    }
    out.iter().flatten().all(|v| v.is_finite() && v.abs() < 10.0).then_some(out)
}

/// OkLab of a scene-relative colour (mid grey = 1).
fn oklab(c: [f64; 3]) -> [f64; 3] {
    lightcraft_color::perceptual::oklab_from_2020(c.map(|v| (v * MID_GREY) as f32)).map(f64::from)
}

/// Refines `start` (rows summing to 1) to minimise the perceptual error, Σ ΔE_OkLab², between the corrected scans and
/// the true colours: damped Gauss–Newton over the six free entries (two per row), a numeric Jacobian, 30 steps at
/// most. Least squares in linear light shrinks colours towards grey and spreads its error unevenly over hues; ΔE
/// counts lost chroma and hue alike. Returns `start` if no step improves it.
fn refine_colour(samples: &[([f64; 3], [f64; 3])], start: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let truth: Vec<[f64; 3]> = samples.iter().map(|(_, t)| oklab(t.map(|v| v.max(0.0)))).collect();
    // free parameters: m[r][1], m[r][2]; m[r][0] = 1 − m[r][1] − m[r][2]
    let unpack = |p: &[f64; 6]| -> [[f64; 3]; 3] { std::array::from_fn(|r| [1.0 - p[2 * r] - p[2 * r + 1], p[2 * r], p[2 * r + 1]]) };
    let residuals = |p: &[f64; 6]| -> Vec<f64> {
        let m = unpack(p);
        samples
            .iter()
            .zip(&truth)
            .flat_map(|((x, _), t)| {
                let c: [f64; 3] = std::array::from_fn(|r| (m[r][0] * x[0] + m[r][1] * x[1] + m[r][2] * x[2]).max(0.0));
                let o = oklab(c);
                [o[0] - t[0], o[1] - t[1], o[2] - t[2]]
            })
            .collect()
    };
    let cost = |r: &[f64]| r.iter().map(|v| v * v).sum::<f64>();
    let mut p: [f64; 6] = std::array::from_fn(|i| start[i / 2][1 + i % 2]);
    let mut r = residuals(&p);
    let mut c0 = cost(&r);
    let mut lambda = 1e-3;
    for _ in 0..30 {
        let h = 1e-5;
        let jac: Vec<Vec<f64>> = (0..6)
            .map(|j| {
                let mut q = p;
                q[j] += h;
                residuals(&q).iter().zip(&r).map(|(a, b)| (a - b) / h).collect()
            })
            .collect();
        // (JᵀJ + λ·diag) δ = −Jᵀr
        let mut sys = vec![vec![0.0f64; 7]; 6];
        for (i, row) in sys.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().take(6).enumerate() {
                *v = jac[i].iter().zip(&jac[j]).map(|(a, b)| a * b).sum::<f64>();
            }
            row[i] *= 1.0 + lambda;
            row[6] = -jac[i].iter().zip(&r).map(|(a, b)| a * b).sum::<f64>();
        }
        let Some(delta) = basis::gauss_jordan(sys) else { break };
        let q: [f64; 6] = std::array::from_fn(|i| p[i] + delta.get(i).copied().unwrap_or(0.0));
        let rq = residuals(&q);
        let cq = cost(&rq);
        if cq.is_finite() && cq < c0 {
            let gain = c0 - cq;
            (p, r, c0) = (q, rq, cq);
            lambda = (lambda * 0.3).max(1e-9);
            if gain < 1e-12 {
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e6 {
                break;
            }
        }
    }
    let m = unpack(&p);
    if m.iter().flatten().all(|v| v.is_finite() && v.abs() < 10.0) { m } else { start }
}

/// The 190 measured surfaces (rawtoaces-data training reflectances) lit by `neg`'s reference light: each one's layer
/// densities on the film (its ideal exposure, mid grey = log exposure 0) and its true colour (Rec.2020, adapted to
/// D65, mid grey = 1). `None` if the light or the stock is degenerate.
fn training_patches(neg: &Stock) -> Option<Vec<([f64; 3], [f64; 3])>> {
    let light = spectrum::illuminant(&neg.reference_illuminant);
    let lit: [Spectrum; 3] = std::array::from_fn(|l| mul(&light, &neg.sensitivity[l]));
    let grey: [f64; 3] = lit.each_ref().map(|s| MID_GREY * s.iter().sum::<f64>());
    let white = xyz(&light);
    if !(white[1].is_finite() && white[1] > 0.0) || grey.iter().any(|g| !(g.is_finite() && *g > 0.0)) {
        return None;
    }
    let k = 1.0 / (white[1] * MID_GREY);
    let to_2020 = REC2020.from_xyz().mul(&bradford(Xy::from_xyz(white), D65)).mul(&Mat3::diag(k, k, k));
    Some(
        crate::training_table::REFLECTANCES
            .iter()
            .map(|refl| {
                let d = std::array::from_fn(|l| neg.curves[l].eval((spectrum::dot(&lit[l], refl) / grey[l]).max(1e-7).log10()));
                (d, to_2020.apply(xyz(&mul(&light, refl))))
            })
            .collect(),
    )
}

/// A lab scan's soft shoulder: linear near black, `white` maps to 1.
fn shoulder(v: f64, white: f64) -> f64 {
    let v = v.max(0.0);
    v * (1.0 + v / (white * white)) / (1.0 + v)
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
        LookParams { stock: stock.into(), output: Output::Print, ..Default::default() }
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
    }

    /// Filtration changes the print's colour, not its brightness (the filter factor is compensated), and stays
    /// usable over its whole range: the grey wedge keeps its order.
    #[test]
    fn print_filters_change_colour_not_brightness() {
        let g = [MID_GREY as f32; 3];
        let base = luma(look(&params("kodak_portra_400")).unwrap().apply(g));
        for (m, y) in [(30.0, 0.0), (-30.0, 0.0), (0.0, 30.0), (0.0, -30.0), (15.0, -15.0), (100.0, 0.0)] {
            let l = look(&LookParams { print_magenta: m, print_yellow: y, ..params("kodak_portra_400") }).unwrap();
            assert!((luma(l.apply(g)) - base).abs() < 0.01, "M {m} Y {y}: mid grey {:?} vs {base}", l.apply(g));
            let ys: Vec<f32> = (0..25).map(|i| luma(l.apply([MID_GREY as f32 * (i as f32 * 0.5 - 8.0).exp2(); 3]))).collect();
            assert!(ys.windows(2).all(|w| w[1] >= w[0] - 1e-4), "M {m} Y {y}: {ys:?}");
        }
    }

    /// Push/pull keeps mid grey where it was and changes the curve: pushed is contrastier, pulled softer.
    #[test]
    fn push_and_pull_change_contrast_not_brightness() {
        for base in [scan("kodak_portra_400"), params("kodak_gold_200"), scan("fujifilm_velvia_100")] {
            let g = [MID_GREY as f32; 3];
            let looks: Vec<Look> = [-2.0, 0.0, 2.0].iter().map(|&e| look(&LookParams { exposure_ev: e, ..base.clone() }).unwrap()).collect();
            let mids: Vec<f32> = looks.iter().map(|l| luma(l.apply(g))).collect();
            assert!(mids.iter().all(|m| (m - mids[1]).abs() < 0.02), "{}: mid grey {mids:?}", base.stock);
            let c: Vec<f32> = looks.iter().map(contrast_of).collect();
            assert!(c[0] < c[1] && c[1] < c[2], "{}: contrast pulled / normal / pushed {c:?}", base.stock);
        }
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
        let c = look(&LookParams { contrast: 40.0, whites: -20.0, blacks: 10.0, ..params("kodak_ektar_100") }).unwrap();
        assert!(Arc::ptr_eq(&a.baked, &c.baked), "Contrast, Whites and Blacks don't rebake");
    }

    fn scan(stock: &str) -> LookParams {
        LookParams { stock: stock.into(), output: Output::Scan, ..Default::default() }
    }

    /// Log-luminance spread of the output between a scene 2 stops over and 2 stops under mid grey.
    fn contrast_of(l: &Look) -> f32 {
        let g = MID_GREY as f32;
        (luma(l.apply([g * 4.0; 3])) / luma(l.apply([g / 4.0; 3]))).log2()
    }

    #[test]
    fn lab_scans_are_the_default_neutral_and_mid_grey() {
        assert_eq!(LookParams::default().output, Output::Scan);
        for id in ["kodak_portra_400", "kodak_ektar_100", "kodak_gold_200"] {
            let l = look(&scan(id)).unwrap();
            let g = l.apply([MID_GREY as f32; 3]);
            assert!(spread(g) < 0.005, "{id}: {g:?}");
            assert!((luma(g) - MID_GREY as f32).abs() < 0.01, "{id}: mid grey scans to {g:?}");
            for v in [0.03f32, 0.6] {
                assert!(spread(l.apply([v; 3])) < 0.05, "{id} grey {v}: {:?}", l.apply([v; 3]));
            }
        }
    }

    /// The reason lab scans exist here: a scan is softer than a print on paper.
    #[test]
    fn lab_scans_are_flatter_than_prints() {
        for id in ["kodak_portra_400", "kodak_ektar_100", "kodak_gold_200"] {
            let (s, p) = (contrast_of(&look(&scan(id)).unwrap()), contrast_of(&look(&params(id)).unwrap()));
            assert!(s > 1.0 && s < p, "{id}: scan spans {s} stops over ±2 scene stops, print {p}");
        }
    }

    /// A lab scan's black point is the clear film: no light is black, in every channel; and white is reached about
    /// four stops over grey, near neutral.
    #[test]
    fn lab_scans_reach_a_neutral_black_and_white() {
        for id in ["kodak_portra_400", "kodak_ektar_100", "kodak_gold_200"] {
            let l = look(&scan(id)).unwrap();
            let black = l.apply([0.0; 3]);
            assert!(black.iter().all(|c| *c < 0.003) && spread(black) < 0.002, "{id}: black {black:?}");
            let deep = l.apply([MID_GREY as f32 / 256.0; 3]);
            assert!(luma(deep) < 0.01, "{id}: 8 stops under grey {deep:?}");
            let white = l.apply([MID_GREY as f32 * 16.0; 3]);
            // near neutral: the stock's own shoulder may tint it a little (Portra warm), never a cast
            assert!(luma(white) > 0.8 && spread(white) < 0.06, "{id}: 4 stops over grey {white:?}");
            let over = l.apply([MID_GREY as f32 * 64.0; 3]);
            assert!(luma(over) > 0.95, "{id}: 6 stops over grey {over:?}");
        }
    }

    /// The lab scanner's colour matrix keeps skies blue and skin true: measured surfaces (rawtoaces-data) come out much
    /// closer to their true hue than when the film's layers are read straight as display primaries (which rotated
    /// skies about 20° towards cyan), and no worse overall.
    #[test]
    fn lab_scan_colour_matrix_keeps_skies_blue() {
        use lightcraft_color::perceptual::oklab_from_2020;
        let hue = |c: [f64; 3]| {
            let l = oklab_from_2020(c.map(|v| v as f32));
            (l[2].atan2(l[1]).to_degrees(), l[1].hypot(l[2]))
        };
        for id in ["kodak_portra_400", "kodak_ektar_100", "kodak_gold_200"] {
            let neg = stock(id).unwrap();
            let fitted = Scanner::new(neg).unwrap();
            let mut plain = Scanner::new(neg).unwrap();
            plain.colour = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
            assert!(fitted.colour.iter().all(|r| (r.iter().sum::<f64>() - 1.0).abs() < 1e-9), "{id}: rows sum to 1");
            // mean |hue error| (°) with and without the matrix over the chromatic surfaces with hue in lo..hi
            let hue_error = |lo: f32, hi: f32| {
                let (mut n, mut with, mut without) = (0, 0.0f32, 0.0f32);
                for (d, truth) in training_patches(neg).unwrap() {
                    let (h, c) = hue(truth.map(|v| v * MID_GREY));
                    if h < lo || h > hi || c < 0.03 {
                        continue;
                    }
                    let diff = |s: &Scanner| ((hue(s.view(neg, d)).0 - h + 540.0).rem_euclid(360.0) - 180.0).abs();
                    n += 1;
                    with += diff(&fitted);
                    without += diff(&plain);
                }
                (n, with / n.max(1) as f32, without / n.max(1) as f32)
            };
            let (n, f, p) = hue_error(-150.0, -80.0);
            assert!(n >= 10 && f < p - 8.0 && f < 10.0, "{id}: sky hue error {f}° with the matrix, {p}° without ({n} surfaces)");
            let (_, f, p) = hue_error(35.0, 75.0);
            assert!(f < p && f < 8.0, "{id}: skin / orange hue error {f}° with the matrix, {p}° without");
            let (_, f, p) = hue_error(-180.0, 180.0);
            assert!(f < p - 2.0, "{id}: overall hue error {f}° with the matrix, {p}° without");
        }
    }

    /// One scanner setup for every stock: their own contrast shows (they are not all scanned to the same curve).
    #[test]
    fn lab_scans_keep_each_stocks_contrast() {
        let c: Vec<(&str, f32)> =
            ["kodak_portra_400", "kodak_ektar_100", "kodak_gold_200"].iter().map(|id| (*id, contrast_of(&look(&scan(id)).unwrap()))).collect();
        let (lo, hi) = c.iter().fold((f32::MAX, f32::MIN), |(lo, hi), (_, v)| (lo.min(*v), hi.max(*v)));
        assert!(hi - lo > 0.1, "the stocks scan to the same contrast: {c:?}");
    }

    #[test]
    fn lab_scans_keep_order_hue_and_range() {
        let l = look(&scan("kodak_portra_400")).unwrap();
        let ys: Vec<f32> = [0.005f32, 0.02, 0.08, 0.184, 0.5, 1.5, 6.0].iter().map(|&v| luma(l.apply([v; 3]))).collect();
        assert!(ys.windows(2).all(|w| w[1] >= w[0] - 1e-4), "{ys:?}");
        let r = l.apply([0.4, 0.05, 0.05]);
        assert!(r[0] > r[1] && r[0] > r[2], "{r:?}");
        let b = l.apply([0.03, 0.06, 0.4]);
        assert!(b[2] > b[0] && b[2] > b[1], "{b:?}");
        for p in [[f32::NAN; 3], [f32::INFINITY; 3], [1e6; 3], [0.0; 3]] {
            assert!(l.apply(p).iter().all(|c| c.is_finite() && (0.0..=1.0).contains(c)), "{p:?}");
        }
    }

    #[test]
    fn scan_and_print_are_baked_apart_and_slides_ignore_the_output() {
        let s = look(&scan("kodak_gold_200")).unwrap();
        let p = look(&params("kodak_gold_200")).unwrap();
        assert!(!Arc::ptr_eq(&s.baked, &p.baked));
        let vs = look(&scan("fujifilm_velvia_100")).unwrap();
        let vp = look(&params("fujifilm_velvia_100")).unwrap();
        assert!(Arc::ptr_eq(&vs.baked, &vp.baked), "a slide is viewed directly either way");
        // a scan has no paper: a paper id that isn't one is no error there
        assert!(look(&LookParams { paper: "kodak_ektar_100".into(), ..scan("kodak_portra_400") }).is_ok());
    }

    /// Light ▸ Contrast, Whites and Blacks act on the scan or print: contrast around mid grey, Whites on the top
    /// (shadows untouched, +100 clips), Blacks on the bottom (highlights untouched, −100 crushes to black).
    #[test]
    fn contrast_whites_and_blacks_act_on_the_scan_or_print() {
        for base in [scan("kodak_portra_400"), params("kodak_portra_400"), scan("fujifilm_velvia_100")] {
            let plain = look(&base).unwrap();
            let flat = look(&LookParams { contrast: -100.0, ..base.clone() }).unwrap();
            let punchy = look(&LookParams { contrast: 100.0, ..base.clone() }).unwrap();
            let (c0, c_lo, c_hi) = (contrast_of(&plain), contrast_of(&flat), contrast_of(&punchy));
            assert!(c_lo < c0 && c0 < c_hi, "{:?}: contrast {c_lo} / {c0} / {c_hi}", base.output);
            // mid grey stays put under Contrast
            let g = [MID_GREY as f32; 3];
            assert!((luma(flat.apply(g)) - luma(plain.apply(g))).abs() < 0.005);
            let hi = [MID_GREY as f32 * 8.0; 3];
            let lo = [MID_GREY as f32 / 16.0; 3];
            let whites = look(&LookParams { whites: -100.0, ..base.clone() }).unwrap();
            assert!(luma(whites.apply(hi)) < luma(plain.apply(hi)) - 0.01, "Whites − darkens highlights");
            assert!((luma(whites.apply(lo)) - luma(plain.apply(lo))).abs() < 0.002, "Whites leaves shadows");
            let blacks = look(&LookParams { blacks: 100.0, ..base.clone() }).unwrap();
            let (lifted, base_lo) = (luma(blacks.apply(lo)), luma(plain.apply(lo)));
            assert!(lifted > base_lo * 1.1, "Blacks + lifts shadows: {base_lo} -> {lifted}");
            assert!((luma(blacks.apply(hi)) - luma(plain.apply(hi))).abs() < 0.002, "Blacks leaves highlights");
            let crushed = look(&LookParams { blacks: -100.0, ..base.clone() }).unwrap();
            assert!(luma(crushed.apply([MID_GREY as f32 / 64.0; 3])) < 0.001, "Blacks − crushes 6 stops under grey to black");
            let clipped = look(&LookParams { whites: 100.0, ..base.clone() }).unwrap();
            assert!(luma(clipped.apply([MID_GREY as f32 * 6.0; 3])) > 0.99, "Whites + clips 2.5 stops over grey");
        }
        let wild = look(&LookParams { contrast: f64::NAN, whites: 1e300, blacks: f64::NEG_INFINITY, ..scan("kodak_gold_200") }).unwrap();
        assert!(wild.apply([0.2; 3]).iter().all(|c| c.is_finite()));
    }

    fn dummy_baked() -> Result<Baked, FilmError> {
        let t = || Table1::from_fn(0.0, 1.0, 2, |x| x);
        Ok(Baked {
            to_exposure: [[1.0; 3]; 3],
            film_curves: [t(), t(), t()],
            path: Path::Direct { view: Grid3::bake(2, [0.0; 3], [1.0; 3], |p| p)? },
        })
    }

    #[test]
    fn a_pair_baked_twice_at_once_keeps_one_cache_entry() {
        // another thread finishes baking the same pair while this one bakes: both must end up with the entry that
        // won, so later looks share it
        let mut inner = None;
        let outer = cached("test_race_film", "test_race_paper", || {
            inner = Some(cached("test_race_film", "test_race_paper", dummy_baked).unwrap());
            dummy_baked()
        })
        .unwrap();
        let later = cached("test_race_film", "test_race_paper", dummy_baked).unwrap();
        assert!(Arc::ptr_eq(&outer, &later));
        assert!(Arc::ptr_eq(&inner.unwrap(), &later));
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
