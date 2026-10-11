# Film Simulation (issue #988) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A develop setting that renders the photo as if shot on a real film stock (Kodak Portra 400, Ektar 100,
Gold 200, Fujifilm Velvia 100), printed on photo paper and scanned. It replaces the tone map with a physically based
film → print → scan model driven by measured stock data.

**Architecture:** A new L1 crate `lightcraft-film` holds our own spectral model and the stock data. The data are the
spektrafilm profiles, vendored unmodified under CC BY-SA 4.0. Per (stock, paper) pair the crate bakes two 33³ lookup
grids once:
- negative densities → paper log exposure;
- paper densities → linear Rec.2020.

Per pixel the work is then cheap: a 3×3 exposure matrix, three 1-D characteristic curves, two trilinear lookups and
three paper curves. The pipeline's per-pixel stage (`crates/pipeline/src/finish.rs`) calls `Look::apply` in place of
the tone map when `DevelopSettings.film.enabled`. Phase 1 renders on the CPU only; the GPU path declines, as it does for
LUT profiles.

**Tech Stack:** Rust 2024 workspace crate, `serde`/`serde_json`, `std::thread::scope` for baking, `lightcraft-color`
for XYZ ↔ Rec.2020 and Bradford adaptation. A Python generator (`colour-science`, rawtoaces-data) produces the CIE
tables once.

**Spec:** There is no separate design document. The issue is
[storytold/lightcraft#988](https://github.com/storytold/lightcraft/issues/988), and the design, the licence analysis
and the data checks are in **Background and decisions** below. Executors read that section first.

---

## Background and decisions

### Licence analysis (done 2026-10-11)

| Material | Licence | Use in LightCraft |
|---|---|---|
| spektrafilm code (Python), spektrafilm-rs (Rust), spektrafilm-ofx (C++), darktable's module, ART's bridge | GPL-3.0 | **Never read, port or copy** (CLAUDE.md:60). |
| vkdt `filmsim` (BSD-2) | BSD-2-Clause, but it reimplements spektrafilm | **Don't read it either**: we write our own model from published colour science, so the provenance stays clean. |
| spektrafilm profiles `src/spektrafilm/data/profiles/*.json` | CC BY-SA 4.0 (`SPEKTRAFILM_LICENSE.txt`) | **Vendored unmodified** under `assets/film/spektrafilm/`. CC BY-SA is allowed with attribution (CLAUDE.md:80). |

You may read the spektrafilm README, the pixls.us write-up "Spectral film simulations from scratch", Hunt *The
Reproduction of Colour* ch. 15, Giorgianni & Madden *Digital Color Management*, Kodak and Fujifilm datasheets, and
the profile JSON files themselves.

**CC BY-SA obligations**, which Task 2 and Task 10 implement:
- Credit "spektrafilm by Andrea Volpato", link https://github.com/andreavolpato/spektrafilm and include the licence
  notice. These go with the files, in NOTICE and in the About dialog.
- Mark any modification. We ship the files unmodified.
- Keep the profiles under CC BY-SA. Our code stays MIT OR Apache-2.0; share-alike covers only the data.

**The author's non-binding requests:**
- Don't use "spektrafilm" in product branding. The UI says "Film Simulation"; credits may say "film data: spektrafilm".
- Don't resell the data.
- Don't train commercial AI on it.

### Decisions for the maintainer before Task 2 (execution-plan §7: the user's call)

1. **Ship CC BY-SA data in the app?** It would be the first CC BY-SA asset in the repo. **Recommended: yes.** It is
   allowed by CLAUDE.md:80; the share-alike covers only the JSON.
2. **Contact Andrea Volpato first?** The README invites projects that can't use GPL to get in touch, and it claims
   that anything "directly inspired by its methods" is a derivative. That claim isn't legally sound, but it is the
   author's wish. **Recommended:** open a short, friendly discussion on the spektrafilm repo before release, saying we
   use the profiles and wrote our own model.
3. **Which stocks ship in phase 1?** Recommended (≈1.04 MB of JSON, embedded in native and wasm builds):
   - `kodak_portra_400`, `kodak_ektar_100`, `kodak_gold_200` (colour negatives);
   - `fujifilm_velvia_100` (slide);
   - `kodak_portra_endura` (paper).

   More stocks are a follow-up plan.

If the maintainer says no to (1), stop after Task 1 and report.

### The model (our own; every formula is below)

- **Upsampling (Task 3).**
  - Three smooth reflectance spectra `b_r, b_g, b_b` that sum to 1. Under D65 their colours are the Rec.2020
    primaries.
  - They are found as the smoothest such set: least squared first differences, solved once in closed form via KKT.
  - A linear Rec.2020 colour `(r, g, b)` becomes the reflectance `r·b_r + g·b_g + b·b_b`.
- **Exposure (Task 5).**
  - Layer exposure is `E_l = Σ_λ illum_ref(λ) · reflectance(λ) · sens_l(λ)`, which is linear in RGB, so it becomes a
    3×3 matrix per stock.
  - Rows are scaled so that 18.4 % grey gives `E_l = 1` (log10 0).
  - **Measured:** log-exposure 0 sits mid-scale on the straight line of every shipped curve. Portra 400 densities at 0
    are 0.68/0.70/0.86 (slope ≈0.5); Velvia 0.51/0.54/0.55.
- **Film density (Task 5).** `D_l = curve_l(log10 E_l · gain)`, read from the profile's `density_curves` (density above
  base).
- **Negative → paper (Task 5).**
  - Negative transmittance: `T(λ) = 10^-(base(λ) + Σ D_i·dye_i(λ))`.
    - **Measured:** `base + Σ D(0)·dye` reproduces each profile's `midscale_neutral_density` with RMS 0.085 (Portra
      400), 0.02 (Endura) and 0.012 (Velvia). So the dye spectra are per unit density and need no rescaling.
  - Paper log exposure: `log10 Σ_λ lamp(λ)·T(λ)·sens_paper_j(λ)`.
  - The enlarger lamp is our own approximation: a 3200 K blackbody with a logistic roll-off above 700 nm. It is not
    measured filter data.
- **Neutral print balance (Task 5).** Per-layer log offsets are chosen so the scanned print of an 18.4 % grey is
  neutral and keeps its brightness. This is a Newton solve over three unknowns with a numeric Jacobian. It is an
  ideal enlarger, not a dichroic filter model.
- **Print exposure and filtration (Task 5).**
  - User offsets on top of the balance: print exposure in EV, all layers.
  - Magenta CC on the green-sensitive layer and yellow CC on the blue-sensitive layer, at 1 CC = 0.01 density.
- **Scan / view (Task 5).**
  - Reflectance (paper) or transmittance (slide) is `10^-(base + Σ D·dye)`, seen under D50, giving XYZ.
  - Bradford-adapt from the unexposed base's white to D65, then go to linear Rec.2020, with the base mapped to 1.0.
  - The paper base is display white.
- **Baking (Task 4/5).**
  - The densities → paper-log-exposure grid depends only on (film, paper).
  - The paper densities → Rec.2020 grid depends only on the paper; a slide's densities → Rec.2020 grid only on the
    slide.
  - Both are baked once and cached, so dragging a film slider never rebakes.
- **Verified channel order** (from the data):
  - `log_sensitivity` columns are R, G, B (peaks 645/545/400 nm on Portra 400).
  - `channel_density` columns are C, M, Y (peaks 695/540/450 nm).
  - Slide curves fall with exposure; negative and paper curves rise.

### Profile JSON shape (from `kodak_portra_400.json`, `kodak_portra_endura.json`, `fujifilm_velvia_100.json`)

```text
metadata: {version, copyright, created, license, citation, datasource}
info: {stock, name, type: negative|positive, support: film|paper, target_print: id|null,
       reference_illuminant: "D55" | "TH-KG3" | …, viewing_illuminant, densitometer, …}
data: wavelengths[81] (380..780 step 5)
      log_sensitivity[81][3]   (log10; never null in the shipped files)
      channel_density[81][3]   (null outside the measured band)
      base_density[81]         (may contain null)
      midscale_neutral_density[81] (may contain null)
      log_exposure[256]        (-3.0 .. 4.0)
      density_curves[256][3]
      (+ density_curves_layers, density_curves_model, hanatos2025_*: ignored)
```

Missing samples are JSON `null`, never `NaN`. serde_json rejects a `NaN` literal, which is fine: it becomes an error,
not a panic.

---

## Global Constraints

- **Never crash.** Every new production crate root carries
  `#![forbid(unsafe_code)]` and
  `#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]`.
  - No `unwrap`, `expect` or `panic` outside tests.
  - Input-derived numbers are hostile: profile JSON, pixel values and settings use checked or clamped math, and NaN or
    inf never reach an index.
  - Locks use `lock().unwrap_or_else(PoisonError::into_inner)`; worker joins are `Result`s.
- **Clean-room.** Don't open any GPL film-simulation source, or vkdt's filmsim (see the table above).
- **Pure Rust.** No new C/C++ dependency. `lightcraft-film` depends only on `lightcraft-color`, `serde` and
  `serde_json`.
- **Layering.** `lightcraft-film` is L1 (`xtask/src/layers.rs` TABLE). `pipeline` (L2) and `engine` (L4) depend on it.
  Nothing below L5 uses egui.
- **Everything is a command.** Film on/off and stock/paper go through `develop.film`; sliders are `develop` control
  specs (`film.*`).
- **Old edits are untouched.** `DevelopSettings.film` is skipped when serialising while at its default, so existing
  settings JSON, `hash64` and preview caches don't change (the same pattern as `hdr`).
- **Assets.** Every vendored file has an `assets/ATTRIBUTION.md` row, added in the same commit; the licence file sits
  next to the data.
- **Build hygiene.** Use `CARGO_TARGET_DIR=target/agent-988` for every cargo command, and delete it when done. Cold
  builds take more than 10 minutes, so run them in the background.
- **Commits.** One task per commit, message `film (#988): <what>`, ending with the environment's Co-Authored-By line.
  Run `cargo xtask ci` before each commit and only commit green states.
- **Clippy is `-D warnings`.** Write index loops as iterators or `std::array::from_fn` (clippy `needless_range_loop`).
  The code below already does.

## Review Focus

1. **Edits saved before this feature** (no `film` key in the settings or sidecar JSON) must render pixel-identically
   and keep their settings hash. Task 6 pins the JSON; Task 8 pins the render.
2. **Hostile pixels** must give finite output in 0..1, never a panic: NaN, ±inf, negatives, and HDR highlights at 1e6
   (from Exposure +5 on a bright raw). Task 4 covers the grids and Task 5 the look.
3. **A bad stock id** must render without film, not crash. A hand-edited sidecar can name an unknown stock, or a
   paper as the film. The `develop.film` command refuses such ids with an actionable error. Tasks 5, 7 and 8.
4. **Dragging any film slider must not rebake** the grids. The (film, paper) cache must return the same `Arc`. Task 5.
5. **Film combined with HDR editing, B&W treatment, or the Film section's eye switched off.** HDR stays at most SDR
   white inside the film stage, B&W still converts, and the eye off renders exactly as without film. Task 8.

---

## File structure

| File | Responsibility |
|---|---|
| `crates/film/Cargo.toml` | New crate `lightcraft-film` (L1). |
| `crates/film/src/lib.rs` | Crate attributes, module list, re-exports, `FilmError`, `CREDITS`. |
| `crates/film/src/cie_table.rs` | Generated: CIE 1931 CMFs and the CIE daylight basis S0/S1/S2 on the 81-sample grid. |
| `crates/film/src/spectrum.rs` | The 81-sample grid, XYZ, Planck, CIE daylight, illuminants, enlarger lamp. |
| `crates/film/src/profile.rs` | `Stock`, `Kind`, `Curve`; validated parse of the profile JSON. |
| `crates/film/src/stocks.rs` | The built-in stocks (`include_str!`), lookup, lists. |
| `crates/film/src/basis.rs` | The Rec.2020 → reflectance upsampling basis (KKT solve). |
| `crates/film/src/grid.rs` | `Table1` (1-D uniform table) and `Grid3` (baked 3-D grid, trilinear). |
| `crates/film/src/look.rs` | `LookParams`, `Look`, baking, neutral balance, cache, `Look::apply`. |
| `tools/film_cie_table.py` | Generator for `cie_table.rs`. |
| `assets/film/spektrafilm/*.json`, `SPEKTRAFILM_LICENSE.txt`, `README.md` | Vendored stock data and its licence. |
| `crates/develop/src/settings.rs` | `Film` settings struct, field on `DevelopSettings`. |
| `crates/develop/src/controls.rs` | `Section::Film`, `film.*` controls. |
| `crates/develop/src/presets.rs` | `SettingsGroup::Film`. |
| `crates/engine/src/cmd/develop.rs` | `develop.film`, `develop.filmStocks` commands. |
| `crates/engine/src/tests_film.rs` | Engine tests for the commands. |
| `crates/pipeline/src/finish.rs` | `FinishParams.film`; the film stage in place of the tone map. |
| `crates/pipeline/src/tests_film.rs` | Pipeline render tests. |
| `crates/engine/src/media.rs` | GPU path declines film renders. |
| `crates/ui-egui/src/panels/edit.rs` | The Film section. |
| `crates/ui-egui/src/panels/dialogs.rs` | About ▸ Data tab showing `lightcraft_film::CREDITS`. |
| `docs/film-simulation.md` | User and developer documentation, model, credits. |
| `assets/ATTRIBUTION.md`, `NOTICE`, `docs/parity-checklist.md`, `docs/gaps.md`, `ROADMAP.md`, `docs/whats-new.md` | Attribution and progress docs. |

---

### Task 1: `lightcraft-film` crate: spectral grid, CIE tables, illuminants

**Files:**
- Create: `crates/film/Cargo.toml`, `crates/film/src/lib.rs`, `crates/film/src/spectrum.rs`, `crates/film/src/cie_table.rs` (generated), `tools/film_cie_table.py`
- Modify: `Cargo.toml` (workspace deps, profile opt-level), `xtask/src/layers.rs` (TABLE), `assets/ATTRIBUTION.md`, `NOTICE`

**Interfaces:**
- Produces:
  - `lightcraft_film::spectrum::{N: usize = 81, WL0, STEP, Spectrum = [f64; 81]}`
  - `wavelength(i) -> f64`, `dot(&S, &S) -> f64`, `mul(&S, &S) -> S`, `xyz(&S) -> [f64; 3]`, `xy(&S) -> Option<(f64, f64)>`
  - `planck(K) -> S`, `daylight(cct) -> S`, `illuminant(&str) -> S`, `enlarger_lamp() -> S`
  - `lightcraft_film::FilmError`
  - `pub(crate) cie_table::{CMF_X, CMF_Y, CMF_Z, DAYLIGHT_S0, DAYLIGHT_S1, DAYLIGHT_S2}`

- [ ] **Step 1: Create the worktree build dir and the crate manifest**

`crates/film/Cargo.toml`:

```toml
[package]
name = "lightcraft-film"
version.workspace = true
edition.workspace = true
license.workspace = true
description = "LightCraft film simulation: spectral film, print and scan model fed by measured film-stock data (L1)"

[lints]
workspace = true

[dependencies]
lightcraft-color = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
```

Root `Cargo.toml`:
- Next to `lightcraft-color = { path = "crates/color" }` in `[workspace.dependencies]`, add
  `lightcraft-film = { path = "crates/film" }`.
- Next to `lightcraft-color = { opt-level = 3 }` (the dev-profile overrides, around line 152), add
  `lightcraft-film = { opt-level = 3 }`.

In `xtask/src/layers.rs` `TABLE`, add `("film", Class::Layer(1)),` after `("develop", Class::Layer(1)),`.

- [ ] **Step 2: Write the generator `tools/film_cie_table.py`**

```python
# Writes crates/film/src/cie_table.rs: the CIE 1931 2° colour-matching functions and the CIE daylight
# basis functions S0, S1, S2, sampled 380..780 nm every 5 nm, for lightcraft-film.
#
#   git clone https://github.com/AcademySoftwareFoundation/rawtoaces-data ../rawtoaces-data
#   git -C ../rawtoaces-data checkout v1.1.0
#   python3 -m venv .venv-spectral && .venv-spectral/bin/pip install -r tools/requirements-spectral-camera-table.txt
#   .venv-spectral/bin/python tools/film_cie_table.py ../rawtoaces-data crates/film/src/cie_table.rs
#
# The colour-matching functions are rawtoaces-data's copy of the CIE 1931 observer (CIE, CC-BY-4.0); the daylight
# basis is CIE 15:2004 Table T.2 as shipped by colour-science (BSD-3-Clause).
import json
import subprocess
import sys

import colour
import numpy as np

WL = np.arange(380, 781, 5)


def cmf(root):
    data = json.load(open(f"{root}/data/cmf/cmf_1931.json"))["spectral_data"]["data"]["main"]
    return np.array([[data[str(w)][i] for i in range(3)] for w in WL], dtype=float)


def daylight_basis():
    basis = colour.colorimetry.SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES
    return [np.array([basis[k][w] for w in WL], dtype=float) for k in ("S0", "S1", "S2")]


def const(name, values, doc):
    body = ", ".join(f"{v:.6g}" for v in values)
    return f"/// {doc}\n#[rustfmt::skip]\npub(crate) const {name}: Spectrum = [{body}];\n"


def main():
    root, out = sys.argv[1], sys.argv[2]
    commit = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    m = cmf(root)
    s0, s1, s2 = daylight_basis()
    parts = [
        "// Generated by tools/film_cie_table.py. Do not edit by hand.\n"
        f"// CIE 1931 2° colour-matching functions: rawtoaces-data (https://github.com/AcademySoftwareFoundation/rawtoaces-data,\n"
        f"// commit {commit}), CIE, CC-BY-4.0. CIE daylight basis S0/S1/S2: CIE 15:2004 Table T.2 via colour-science\n"
        f"// {colour.__version__} (BSD-3-Clause). See NOTICE and assets/ATTRIBUTION.md.\n"
        "use crate::spectrum::Spectrum;\n",
        const("CMF_X", m[:, 0], "CIE 1931 x̄."),
        const("CMF_Y", m[:, 1], "CIE 1931 ȳ."),
        const("CMF_Z", m[:, 2], "CIE 1931 z̄."),
        const("DAYLIGHT_S0", s0, "CIE daylight mean S0."),
        const("DAYLIGHT_S1", s1, "CIE daylight first characteristic vector S1."),
        const("DAYLIGHT_S2", s2, "CIE daylight second characteristic vector S2."),
    ]
    open(out, "w").write("\n".join(parts))


main()
```

If `colour.colorimetry.SDS_BASIS_FUNCTIONS_CIE_ILLUMINANT_D_SERIES` is missing in the pinned colour version, find the
name with `python -c "import colour; print([n for n in dir(colour.colorimetry) if 'D_SERIES' in n])"`. Use that name
and record it in the script.

- [ ] **Step 3: Generate the table**

Run, each as its own command:

```bash
git clone https://github.com/AcademySoftwareFoundation/rawtoaces-data ../rawtoaces-data
git -C ../rawtoaces-data checkout v1.1.0
python3 -m venv .venv-spectral
.venv-spectral/bin/pip install -r tools/requirements-spectral-camera-table.txt
.venv-spectral/bin/python tools/film_cie_table.py ../rawtoaces-data crates/film/src/cie_table.rs
```

Expected: `crates/film/src/cie_table.rs` has six 81-value constants. Don't commit `.venv-spectral` (check that
`git status` doesn't show it; add it to `.gitignore` if it does).

- [ ] **Step 4: Write `lib.rs` and the failing tests in `spectrum.rs`**

`crates/film/src/lib.rs`:

```rust
//! Film simulation (issue #988): a scene-linear Rec.2020 colour is exposed onto a virtual film stock (measured
//! spectral sensitivities and characteristic curves), printed onto a virtual paper and scanned under D50, or — for
//! slide film — viewed directly on a D50 light box. The model is our own (`docs/film-simulation.md`); the stock data
//! are spektrafilm profiles by Andrea Volpato, CC BY-SA 4.0, vendored unmodified in `assets/film/spektrafilm/`.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod cie_table;
pub mod spectrum;

/// Why a film look can't be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FilmError {
    /// A stock's data are malformed (id or "profile", and what is wrong).
    Profile(String),
    /// No stock with this id ships with LightCraft.
    UnknownStock(String),
    /// The paper id names a film, or a negative has no paper.
    NotAPaper(String),
    /// The film id names a paper.
    NotAFilm(String),
}

impl std::fmt::Display for FilmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilmError::Profile(m) => write!(f, "film data: {m}"),
            FilmError::UnknownStock(id) => write!(f, "unknown film stock `{id}`"),
            FilmError::NotAPaper(id) => write!(f, "`{id}` is not a print paper"),
            FilmError::NotAFilm(id) => write!(f, "`{id}` is a print paper, not a film"),
        }
    }
}

impl std::error::Error for FilmError {}
```

`crates/film/src/spectrum.rs` (implementation from the **Background → The model** section; tests first):

```rust
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
```

Add `pub mod spectrum;` (already in `lib.rs` above) and keep `mod cie_table;`.

- [ ] **Step 5: Run the tests**

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-film`
Expected: 6 passed.
- If `cmf_y_peaks_at_555_nm` fails, the generator read the wrong column.
- If the D65 test fails by more than 0.001, check that M1/M2 are rounded and that the S1/S2 signs match CIE 15.

- [ ] **Step 6: Attribution and NOTICE**

Add to `assets/ATTRIBUTION.md`, after the `crates/raw/src/spectral_table.rs` row:

```markdown
| `crates/film/src/cie_table.rs` | CIE 1931 2° colour-matching functions and the CIE daylight basis functions S0/S1/S2, 380–780 nm every 5 nm | CIE (Commission Internationale de l'Éclairage); colour-matching functions as shipped in rawtoaces-data (Academy Software Foundation); daylight basis as shipped in colour-science (Colour Developers) | https://github.com/AcademySoftwareFoundation/rawtoaces-data (v1.1.0), https://github.com/colour-science/colour | colour-matching functions CC-BY-4.0; daylight basis BSD-3-Clause (colour-science) / CIE 15:2004 Table T.2 | 2026-10-11 | generated by `tools/film_cie_table.py`: resampled to the 5 nm grid, values rounded to 6 significant digits |
```

Add to `NOTICE`, after the `crates/raw/src/spectral_table.rs` paragraph:

```text
crates/film/src/cie_table.rs (lightcraft-film) holds the CIE 1931 2° colour-matching functions as
shipped in rawtoaces-data (CIE, CC-BY-4.0) and the CIE daylight basis functions S0, S1, S2 (CIE 15:2004)
as shipped in colour-science (BSD-3-Clause), generated with tools/film_cie_table.py. See
assets/ATTRIBUTION.md.
```

- [ ] **Step 7: Gate and commit**

Run: `CARGO_TARGET_DIR=target/agent-988 cargo xtask ci`. Expected: green. The layers check accepts the L1 crate.

```bash
git add crates/film Cargo.toml Cargo.lock xtask/src/layers.rs tools/film_cie_table.py assets/ATTRIBUTION.md NOTICE
git commit -m "film (#988): lightcraft-film crate with the spectral grid, CIE tables and illuminants"
```

---

### Task 2: Vendored stocks, validated profile parser, built-in registry

**Files:**
- Create: `assets/film/spektrafilm/{kodak_portra_400,kodak_ektar_100,kodak_gold_200,fujifilm_velvia_100,kodak_portra_endura}.json`, `assets/film/spektrafilm/SPEKTRAFILM_LICENSE.txt`, `assets/film/spektrafilm/README.md`, `crates/film/src/profile.rs`, `crates/film/src/stocks.rs`
- Modify: `crates/film/src/lib.rs`, `assets/ATTRIBUTION.md`, `NOTICE`

**Interfaces:**
- Consumes: `spectrum::{N, Spectrum, wavelength}`, `FilmError`.
- Produces:
  - `lightcraft_film::{Kind (Negative|Positive|Paper, Serialize lowercase), Stock, Curve}`
  - `Stock::parse(&str) -> Result<Stock, FilmError>`
  - `Curve::{eval(f64) -> f64, range() -> (f64, f64), domain() -> (f64, f64)}`
  - `lightcraft_film::{stock(&str) -> Option<&'static Stock>, films() -> Vec<StockInfo>, papers() -> Vec<StockInfo>, StockInfo {id, name, kind}, DEFAULT_PAPER}`
  - `Stock` fields: `id, name, kind, reference_illuminant, target_print, sensitivity: [Spectrum; 3]` (linear),
    `dye: [Spectrum; 3]` (C, M, Y; nulls → 0), `base: Spectrum` (nulls → 0), `curves: [Curve; 3]`.

- [ ] **Step 1: Vendor the files unmodified** (only after maintainer decision 1 is "yes")

Pin to one spektrafilm commit and record it. Run each command on its own, from the worktree root:

```bash
gh api repos/andreavolpato/spektrafilm/commits/main --jq .sha
```

Note the SHA (call it `<SHA>`), then for each of the five ids and the licence:

```bash
mkdir -p assets/film/spektrafilm
curl -sSfL -o assets/film/spektrafilm/kodak_portra_400.json https://raw.githubusercontent.com/andreavolpato/spektrafilm/<SHA>/src/spektrafilm/data/profiles/kodak_portra_400.json
curl -sSfL -o assets/film/spektrafilm/kodak_ektar_100.json https://raw.githubusercontent.com/andreavolpato/spektrafilm/<SHA>/src/spektrafilm/data/profiles/kodak_ektar_100.json
curl -sSfL -o assets/film/spektrafilm/kodak_gold_200.json https://raw.githubusercontent.com/andreavolpato/spektrafilm/<SHA>/src/spektrafilm/data/profiles/kodak_gold_200.json
curl -sSfL -o assets/film/spektrafilm/fujifilm_velvia_100.json https://raw.githubusercontent.com/andreavolpato/spektrafilm/<SHA>/src/spektrafilm/data/profiles/fujifilm_velvia_100.json
curl -sSfL -o assets/film/spektrafilm/kodak_portra_endura.json https://raw.githubusercontent.com/andreavolpato/spektrafilm/<SHA>/src/spektrafilm/data/profiles/kodak_portra_endura.json
curl -sSfL -o assets/film/spektrafilm/SPEKTRAFILM_LICENSE.txt https://raw.githubusercontent.com/andreavolpato/spektrafilm/<SHA>/SPEKTRAFILM_LICENSE.txt
shasum -a 256 assets/film/spektrafilm/*
```

Write `assets/film/spektrafilm/README.md`:

```markdown
# Film stock data

These five files are **spektrafilm** profiles by **Andrea Volpato**, copied **unmodified** from
https://github.com/andreavolpato/spektrafilm (commit `<SHA>`, `src/spektrafilm/data/profiles/`), licensed
**CC BY-SA 4.0** — see `SPEKTRAFILM_LICENSE.txt` (unmodified). Redistribution and derivatives must credit the
author, link the project, keep this licence and stay CC BY-SA 4.0; modifications must be noted in a `CHANGELOG.txt`
next to them. The licence covers these data files only; LightCraft's code is MIT OR Apache-2.0.

Datasource (per the files' metadata): processed from Kodak and Fujifilm data sheets and scientific publications;
original measurement data are property of their holders.

LightCraft reads them with its own film model (`crates/film`, `docs/film-simulation.md`); no spektrafilm code is
used.

| File | SHA-256 |
|---|---|
| kodak_portra_400.json | `<sha>` |
| kodak_ektar_100.json | `<sha>` |
| kodak_gold_200.json | `<sha>` |
| fujifilm_velvia_100.json | `<sha>` |
| kodak_portra_endura.json | `<sha>` |
| SPEKTRAFILM_LICENSE.txt | `<sha>` |
```

Fill in `<SHA>` and the `<sha>` values from the commands above.

- [ ] **Step 2: Write the failing tests for the parser** (end of `crates/film/src/profile.rs`)

```rust
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
```

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-film profile`
Expected: FAIL to compile (module missing).

- [ ] **Step 3: Implement `crates/film/src/profile.rs`** (put it above the tests)

```rust
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
        for (c, v) in row.iter().enumerate() {
            out[c][i] = checked(id, what, v.unwrap_or(fill))?;
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
        if d.log_exposure.windows(2).any(|w| w[1] <= w[0]) {
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
```

- [ ] **Step 4: Write `crates/film/src/stocks.rs` with its tests**

```rust
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

    /// The model's assumption: dye spectra are per unit layer density, so base + Σ D(0)·dye is the mid-scale
    /// neutral the data sheet measured.
    #[test]
    fn dyes_reproduce_the_midscale_neutral() {
        for j in BUILTIN {
            let s = Stock::parse(j).unwrap();
            let v: serde_json::Value = serde_json::from_str(j).unwrap();
            let mid = v["data"]["midscale_neutral_density"].as_array().unwrap();
            let d0: Vec<f64> = s.curves.iter().map(|c| c.eval(0.0)).collect();
            let errs: Vec<f64> = (0..N)
                .filter_map(|i| {
                    let m = mid[i].as_f64()?;
                    Some(s.base[i] + d0[0] * s.dye[0][i] + d0[1] * s.dye[1][i] + d0[2] * s.dye[2][i] - m)
                })
                .collect();
            let rms = (errs.iter().map(|e| e * e).sum::<f64>() / errs.len() as f64).sqrt();
            assert!(errs.len() > 40 && rms < 0.15, "{}: rms {rms} over {}", s.id, errs.len());
        }
    }
}
```

In `lib.rs` add:

```rust
mod profile;
mod stocks;

pub use profile::{Curve, Kind, Stock};
pub use stocks::{DEFAULT_PAPER, StockInfo, films, papers, stock};
```

- [ ] **Step 5: Run the tests**

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-film`
Expected: all pass.
- If `dyes_reproduce_the_midscale_neutral` fails for Ektar or Gold, the RMS for Portra/Endura/Velvia was measured at
  0.085/0.02/0.012. Report the failing stock's RMS and leave it out of `BUILTIN` rather than raising the tolerance.

- [ ] **Step 6: Attribution and NOTICE**

`assets/ATTRIBUTION.md` row (fill in `<SHA>`):

```markdown
| `assets/film/spektrafilm/` (five `*.json` stock profiles, `SPEKTRAFILM_LICENSE.txt`, `README.md`) | Film and print-paper stock data: Kodak Portra 400, Ektar 100, Gold 200, Fujifilm Velvia 100, Kodak Portra Endura (spectral sensitivities, dye densities, characteristic curves) | spektrafilm by Andrea Volpato; processed from Kodak and Fujifilm data sheets and scientific publications | https://github.com/andreavolpato/spektrafilm (commit `<SHA>`, `src/spektrafilm/data/profiles/`) | CC BY-SA 4.0 (`assets/film/spektrafilm/SPEKTRAFILM_LICENSE.txt`) | 2026-10-11 | none (unmodified); `README.md` added by LightCraft contributors |
```

`NOTICE` paragraph:

```text
assets/film/spektrafilm/ holds film and print-paper stock profiles from spektrafilm by Andrea Volpato
(https://github.com/andreavolpato/spektrafilm), unmodified, licensed under CC BY-SA 4.0 (see
assets/film/spektrafilm/SPEKTRAFILM_LICENSE.txt). They are compiled into lightcraft-film. This licence
covers those data only; LightCraft's own code, including its film model, is MIT OR Apache-2.0 and
uses no spektrafilm code.
```

- [ ] **Step 7: Gate and commit**

Run: `CARGO_TARGET_DIR=target/agent-988 cargo xtask ci`. Expected: green. `xtask assets` finds the licence file named in
the Licence column.

```bash
git add assets/film crates/film assets/ATTRIBUTION.md NOTICE
git commit -m "film (#988): vendor five spektrafilm stock profiles (CC BY-SA 4.0) and a validated parser"
```

---

### Task 3: Spectral upsampling basis

**Files:**
- Create: `crates/film/src/basis.rs`
- Modify: `crates/film/src/lib.rs` (`mod basis;`)

**Interfaces:**
- Consumes: `spectrum::{N, Spectrum, daylight, xyz, mul}`, `cie_table::{CMF_X, CMF_Y, CMF_Z}`, `lightcraft_color::REC2020`.
- Produces: `pub(crate) fn basis::rec2020() -> &'static [Spectrum; 3]` (reflectance per unit r, g, b; they sum to 1).

- [ ] **Step 1: Write the failing tests** (in `basis.rs`)

```rust
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
```

(The blue primary is `1 − b_r − b_g`, so its XYZ matches only as far as `daylight(6504)` matches Rec.2020's D65 white,
within about 5e-4. Hence the looser tolerance.)

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-film basis`
Expected: FAIL (doesn't compile).

- [ ] **Step 2: Implement `basis.rs`** (above the tests)

```rust
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
fn gauss_jordan(mut a: Vec<Vec<f64>>) -> Option<Vec<f64>> {
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
```

Add to `lib.rs` (the allow keeps `clippy -D warnings` green until Task 5 uses the module; Task 5 removes it):

```rust
#[allow(dead_code)] // used by `look` from Task 5 on
mod basis;
```

- [ ] **Step 3: Run the tests**

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-film basis`. Expected: 4 passed.

- [ ] **Step 4: Gate and commit**

Run `cargo xtask ci` (with `CARGO_TARGET_DIR=target/agent-988`), then:

```bash
git add crates/film
git commit -m "film (#988): smooth Rec.2020 spectral upsampling basis"
```

---

### Task 4: Lookup tables (`Table1`, `Grid3`)

**Files:**
- Create: `crates/film/src/grid.rs`
- Modify: `crates/film/src/lib.rs` (`mod grid;`)

**Interfaces:**
- Consumes: `FilmError`.
- Produces:
  - `pub(crate) struct Table1` with `from_fn(lo: f64, hi: f64, n: usize, f: impl Fn(f64) -> f64) -> Table1` and
    `eval(f32) -> f32`.
  - `pub(crate) struct Grid3` with
    `bake(n: usize, lo: [f64; 3], hi: [f64; 3], f: impl Fn([f64; 3]) -> [f64; 3] + Sync) -> Result<Grid3, FilmError>`
    and `sample([f32; 3]) -> [f32; 3]`.

- [ ] **Step 1: Write the failing tests** (in `grid.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table1_interpolates_and_clamps() {
        let t = Table1::from_fn(0.0, 2.0, 3, |x| x * x);
        assert_eq!(t.eval(0.0), 0.0);
        assert_eq!(t.eval(0.5), 0.5);
        assert_eq!(t.eval(1.5), 2.5);
        assert_eq!(t.eval(-3.0), 0.0);
        assert_eq!(t.eval(7.0), 4.0);
        assert_eq!(t.eval(f32::NAN), 0.0);
        assert_eq!(t.eval(f32::INFINITY), 4.0);
        assert_eq!(t.eval(f32::NEG_INFINITY), 0.0);
    }

    #[test]
    fn table1_survives_a_degenerate_range_and_bad_values() {
        let t = Table1::from_fn(1.0, 1.0, 0, |_| f64::NAN);
        assert_eq!(t.eval(1.0), 0.0);
    }

    #[test]
    fn grid3_is_exact_for_linear_functions() {
        let g = Grid3::bake(9, [0.0, -1.0, 2.0], [1.0, 1.0, 4.0], |p| [p[0] + 2.0 * p[1], p[2], -p[0]]).unwrap();
        for p in [[0.3f32, 0.2, 2.5], [0.0, -1.0, 2.0], [1.0, 1.0, 4.0], [0.77, -0.31, 3.9]] {
            let v = g.sample(p);
            let e = [p[0] + 2.0 * p[1], p[2], -p[0]];
            for c in 0..3 {
                assert!((v[c] - e[c]).abs() < 1e-5, "{p:?}: {v:?} vs {e:?}");
            }
        }
    }

    #[test]
    fn grid3_clamps_hostile_input() {
        let g = Grid3::bake(5, [0.0; 3], [1.0; 3], |p| p).unwrap();
        for p in [[f32::NAN; 3], [f32::INFINITY; 3], [-1e30; 3], [1e30, -1e30, 0.5]] {
            assert!(g.sample(p).iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)), "{p:?}");
        }
        assert_eq!(g.sample([2.0, 2.0, 2.0]), [1.0, 1.0, 1.0]);
    }

    #[test]
    fn grid3_replaces_non_finite_results_and_caps_size() {
        let g = Grid3::bake(1000, [0.0; 3], [1.0; 3], |_| [f64::NAN, f64::INFINITY, 1.0]).unwrap();
        assert_eq!(g.n, MAX_GRID);
        assert_eq!(g.sample([0.5; 3]), [0.0, 0.0, 1.0]);
    }
}
```

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-film grid`. Expected: FAIL (doesn't compile).

- [ ] **Step 2: Implement `grid.rs`** (above the tests)

```rust
//! Baked lookup tables: [`Table1`] (a uniform 1-D table, for characteristic curves) and [`Grid3`] (a 3-D grid with
//! trilinear interpolation, for the spectral stages). Lookups clamp to the table and turn NaN into the low end.

use crate::FilmError;

/// Largest grid side (65³ samples ≈ 3.3 MB).
pub(crate) const MAX_GRID: usize = 65;

#[derive(Clone, Debug)]
pub(crate) struct Table1 {
    x0: f32,
    inv_dx: f32,
    ys: Vec<f32>,
}

impl Table1 {
    /// `f` sampled at `n` (2..65536) points over `lo..=hi`; non-finite values become 0.
    pub(crate) fn from_fn(lo: f64, hi: f64, n: usize, f: impl Fn(f64) -> f64) -> Table1 {
        let n = n.clamp(2, 1 << 16);
        let dx = ((hi - lo) / (n - 1) as f64).max(1e-9);
        let ys = (0..n)
            .map(|i| {
                let v = f(lo + dx * i as f64);
                if v.is_finite() { v as f32 } else { 0.0 }
            })
            .collect();
        Table1 { x0: lo as f32, inv_dx: (1.0 / dx) as f32, ys }
    }

    pub(crate) fn eval(&self, x: f32) -> f32 {
        let last = self.ys.len().saturating_sub(1);
        let t = (x - self.x0) * self.inv_dx;
        let t = if t.is_finite() {
            t.clamp(0.0, last as f32)
        } else if t > 0.0 {
            last as f32
        } else {
            0.0
        };
        let i = (t as usize).min(last.saturating_sub(1));
        let a = self.ys.get(i).copied().unwrap_or(0.0);
        let b = self.ys.get(i + 1).copied().unwrap_or(a);
        a + (b - a) * (t - i as f32)
    }
}

pub(crate) struct Grid3 {
    pub(crate) n: usize,
    lo: [f32; 3],
    inv_step: [f32; 3],
    /// `n³` samples, x fastest: index `(z·n + y)·n + x`.
    data: Vec<[f32; 3]>,
}

impl Grid3 {
    /// `f` sampled on an `n³` grid (n clamped to 2..=[`MAX_GRID`]) spanning `lo..=hi`, on all cores. Non-finite
    /// results become 0.
    pub(crate) fn bake(n: usize, lo: [f64; 3], hi: [f64; 3], f: impl Fn([f64; 3]) -> [f64; 3] + Sync) -> Result<Grid3, FilmError> {
        let n = n.clamp(2, MAX_GRID);
        let step: [f64; 3] = std::array::from_fn(|c| ((hi[c] - lo[c]) / (n - 1) as f64).max(1e-9));
        let mut data = vec![[0.0f32; 3]; n * n * n];
        let threads = std::thread::available_parallelism().map_or(1, |v| v.get()).clamp(1, n);
        let per = n.div_ceil(threads) * n * n;
        let f = &f;
        let ok = std::thread::scope(|s| {
            let handles: Vec<_> = data
                .chunks_mut(per)
                .enumerate()
                .map(|(t, chunk)| {
                    s.spawn(move || {
                        for (j, out) in chunk.iter_mut().enumerate() {
                            let idx = t * per + j;
                            let (x, y, z) = (idx % n, (idx / n) % n, idx / (n * n));
                            let p = [lo[0] + x as f64 * step[0], lo[1] + y as f64 * step[1], lo[2] + z as f64 * step[2]];
                            *out = f(p).map(|v| if v.is_finite() { v as f32 } else { 0.0 });
                        }
                    })
                })
                .collect();
            // every handle joined here (not short-circuiting), so a panicking worker is an error instead of a
            // panic in `scope`
            let joined: Vec<bool> = handles.into_iter().map(|h| h.join().is_ok()).collect();
            joined.iter().all(|&ok| ok)
        });
        if !ok {
            return Err(FilmError::Profile("baking the film look failed".into()));
        }
        Ok(Grid3 { n, lo: lo.map(|v| v as f32), inv_step: step.map(|s| (1.0 / s) as f32), data })
    }

    /// Trilinear lookup, clamped to the grid; NaN reads as the low end.
    pub(crate) fn sample(&self, p: [f32; 3]) -> [f32; 3] {
        let n = self.n;
        let top = (n - 1) as f32;
        let cell: [(usize, f32); 3] = std::array::from_fn(|c| {
            let t = (p[c] - self.lo[c]) * self.inv_step[c];
            let t = if t.is_finite() {
                t.clamp(0.0, top)
            } else if t > 0.0 {
                top
            } else {
                0.0
            };
            let i = (t as usize).min(n - 2);
            (i, t - i as f32)
        });
        let at = |x: usize, y: usize, z: usize| self.data.get((z * n + y) * n + x).copied().unwrap_or([0.0; 3]);
        let mut out = [0.0f32; 3];
        for (dz, wz) in [(0, 1.0 - cell[2].1), (1, cell[2].1)] {
            for (dy, wy) in [(0, 1.0 - cell[1].1), (1, cell[1].1)] {
                for (dx, wx) in [(0, 1.0 - cell[0].1), (1, cell[0].1)] {
                    let w = wx * wy * wz;
                    if w == 0.0 {
                        continue;
                    }
                    let v = at(cell[0].0 + dx, cell[1].0 + dy, cell[2].0 + dz);
                    for (o, vc) in out.iter_mut().zip(v) {
                        *o += w * vc;
                    }
                }
            }
        }
        out
    }
}
```

Add to `lib.rs` (Task 5 removes the allow):

```rust
#[allow(dead_code)] // used by `look` from Task 5 on
mod grid;
```

- [ ] **Step 3: Run the tests**

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-film grid`. Expected: 5 passed.

- [ ] **Step 4: Gate and commit**

```bash
git add crates/film
git commit -m "film (#988): baked 1-D tables and 3-D grids for the film stages"
```

---

### Task 5: The film look: exposure, print, scan, neutral balance, cache

**Files:**
- Create: `crates/film/src/look.rs`
- Modify: `crates/film/src/lib.rs`

**Interfaces:**
- Consumes:
  - `basis::rec2020()`
  - `spectrum::{N, Spectrum, illuminant, enlarger_lamp, daylight, xyz, mul}`
  - `profile::{Stock, Kind, Curve}`, `stocks::{stock, DEFAULT_PAPER}`
  - `grid::{Table1, Grid3}`
  - `lightcraft_color::{REC2020, D65, Mat3, Xy, bradford}`
- Produces:
  - `pub struct LookParams { pub stock: String, pub paper: String, pub exposure_ev: f64, pub print_exposure_ev: f64, pub print_magenta: f64, pub print_yellow: f64 }`
    (derives `Clone, Debug, Default, PartialEq`).
  - `pub struct Look` (derives `Clone`; manual `Debug`).
  - `pub fn look(&LookParams) -> Result<Look, FilmError>`.
  - `Look::apply(&self, [f32; 3]) -> [f32; 3]`: linear Rec.2020 in, display-linear Rec.2020 in 0..1 out.

- [ ] **Step 1: Write the failing tests** (in `look.rs`)

```rust
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
}
```

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-film look`. Expected: FAIL (doesn't compile).

- [ ] **Step 2: Implement `look.rs`** (above the tests)

```rust
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
```

In `lib.rs`:
- add `mod look;` and `pub use look::{Look, LookParams, look};`;
- remove the two `#[allow(dead_code)]` lines from Tasks 3 and 4.

If clippy then reports any item in `basis` or `grid` as unused, delete that item rather than allowing it.

(`Mat3(cols).transpose()`: `cols[c]` is column `c`, so transposing gives the Jacobian with rows as residuals.
`Mat3::inverse` returns `Option<Mat3>`.)

- [ ] **Step 3: Run the tests**

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-film`
Expected: all pass. If a test fails, look at the numbers before changing a tolerance:
- If `mid_grey_prints_neutral` fails, print `neutral_balance`'s iterations; a spread above 0.005 means Newton didn't
  converge.
- If `brighter_scenes…` fails at the top end, the film curve's shoulder is clamping; that's fine only if it is flat,
  not decreasing.

- [ ] **Step 4: Measure baking and per-pixel cost**

Add a release-mode timing test in `look.rs`'s tests, marked `#[ignore]` and run on demand:

```rust
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
```

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test --release -p lightcraft-film bake_and_apply_timing -- --ignored --nocapture`
Expected: bake under 300 ms and 2.5 MP apply under 60 ms single-threaded on an M1-class machine. Record the numbers in
the commit body. If bake exceeds 300 ms, lower `GRID` to 25 and re-run the tests.

- [ ] **Step 5: Gate and commit**

```bash
git add crates/film
git commit -m "film (#988): film look: exposure, print and scan with neutral print balance, baked and cached"
```

---

### Task 6: Develop settings, controls and preset group

**Files:**
- Modify: `crates/develop/src/settings.rs` (new `Film` struct, field on `DevelopSettings` and its `Default`)
- Modify: `crates/develop/src/controls.rs` (`Section::Film`, label, four controls)
- Modify: `crates/develop/src/presets.rs` (`SettingsGroup::Film`)
- Test: `crates/develop/src/settings.rs` tests module (or the crate's existing settings test file; check with
  `grep -n "mod tests" crates/develop/src/settings.rs`)

**Interfaces:**
- Produces:
  - `lightcraft_develop::Film { enabled: bool, stock: String, paper: String, exposure: f64, print_exposure: f64, print_magenta: f64, print_yellow: f64 }`
    with `Film::is_default()`.
  - `DevelopSettings.film`.
  - Control ids `film.exposure`, `film.printExposure`, `film.printMagenta`, `film.printYellow`.
  - `Section::Film`; section id `"film"` for `disabled_sections`.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod film_tests {
    use crate::{DevelopSettings, Film, controls};

    #[test]
    fn settings_without_film_keep_their_json() {
        let j = DevelopSettings::default().to_json();
        assert!(j.get("film").is_none(), "{j}");
        // an edit saved before film simulation reads back with film off
        let old = DevelopSettings::from_json(&j).unwrap();
        assert!(!old.film.enabled && old.film.is_default());
    }

    #[test]
    fn film_settings_round_trip() {
        let mut s = DevelopSettings::default();
        s.film = Film { enabled: true, stock: "kodak_ektar_100".into(), print_magenta: 12.0, ..Film::default() };
        let back = DevelopSettings::from_json(&s.to_json()).unwrap();
        assert_eq!(back.film, s.film);
    }

    #[test]
    fn film_controls_clamp() {
        let mut s = DevelopSettings::default();
        assert!(controls::set(&mut s, "film.printMagenta", 500.0));
        assert_eq!(s.film.print_magenta, 100.0);
        assert!(controls::set(&mut s, "film.exposure", -9.0));
        assert_eq!(s.film.exposure, -3.0);
        assert!(controls::set(&mut s, "film.printExposure", 0.5));
        assert!(controls::set(&mut s, "film.printYellow", -5.0));
    }
}
```

Check the exact names first, and adapt the test calls if they differ (they must exist):
- `grep -n "pub fn to_json\|pub fn from_json" crates/develop/src/*.rs` for the JSON helpers, used by
  `crates/engine/src/sidecar.rs:167`.
- `grep -n "pub fn set" crates/develop/src/controls.rs` for `controls::set`.

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-develop film_tests`. Expected: FAIL (doesn't compile).

- [ ] **Step 2: Add the `Film` struct** to `crates/develop/src/settings.rs`, after `impl Default for Grain`

```rust
/// Film simulation (issue #988): the photo exposed on a film stock, printed and scanned, in place of the tone map
/// (`lightcraft_film`). Not written while it is the default, so settings without it keep their JSON and `hash64`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Film {
    pub enabled: bool,
    /// Film stock id (`lightcraft_film::films`), e.g. `kodak_portra_400`.
    pub stock: String,
    /// Print paper id for a negative; empty = the stock's own paper.
    pub paper: String,
    /// Film exposure, EV −3..3 (after Light ▸ Exposure).
    pub exposure: f64,
    /// Print exposure, EV −2..2 (more is a darker print).
    pub print_exposure: f64,
    /// Enlarger filtration on top of the automatic neutral balance, CC −100..100.
    pub print_magenta: f64,
    pub print_yellow: f64,
}

impl Default for Film {
    fn default() -> Self {
        Self {
            enabled: false,
            stock: "kodak_portra_400".into(),
            paper: String::new(),
            exposure: 0.0,
            print_exposure: 0.0,
            print_magenta: 0.0,
            print_yellow: 0.0,
        }
    }
}

impl Film {
    pub fn is_default(&self) -> bool {
        *self == Film::default()
    }
}
```

In `DevelopSettings`, after `pub grain: Grain,`:

```rust
    /// Film simulation; omitted while default (see [`Film`]).
    #[serde(skip_serializing_if = "Film::is_default")]
    pub film: Film,
```

and in `impl Default for DevelopSettings`, after `grain: Grain::default(),`, add `film: Film::default(),`.

Build: `CARGO_TARGET_DIR=target/agent-988 cargo build -p lightcraft-develop`. Fix every `DevelopSettings { … }` struct
literal the compiler reports by adding `film: Film::default()` (or `..Default::default()` where the literal already
uses it).

- [ ] **Step 3: Controls and section**

In `crates/develop/src/controls.rs`:
- Add `Film,` to `enum Section` after `Grain,`.
- Add `Section::Film => "Film",` to `label()`.
- Add, after the three `grain.*` lines in `controls! { … }`:

```rust
    "film.exposure" => film.exposure, "Film Exposure", Film, -3, 3, 0, 0.05, 2, Centered;
    "film.printExposure" => film.print_exposure, "Print Exposure", Film, -2, 2, 0, 0.05, 2, Centered;
    "film.printMagenta" => film.print_magenta, "Magenta Filter", Film, -100, 100, 0, 1, 0, Centered;
    "film.printYellow" => film.print_yellow, "Yellow Filter", Film, -100, 100, 0, 1, 0, Centered;
```

Then build and fix every non-exhaustive `match` on `Section` the compiler reports:
`grep -rn "Section::Grain" crates apps --include='*.rs'` lists the sites. Map `Section::Film` next to `Section::Grain`
in each, using its section id `"film"` where a site maps sections to ids.

- [ ] **Step 4: Preset group**

In `crates/develop/src/presets.rs`:
- Add `Film` to `enum SettingsGroup` after `Grain`.
- Add `SettingsGroup::Film => "Film Simulation",` to its label match.
- Add `SettingsGroup::Film => &["film"],` to `keys()`.
- Add `SettingsGroup::Film` to any "all groups" list. Find them with
  `grep -rn "SettingsGroup::Grain" crates apps --include='*.rs'` (5 sites today); add `Film` after `Grain` in each.

- [ ] **Step 5: Run the tests**

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-develop`. Expected: all pass, including the existing
settings-hash and process-version tests (`tests_process.rs`). If any test pins the full JSON of default settings and
now fails, the `skip_serializing_if` is missing; fix that rather than the test.

- [ ] **Step 6: Gate and commit**

Run `cargo xtask ci`. `cargo xtask parity` may report the new `ctl:` ids as untracked; that's fine until Task 10, but
`ci` must still be green. Then:

```bash
git add crates/develop crates/engine crates/ui-egui apps
git commit -m "film (#988): film simulation settings, controls and preset group"
```

---

### Task 7: Commands `develop.film` and `develop.filmStocks`

**Files:**
- Modify: `crates/engine/Cargo.toml` (`lightcraft-film = { workspace = true }`)
- Modify: `crates/engine/src/cmd/develop.rs` (two commands, next to `develop.hdr`)
- Create: `crates/engine/src/tests_film.rs`
- Modify: `crates/engine/src/lib.rs` (`#[cfg(test)] mod tests_film;` beside the other `mod tests_*;`)

**Interfaces:**
- Consumes: `lightcraft_film::{films, papers}`, `Film` settings.
- Produces:
  - `develop.film {enabled?: bool, stock?: string, paper?: string}` → `{enabled, stock, paper}`.
  - `develop.filmStocks {}` → `{films: [{id, name, kind}], papers: [...]}`.

- [ ] **Step 1: Write the failing tests** (`crates/engine/src/tests_film.rs`)

```rust
//! Film simulation commands (issue #988).

use serde_json::json;

use crate::Session;

fn active_demo() -> (Session, lightcraft_catalog::PhotoId) {
    let mut s = Session::with_demo();
    let id = s.catalog.photos().find(|p| p.meta.title == "Golden horizon").map(|p| p.id).unwrap();
    s.execute("library.select", &json!({"ids": [id.0]})).unwrap();
    (s, id)
}

#[test]
fn film_command_turns_film_on_with_a_stock_and_undoes() {
    let (mut s, id) = active_demo();
    let r = s.execute("develop.film", &json!({"enabled": true, "stock": "kodak_ektar_100"})).unwrap();
    assert_eq!(r["enabled"], true);
    assert_eq!(r["stock"], "kodak_ektar_100");
    let f = s.develop_of(id).unwrap().film.clone();
    assert!(f.enabled && f.stock == "kodak_ektar_100" && f.paper.is_empty());
    // no `enabled`: toggles
    assert_eq!(s.execute("develop.film", &json!({})).unwrap()["enabled"], false);
    s.execute("edit.undo", &json!({})).unwrap();
    assert!(s.develop_of(id).unwrap().film.enabled);
}

#[test]
fn film_command_refuses_unknown_and_wrong_kind_ids() {
    let (mut s, id) = active_demo();
    for p in [
        json!({"stock": "kodak_tri_x"}),
        json!({"stock": "kodak_portra_endura"}),
        json!({"paper": "kodak_portra_400"}),
        json!({"stock": 7}),
        json!({"enabled": true, "paper": "nope"}),
    ] {
        let e = s.execute("develop.film", &p).unwrap_err().to_string();
        assert!(e.contains("develop.film"), "{p}: {e}");
    }
    assert!(s.develop_of(id).unwrap().film.is_default(), "a refused command changes nothing");
    // "" is the stock's own paper
    s.execute("develop.film", &json!({"enabled": true, "paper": ""})).unwrap();
}

#[test]
fn film_stocks_lists_films_and_papers() {
    let mut s = Session::with_demo();
    let r = s.execute("develop.filmStocks", &json!({})).unwrap();
    assert_eq!(r["films"][0]["id"], "kodak_portra_400");
    assert_eq!(r["films"][3]["kind"], "positive");
    assert_eq!(r["papers"][0]["id"], "kodak_portra_endura");
}
```

Check `grep -n '"edit.undo"\|"undo"' crates/engine/src/cmd/edit.rs` for the undo command id and use it.

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-engine tests_film`. Expected: FAIL (unknown command).

- [ ] **Step 2: Implement the commands** in `crates/engine/src/cmd/develop.rs`, right after the `develop.hdr` `cmd!`

```rust
        // ---- film simulation (issue #988)
        cmd!(
            "develop.film",
            "Film Simulation",
            [],
            None,
            "{enabled?: bool (default: toggle), stock?: film id (develop.filmStocks), paper?: paper id, \"\" = the stock's own} — film simulation: the photo exposed on a film stock, printed and scanned, in place of the tone map (Contrast, Whites and Blacks don't apply while it is on)",
            has_active,
            |s, p| {
                let id = active(s, "develop.film")?;
                let cur = s.develop_of(id).map(|d| d.film.clone()).unwrap_or_default();
                let enabled = bool_or(p, "enabled", !cur.enabled);
                let pick = |key: &str, list: Vec<lightcraft_film::StockInfo>, empty_ok: bool| -> Result<Option<String>> {
                    let Some(v) = p.get(key) else { return Ok(None) };
                    let v = v.as_str().ok_or_else(|| bad("develop.film", format!("`{key}` must be a string")))?;
                    if (empty_ok && v.is_empty()) || list.iter().any(|f| f.id == v) {
                        Ok(Some(v.to_string()))
                    } else {
                        let ids: Vec<_> = list.iter().map(|f| f.id).collect();
                        Err(bad("develop.film", format!("unknown {key} `{v}` (one of: {})", ids.join(", "))))
                    }
                };
                let stock = pick("stock", lightcraft_film::films(), false)?;
                let paper = pick("paper", lightcraft_film::papers(), true)?;
                let label = if enabled { "Film Simulation On" } else { "Film Simulation Off" };
                edit(s, "develop.film", label, |d| {
                    d.film.enabled = enabled;
                    if let Some(st) = stock {
                        d.film.stock = st;
                    }
                    if let Some(pp) = paper {
                        d.film.paper = pp;
                    }
                    Ok(())
                })?;
                let f = s.develop_of(id).map(|d| d.film.clone()).unwrap_or_default();
                Ok(json!({"enabled": f.enabled, "stock": f.stock, "paper": f.paper}))
            }
        ),
        cmd!(
            "develop.filmStocks",
            "Film Stocks",
            [],
            None,
            "{} — the film stocks and print papers develop.film accepts: {films: [{id, name, kind: negative|positive}], papers: [{id, name, kind}]}",
            always,
            |_, _| Ok(json!({"films": lightcraft_film::films(), "papers": lightcraft_film::papers()}))
        ),
```

`Result` here is `crate::Result`, already imported at the top of `develop.rs`. If the closure-returning-`Result`
helper fights the borrow checker on `p`, inline the two lookups instead.

Add `lightcraft-film = { workspace = true }` to `crates/engine/Cargo.toml` `[dependencies]`, and
`#[cfg(test)] mod tests_film;` next to the other test modules in `crates/engine/src/lib.rs`.

- [ ] **Step 3: Run the tests**

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-engine tests_film`. Expected: 3 passed.

- [ ] **Step 4: Gate and commit**

Run `cargo xtask ci`, then:

```bash
git add crates/engine Cargo.lock
git commit -m "film (#988): develop.film and develop.filmStocks commands"
```

---

### Task 8: Pipeline: the film stage in place of the tone map; GPU declines

**Files:**
- Modify: `crates/pipeline/Cargo.toml` (`lightcraft-film = { workspace = true }`)
- Modify: `crates/pipeline/src/finish.rs:152-261` (`FinishParams.film` and its construction), `crates/pipeline/src/finish.rs:533-545` (the stage)
- Modify: `crates/pipeline/src/lib.rs` (`#[cfg(test)] mod tests_film;` beside `mod tests_hdr;`)
- Create: `crates/pipeline/src/tests_film.rs`
- Modify: `crates/engine/src/media.rs:931-934` (GPU declines film renders)

**Interfaces:**
- Consumes: `lightcraft_film::{look, LookParams, Look}`, `DevelopSettings.film`, `s.section_enabled("film")`.
- Produces:
  - `FinishParams.film: Option<lightcraft_film::Look>`.
  - `pub fn lightcraft_pipeline::finish::film_look(&DevelopSettings) -> Option<lightcraft_film::Look>`, for the GPU
    check and the UI.

- [ ] **Step 1: Write the failing tests** (`crates/pipeline/src/tests_film.rs`)

```rust
//! Film simulation in the per-pixel stage (issue #988).

use lightcraft_develop::{DevelopSettings, Film, Treatment};
use lightcraft_raster::Rgb32f;

use crate::{OutputDepth, RenderRequest, SourceInfo, render};

fn sunset() -> Rgb32f {
    let lib = lightcraft_scenes::demo_library();
    let s = lib.iter().find(|s| s.kind == lightcraft_scenes::Kind::OceanSunset).map(|s| s.render(240, 160));
    s.unwrap_or_else(|| lib[0].render(240, 160))
}

fn raw() -> SourceInfo {
    SourceInfo { raw: true, ..Default::default() }
}

fn film(stock: &str) -> DevelopSettings {
    let mut s = DevelopSettings::default();
    s.film = Film { enabled: true, stock: stock.into(), ..Film::default() };
    s
}

#[test]
fn film_changes_the_render_and_off_is_untouched() {
    let src = sunset();
    let req = RenderRequest::fit(240, 240);
    let plain = render(&src, &raw(), &DevelopSettings::default(), &req);
    // settings that name a stock but are off: pixel-identical to no film at all
    let mut off = film("kodak_portra_400");
    off.film.enabled = false;
    assert_eq!(render(&src, &raw(), &off, &req).image.data, plain.image.data);
    let on = render(&src, &raw(), &film("kodak_portra_400"), &req);
    assert_ne!(on.image.data, plain.image.data);
    let slide = render(&src, &raw(), &film("fujifilm_velvia_100"), &req);
    assert_ne!(slide.image.data, on.image.data);
}

#[test]
fn a_grey_card_prints_grey() {
    let src = Rgb32f::filled(64, 64, [0.184, 0.184, 0.184]);
    let r = render(&src, &raw(), &film("kodak_portra_400"), &RenderRequest::fit(64, 64));
    let px = r.image.data[32 * 64 + 32];
    let spread = px[0].max(px[1]).max(px[2]) - px[0].min(px[1]).min(px[2]);
    assert!(spread <= 3, "{px:?}");
}

#[test]
fn the_film_eye_switches_it_off() {
    let src = sunset();
    let req = RenderRequest::fit(240, 240);
    let mut s = film("kodak_gold_200");
    s.disabled_sections.push("film".into());
    assert_eq!(render(&src, &raw(), &s, &req).image.data, render(&src, &raw(), &DevelopSettings::default(), &req).image.data);
}

#[test]
fn an_unknown_stock_renders_without_film() {
    let src = sunset();
    let req = RenderRequest::fit(240, 240);
    let plain = render(&src, &raw(), &DevelopSettings::default(), &req);
    for stock in ["kodak_tri_x", "kodak_portra_endura", ""] {
        assert_eq!(render(&src, &raw(), &film(stock), &req).image.data, plain.image.data, "{stock}");
    }
}

#[test]
fn film_with_hdr_and_bw_stays_in_range() {
    let src = sunset();
    let mut s = film("kodak_portra_400");
    s.hdr.enabled = true;
    s.hdr.max_ev = 3.0;
    s.light.exposure = 5.0;
    let req = RenderRequest { depth: OutputDepth::F32Hdr, ..RenderRequest::fit(120, 120) };
    let r = render(&src, &raw(), &s, &req);
    if let Some(crate::DeepImage { samples: crate::DeepSamples::F32(v), .. }) = &r.deep {
        assert!(v.iter().all(|c| c.is_finite() && *c <= 1.0 + 1e-4), "film output stays at most SDR white");
    }
    let mut bw = film("kodak_portra_400");
    bw.treatment = Treatment::Bw;
    let px = render(&src, &raw(), &bw, &RenderRequest::fit(120, 120)).image.data[60 * 120 + 60];
    assert!(px[0] == px[1] && px[1] == px[2], "B&W still converts: {px:?}");
}
```

If `RenderRequest::fit(64, 64)` of a 64×64 source isn't exactly 64×64, index the centre with `r.image.width` and
`r.image.height` instead of the literals.

Add `#[cfg(test)] mod tests_film;` in `crates/pipeline/src/lib.rs` next to `mod tests_hdr;`. Add
`lightcraft-film = { workspace = true }` to `crates/pipeline/Cargo.toml`.

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-pipeline tests_film`
Expected: compiles (the settings exist) and `film_changes_the_render…`, `a_grey_card_prints_grey` FAIL. The others
pass trivially for now.

- [ ] **Step 2: Build the look in `FinishParams`**

In `crates/pipeline/src/finish.rs`, add to `struct FinishParams` after `pub lut`:

```rust
    /// Film simulation ([`film_look`]): replaces the tone map (CPU only; the GPU path declines film renders).
    pub film: Option<lightcraft_film::Look>,
```

In `FinishParams::new`, after `lut: …,` add:

```rust
            film: film_look(s),
```

Below `base_tone`, add:

```rust
/// The film look `s` asks for: `None` while film simulation is off, its section's eye is off, or its stock or paper
/// is not one that ships (a hand-edited file) — then the photo renders without film rather than failing.
pub fn film_look(s: &DevelopSettings) -> Option<lightcraft_film::Look> {
    let f = &s.film;
    if !f.enabled || !s.section_enabled("film") {
        return None;
    }
    lightcraft_film::look(&lightcraft_film::LookParams {
        stock: f.stock.clone(),
        paper: f.paper.clone(),
        exposure_ev: f.exposure,
        print_exposure_ev: f.print_exposure,
        print_magenta: f.print_magenta,
        print_yellow: f.print_yellow,
    })
    .ok()
}
```

- [ ] **Step 3: Replace the tone map when a film look is set**

Replace the block at `finish.rs:533-545` (from `// --- tone map on luminance, highlight desaturation` through the
closing `}` of the `if mx > peak` block) with:

```rust
            // --- film simulation (exposure → print → scan) or the tone map on luminance, highlight desaturation
            let mut d = if let Some(film) = &fp.film {
                film.apply(c)
            } else {
                let yl = luminance_2020(c);
                let o = tone.apply(yl);
                let mut d = if yl > 1e-9 { c.map(|v| v * o / yl) } else { [0.0; 3] };
                let k = tone.chroma_scale(o);
                if k != 1.0 {
                    d = d.map(|v| o + (v - o) * k);
                }
                let mx = d[0].max(d[1]).max(d[2]);
                if mx > peak {
                    let t = ((mx - peak) / (mx - o).max(1e-6)).clamp(0.0, 1.0);
                    d = d.map(|v| v + (o - v) * t);
                }
                d
            };
```

The next line (`// --- colour`, `d = ops.apply(d, l_sat, l_hue);`) stays as it is and now reads the outer `d`.

Update the crate doc comment's stage list in `crates/pipeline/src/lib.rs:12`, replacing it with:

```text
//! 3. tone map — contrast / whites / blacks filmic curve on luminance, highlight desaturation; or, with film
//!    simulation on, the film look (`lightcraft_film`: exposure → print → scan) in its place
```

- [ ] **Step 4: GPU declines film renders**

In `crates/engine/src/media.rs`, `develop_image`, add a condition beside the LUT one:

```rust
    // LUT profiles and film simulation have no GPU stage: they render on the CPU
    let fallback = if gpu
        && !lightcraft_pipeline::lut::is_lut_profile(&s.profile.id)
        && lightcraft_pipeline::finish::film_look(s).is_none()
```

(Keep the rest of the condition as it is. If `finish` isn't `pub mod` in pipeline's `lib.rs`, it is: `pub mod finish;`.)

Search for other GPU entry points that render without `develop_image`:
`grep -rn "lightcraft_gpu::try_render\|gpu::render" crates --include='*.rs'`. Each must decline the same way (export
and region renders). Add the same `film_look(s).is_none()` guard to each.

- [ ] **Step 5: Run the tests**

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-pipeline` and
`CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-engine`. Expected: all pass, including `tests_film` and the
GPU equivalence tests (`crates/gpu/tests/equivalence.rs`; film is off there).

- [ ] **Step 6: Look at it**

Render a real photo through the CLI and look at the output (CLAUDE.md: run the app after user-visible changes). Run
each command on its own:

```bash
CARGO_TARGET_DIR=target/agent-988 cargo build --release -p lightcraft-cli
mkdir -p "$CLAUDE_JOB_DIR/tmp/film"
target/agent-988/release/lightcraft-cli snapshot --demo --script /dev/null -o "$CLAUDE_JOB_DIR/tmp/film/plain.png" --size 1600x1000
```

Then a script `film.jsonl` (one control request per line) that selects the first demo photo, runs `develop.film`
`{"enabled": true, "stock": "kodak_portra_400"}`, and takes a screenshot (request shapes are in
`docs/control-protocol.md` → Headless rendering). Run `snapshot --demo --script film.jsonl`, and repeat for
`fujifilm_velvia_100`. Open the PNGs with the Read tool. Expected:
- Portra: softer contrast, warm, pleasant skin and sky.
- Velvia: punchy and saturated.
- Neither has hard clipping or colour casts on neutral surfaces.

Record what you saw in the commit body.

- [ ] **Step 7: Gate and commit**

```bash
git add crates/pipeline crates/engine Cargo.lock
git commit -m "film (#988): film stage in place of the tone map; CPU only for now"
```

---

### Task 9: Edit panel: the Film section

**Files:**
- Modify: `crates/ui-egui/src/panels/edit.rs` (new `section(…, "film", "Film", …)` after the Effects section, before Detail)
- Modify: `crates/ui-egui/Cargo.toml` (`lightcraft-film = { workspace = true }`)
- Create: `crates/ui-egui/src/tests_film.rs`
- Modify: `crates/ui-egui/src/lib.rs` (`#[cfg(test)] mod tests_film;` next to `mod tests_masking;`)

**Interfaces:**
- Consumes: `develop.film`, `develop.filmStocks` (via `app.run`), `control(app, ui, d, id, true)`,
  `crate::widgets` helpers, `crate::i18n::tr`.
- Produces: widget ids `combo:filmStock`, `combo:filmPaper`, plus the four `film.*` sliders.

- [ ] **Step 1: Write the failing headless test**

`crates/ui-egui/src/tests_film.rs`, with the same setup as `tests_masking.rs`:

```rust
//! Headless tests of the Edit panel's Film section (issue #988).

use std::time::Duration;

use serde_json::json;

use crate::headless::Headless;
use crate::{LightcraftApp, Services};

const T: Duration = Duration::from_secs(20);
const SETTLE: Duration = Duration::from_secs(120);

fn edit_panel() -> Headless {
    let services = Services { png: None, ..Default::default() };
    let app = LightcraftApp::new(lightcraft_engine::Session::with_demo(), services);
    let mut h = Headless::new(app, [1200.0, 1600.0], 1.0);
    let r = h.request("ui.set", json!({"view": "detail"}), T);
    assert_eq!(r["ok"], true, "{r}");
    let r = h.request("engine.execute", json!({"command": "panel.edit"}), T);
    assert_eq!(r["ok"], true, "{r}");
    h
}

fn on_screen(h: &Headless, id: &str) -> bool {
    h.app.widgets.iter().any(|(w, _)| w == id)
}

/// Edit ▸ Film: choosing a stock turns film simulation on and shows the print controls; a slide film has no
/// print controls.
#[test]
fn film_section_shows_the_controls_for_the_stock() {
    let mut h = edit_panel();
    h.snapshot(SETTLE);
    assert!(on_screen(&h, "combo:filmStock"));
    assert!(!on_screen(&h, "slider:film.exposure"), "film off: no sliders");
    let r = h.request("engine.execute", json!({"command": "develop.film", "params": {"enabled": true, "stock": "kodak_gold_200"}}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.snapshot(SETTLE);
    for id in ["combo:filmPaper", "slider:film.exposure", "slider:film.printExposure", "slider:film.printMagenta", "slider:film.printYellow"] {
        assert!(on_screen(&h, id), "{id} not on screen");
    }
    let r = h.request("engine.execute", json!({"command": "develop.film", "params": {"stock": "fujifilm_velvia_100"}}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.snapshot(SETTLE);
    assert!(on_screen(&h, "slider:film.exposure"));
    assert!(!on_screen(&h, "slider:film.printExposure") && !on_screen(&h, "combo:filmPaper"), "a slide is not printed");
}
```

Sliders register as `slider:<control id>` (`crates/ui-egui/src/widgets.rs:196`). If `h.app.widgets` keeps rects from
earlier frames, so that the "not on screen" asserts fail after a stock change, check how `widgets` is cleared per
frame (`grep -n "widgets.clear\|widgets =" crates/ui-egui/src/*.rs`) and take a second `h.snapshot` before asserting.
If the Edit panel's sections start collapsed, expand Film first by clicking its header (see how `section()` in
`edit.rs` registers the header id).

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-ui-egui film_section`. Expected: FAIL (no Film
section).

- [ ] **Step 2: Add the section**

In `crates/ui-egui/src/panels/edit.rs`, after the Effects `section(…)` call:

```rust
    // Film simulation (issue #988): beyond Lightroom. The stock list comes from lightcraft-film.
    section(app, ui, &d, "film", "Film", |app, ui, d| {
        let films = lightcraft_film::films();
        let shown = if d.film.enabled {
            films.iter().find(|f| f.id == d.film.stock).map_or(d.film.stock.as_str(), |f| f.name)
        } else {
            crate::i18n::tr("None")
        };
        egui::ComboBox::from_id_salt("combo:filmStock").width(190.0).selected_text(shown).show_ui(ui, |ui| {
            if ui.selectable_label(!d.film.enabled, crate::i18n::tr("None")).clicked() {
                let _ = app.run("develop.film", json!({"enabled": false}));
            }
            for f in &films {
                if ui.selectable_label(d.film.enabled && d.film.stock == f.id, f.name).clicked() {
                    let _ = app.run("develop.film", json!({"enabled": true, "stock": f.id}));
                }
            }
        });
        if d.film.enabled {
            let negative = films.iter().any(|f| f.id == d.film.stock && f.kind == lightcraft_film::Kind::Negative);
            if negative {
                let papers = lightcraft_film::papers();
                let paper = papers.iter().find(|p| p.id == d.film.paper).map_or(crate::i18n::tr("Stock's own paper"), |p| p.name);
                egui::ComboBox::from_id_salt("combo:filmPaper").width(190.0).selected_text(paper).show_ui(ui, |ui| {
                    if ui.selectable_label(d.film.paper.is_empty(), crate::i18n::tr("Stock's own paper")).clicked() {
                        let _ = app.run("develop.film", json!({"paper": ""}));
                    }
                    for p in &papers {
                        if ui.selectable_label(d.film.paper == p.id, p.name).clicked() {
                            let _ = app.run("develop.film", json!({"paper": p.id}));
                        }
                    }
                });
            }
            control(app, ui, d, "film.exposure", true);
            if negative {
                sub_title(ui, crate::i18n::tr("Print"));
                control(app, ui, d, "film.printExposure", true);
                control(app, ui, d, "film.printMagenta", true);
                control(app, ui, d, "film.printYellow", true);
            }
            ui.label(egui::RichText::new(crate::i18n::tr("Contrast, Whites and Blacks are set by the film and paper.")).small().weak());
        }
        ui.add_space(8.0);
    });
```

Register both combo boxes so `ui.clickWidget` and the test can find them. Keep each `show_ui` result and register its
rect, for example:

```rust
        let stock_combo = egui::ComboBox::from_id_salt("combo:filmStock").width(190.0).selected_text(shown).show_ui(ui, |ui| {
            // … as above …
        });
        crate::widgets::register(ui.ctx(), "combo:filmStock", stock_combo.response.rect);
```

Do the same for `combo:filmPaper`. If the Light section can grey out Contrast, Whites and Blacks, grey them out while
`d.film.enabled` instead of only showing the hint.

Add `#[cfg(test)] mod tests_film;` to `crates/ui-egui/src/lib.rs` next to `mod tests_masking;`.

Add every new string ("Film", "None", "Stock's own paper", "Print", the hint) to the i18n catalogs as the localisation
check requires. Run `cargo xtask ci` and add the English keys wherever it reports them missing; other languages may
fall back to English per `docs/localization.md`.

- [ ] **Step 3: Run the test and look at the panel**

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-ui-egui film_section`. Expected: PASS.

Then drive the app headless and look at it:

```bash
CARGO_TARGET_DIR=target/agent-988 cargo build --release -p lightcraft-cli
```

Write a JSON-lines script that:
1. opens the develop view on the first demo photo;
2. runs `develop.film {"enabled": true, "stock": "kodak_portra_400"}`;
3. scrolls the Edit panel to the Film section (or collapses the sections above it);
4. takes `ui.screenshot`.

Run `lightcraft-cli snapshot --demo --script <file> -o "$CLAUDE_JOB_DIR/tmp/film/panel.png" --size 1600x1000` and open
the PNG. Expected:
- the Film section, with the stock combo, the paper combo, four sliders and the hint;
- the photo shows the film look.

- [ ] **Step 4: Gate and commit**

```bash
git add crates/ui-egui Cargo.lock
git commit -m "film (#988): Film section in the Edit panel"
```

---

### Task 10: Credits, documentation, progress docs

**Files:**
- Modify: `crates/film/src/lib.rs` (`CREDITS`)
- Modify: `crates/ui-egui/src/panels/dialogs.rs:169` (`ABOUT_TABS` gains `("data", "Data")`, and its tab body)
- Create: `docs/film-simulation.md`
- Modify: `docs/parity-checklist.md` (new row), `docs/gaps.md`, `ROADMAP.md`, `docs/target-app-parity.md` (only if a
  listed number changes), `docs/whats-new.md`

**Interfaces:**
- Produces: `pub const lightcraft_film::CREDITS: &str`.

- [ ] **Step 1: Write the failing test for the credits text** (in `crates/film/src/lib.rs`)

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn credits_carry_the_cc_by_sa_attribution() {
        for must in ["spektrafilm", "Andrea Volpato", "https://github.com/andreavolpato/spektrafilm", "CC BY-SA 4.0", "unmodified"] {
            assert!(super::CREDITS.contains(must), "{must}");
        }
    }
}
```

Run: `CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-film credits`. Expected: FAIL (no `CREDITS`).

- [ ] **Step 2: Add `CREDITS`**

```rust
/// Credits for the data compiled into this crate (Help ▸ About ▸ Data). Required by the stock data's licence.
pub const CREDITS: &str = "Film simulation stock data: spektrafilm by Andrea Volpato \
(https://github.com/andreavolpato/spektrafilm), licensed CC BY-SA 4.0 \
(https://creativecommons.org/licenses/by-sa/4.0/), unmodified. Kodak Portra 400, Ektar 100, Gold 200, \
Fujifilm Velvia 100 and Kodak Portra Endura, processed by the author from Kodak and Fujifilm data sheets. \
LightCraft's film model is its own; it uses no spektrafilm code.\n\n\
CIE 1931 colour-matching functions (CIE, CC-BY-4.0, via rawtoaces-data) and CIE daylight basis functions \
(CIE 15:2004, via colour-science, BSD-3-Clause).";
```

Run the test again. Expected: PASS.

- [ ] **Step 3: About ▸ Data tab**

In `crates/ui-egui/src/panels/dialogs.rs`:
- Change `ABOUT_TABS` to
  `&[("about", "About"), ("contributors", "Contributors"), ("models", "Models"), ("data", "Data")]`.
- In the `match tab { … }` at about line 1136, add an arm before `_ =>`:

  ```rust
                          3 => {
                              ui.label(egui::RichText::new(crate::i18n::tr("Third-party data")).font(t.semibold(16.0)).color(t.text));
                              ui.add_space(6.0);
                              egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                                  ui.label(lightcraft_film::CREDITS);
                              });
                          }
  ```

  The credits text stays in English: it is a licence notice.

The existing test `about_dialog_tabs_show_the_credits` (`crates/ui-egui/src/headless.rs:3428`) iterates `ABOUT_TABS`,
so it covers the new tab. Run it:
`CARGO_TARGET_DIR=target/agent-988 cargo test -p lightcraft-ui-egui about_dialog_tabs_show_the_credits`.
Expected: PASS.

- [ ] **Step 4: `docs/film-simulation.md`**

Write the page with these sections, each filled in from this plan:
1. **What it does.** The Film section; stock, paper and the sliders; Contrast, Whites and Blacks don't apply.
2. **The model.** The bullets from **Background → The model**, with the formulas.
3. **Stocks.** A table of the five ids, names, kind and the Kodak/Fuji datasheet they derive from (from each file's
   metadata).
4. **Credits and licence.** The CC BY-SA terms, a link to `assets/film/spektrafilm/README.md`, and the statement that
   no spektrafilm code is used.
5. **Limits (phase 1).** CPU only; no halation, film grain or interlayer couplers yet; an ideal enlarger, not dichroic
   filters; generic spectral upsampling, not per camera.
6. **Developer notes.** Where each stage lives; how to add a stock (vendor the file unmodified, add it to `BUILTIN`,
   add a README row, and pass the three stock tests).

- [ ] **Step 5: Progress docs**

- `docs/parity-checklist.md`: add a row in the Effects group after `LR-EDIT-EFFECTS-GRAIN`:

  ```markdown
  | LR-EDIT-EFFECTS-FILMSIM | Film simulation (beyond Lightroom) | P2 | 🟡 | `cmd:develop.film`, `cmd:develop.filmStocks`, `ctl:film.exposure`, `ctl:film.printExposure`, `ctl:film.printMagenta`, `ctl:film.printYellow`, `crates/film/src/look.rs`, `docs/film-simulation.md` | Issue #988. Spectral film → print → scan model of our own fed by spektrafilm stock data (CC BY-SA 4.0): Portra 400, Ektar 100, Gold 200 on Portra Endura; Velvia 100 viewed directly. Neutral print balance, film and print exposure, magenta/yellow filtration. 🟡: CPU only (GPU declines), no halation, film grain or interlayer couplers, ideal enlarger, five stocks |
  ```

  Run `CARGO_TARGET_DIR=target/agent-988 cargo xtask parity --write` to refresh the summary.
- `docs/gaps.md`: add the film-simulation follow-ups (GPU stage, halation, film grain, couplers, more stocks) as one
  ranked entry below the beta blockers. Bump the *Last updated* line and the revision history.
- `ROADMAP.md`: mention film simulation in the dimension that covers creative looks and profiles. Bump *Last updated*
  and the revision history per craftrules `standards/progress-docs.md`.
- `docs/whats-new.md`: one entry, "Film simulation: Kodak Portra 400, Ektar 100, Gold 200 and Fujifilm Velvia 100,
  printed on Portra Endura (Edit ▸ Film)", with credit to spektrafilm for the stock data.

- [ ] **Step 6: Full gate and commit**

Run: `CARGO_TARGET_DIR=target/agent-988 cargo xtask ci`. Expected: green. Known local environment failures are listed in
`.claude/skills/fixing-github-issues/SKILL.md` §5; confirm each also fails on main before dismissing it.

```bash
git add crates/film crates/ui-egui docs ROADMAP.md
git commit -m "film (#988): credits in About ▸ Data, film simulation docs and parity row"
```

- [ ] **Step 7: Clean up**

`rm -rf target/agent-988` (the disk is shared).

---

## Follow-up plans (out of scope here, one plan each)

1. **GPU film stage.** Upload the two grids as 3-D textures and the six tables as 1-D textures, mirror `Look::apply`
   in `finish.wgsl`, drop the decline, and add an equivalence test.
2. **Halation.** A blurred red-layer exposure added back before the film curves; needs a spatial pre-pass like the
   clarity cache.
3. **Film grain.** Density-domain grain per layer with a size that follows the stock, in place of the Effects ▸ Grain
   value noise while film is on.
4. **Interlayer (DIR) couplers.** Saturation and local contrast from inhibitor couplers (Hunt ch. 15), with our own
   parameters.
5. **More stocks.** Portra 160/800, Ektachrome, Kodachrome, Vision3, cine print 2383; push variants.
6. **Camera-specific upsampling.** Use the measured camera sensitivities in `crates/raw/src/spectral.rs` instead of the
   generic basis.
7. **Dichroic enlarger model.** Our own smooth filter curves in place of the ideal per-layer balance.
