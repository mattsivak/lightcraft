# Film simulation

> **Last updated:** 2026-10-11 · Issue [#988](https://github.com/storytold/lightcraft/issues/988) · Beyond Lightroom

LightCraft can render a photo as if it had been shot on a colour film, printed on photographic paper and scanned
(or, for slide film, viewed on a light box). The film model is LightCraft's own, written from published colour
science. The stock data (spectral sensitivities, dye densities and characteristic curves) are
[spektrafilm](https://github.com/andreavolpato/spektrafilm) profiles by Andrea Volpato, shipped unmodified under
CC BY-SA 4.0. No spektrafilm code is used.

## What it does

**Edit ▸ Film** (the section below Effects):

- **Stock**: None, or one of the shipped films. Choosing a film turns film simulation on; None turns it off.
- **Paper** (negatives only): the print paper. "The stock's own" uses the paper named in the film's data, or Kodak
  Portra Endura when that paper doesn't ship. Slide film has no paper: it is viewed directly.
- **Film Exposure** (−3…+3 EV): exposes the film more or less, on top of Light ▸ Exposure.
- **Print Exposure** (−2…+2 EV, negatives only): more print exposure gives a darker print.
- **Magenta Filter** / **Yellow Filter** (−100…+100 CC, negatives only): enlarger filtration on top of the automatic
  neutral balance. More magenta makes the print greener, more yellow makes it bluer, as in a darkroom.

The film stage takes the place of the tone map. Its curves set the contrast and the end points, so **Contrast,
Whites and Blacks don't apply** while film is on (their sliders are disabled). Everything before the tone map
(white balance, exposure, highlights / shadows, texture / clarity, calibration) still feeds the film, and
everything after it (vibrance, saturation, the colour mixer, colour grading, B&W conversion, tone curves, vignette,
grain) still applies to the print. The section's eye switch turns the film
off for a before/after view; the photo then renders exactly as without film. With HDR editing on, the film look is
display-referred: it peaks at SDR white.

Agents and the CLI use the same commands:

- `develop.film {enabled?, stock?, paper?}` turns film on or off and picks the stock and paper. An id that doesn't
  ship is refused with the list of valid ids.
- `develop.filmStocks {}` lists the films and papers (`{films: [{id, name, kind}], papers: [...]}`).
- The sliders are develop controls: `film.exposure`, `film.printExposure`, `film.printMagenta`, `film.printYellow`.
- The settings live in `DevelopSettings.film` (preset group *Film*). An edit without film doesn't write the key, so
  edits saved before this feature keep their settings hash and render pixel-identically.

A hand-edited sidecar that names an unknown stock, or a paper as the film, renders without film instead of failing.

## The model

Everything below runs in linear Rec.2020 on a spectral grid of 81 samples, 380–780 nm every 5 nm.

- **Spectral upsampling** (`crates/film/src/basis.rs`).
  - Three smooth reflectance spectra `b_r, b_g, b_b` that sum to 1. Under D65 their colours are the Rec.2020
    primaries.
  - They are the smoothest such set: least squared first differences, solved once in closed form (KKT system).
  - A linear Rec.2020 colour `(r, g, b)` becomes the reflectance `r·b_r + g·b_g + b·b_b`.
- **Exposure** (`look.rs`, `exposure_matrix`).
  - Layer exposure is `E_l = Σ_λ illum_ref(λ) · reflectance(λ) · sens_l(λ)`, with the stock's reference illuminant
    (D55 for the shipped films). It is linear in RGB, so it becomes a 3×3 matrix per stock.
  - Rows are scaled so that an 18.4 % grey gives `E_l = 1` (log10 0). Measured: log exposure 0 sits mid-scale on
    the straight line of every shipped curve (Portra 400 densities 0.68 / 0.70 / 0.86 there, Velvia 0.51 / 0.54 /
    0.55).
- **Film density.** `D_l = curve_l(log10(E_l · gain))`, read from the profile's `density_curves` (density above
  base); `gain = 2^(film exposure)`.
- **Negative → paper.**
  - Negative transmittance: `T(λ) = 10^-(base(λ) + Σ_i D_i · dye_i(λ))`. Measured: `base + Σ D(0) · dye`
    reproduces each profile's `midscale_neutral_density` with RMS 0.085 (Portra 400), 0.02 (Portra Endura) and
    0.012 (Velvia 100), so the dye spectra are per unit density.
  - Paper log exposure: `H_j = log10 Σ_λ lamp(λ) · T(λ) · sens_paper_j(λ)`.
  - The enlarger lamp is our own approximation: a 3200 K blackbody with a logistic roll-off above 700 nm. It is not
    measured filter data.
- **Neutral print balance.** Per-layer log offsets are chosen so the scanned print of an 18.4 % grey is neutral and
  keeps the brightness the plain offsets give. Newton's method over three unknowns with a numeric Jacobian. It is an
  ideal enlarger, not a dichroic filter model.
- **Print exposure and filtration.** On top of the balance: print exposure adds `EV · log10 2` to every paper layer;
  magenta CC subtracts `CC / 100` density from the green-sensitive layer's exposure, yellow CC from the
  blue-sensitive layer's (1 CC = 0.01 density).
- **Paper density.** `D'_j = paper_curve_j(H_j + balance_j + offset_j)`.
- **Scan / view.**
  - Reflectance (paper) or transmittance (slide) is `10^-(base + Σ D · dye)`, seen under D50, giving XYZ.
  - Bradford-adapted from the unexposed base's white to D65, then converted to linear Rec.2020 with the base mapped
    to 1.0: the paper base is display white.
  - The output is clamped to 0…1; NaN, infinities and negative inputs give finite output.
- **Baking and caching.**
  - The film densities → paper log exposure grid depends only on (film, paper); the paper densities → Rec.2020
    grid only on the paper; a slide's densities → Rec.2020 grid only on the slide.
  - They are baked once into 33³ grids (trilinear lookup) plus 1-D tables of the curves, and cached per
    (film, paper) pair. Per pixel a look costs a 3×3 matrix, six 1-D lookups and two trilinear lookups; dragging a
    film slider never rebakes.
- **Channel order** (checked against the data): `log_sensitivity` columns are R, G, B (peaks at 645 / 545 / 400 nm
  on Portra 400); `channel_density` columns are C, M, Y (peaks at 695 / 540 / 450 nm). Slide curves fall with
  exposure; negative and paper curves rise.

## Stocks

| Id | Name | Kind | Prints on | Data source (from the file's metadata) |
|---|---|---|---|---|
| `kodak_portra_400` | Kodak Portra 400 | colour negative | `kodak_portra_endura` | Kodak data sheets, processed by A. Volpato |
| `kodak_ektar_100` | Kodak Ektar 100 | colour negative | `kodak_portra_endura` | Kodak data sheets, processed by A. Volpato |
| `kodak_gold_200` | Kodak Gold 200 | colour negative | `kodak_portra_endura` | Kodak data sheets, processed by A. Volpato |
| `fujifilm_velvia_100` | Fujifilm Velvia 100 | slide (positive) | viewed directly | Fujifilm data sheets, processed by A. Volpato |
| `kodak_portra_endura` | Kodak Professional Portra Endura | print paper | — | Kodak data sheets, processed by A. Volpato |

Each file's metadata says it was "created by processing raw measurement data from data-sheets and/or scientific
papers" (Kodak and Fujifilm data sheets, scientific publications and technical material), and that the original
data are the property of their holders. The films' reference illuminant is D55; the paper's is a tungsten enlarger
lamp (`TH-KG3`), for which we use our own lamp above.

## Credits and licence

- Stock data: **spektrafilm by Andrea Volpato**, https://github.com/andreavolpato/spektrafilm, licensed
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The five JSON files are copied unmodified into
  [`assets/film/spektrafilm/`](../assets/film/spektrafilm/README.md) together with `SPEKTRAFILM_LICENSE.txt`; the
  README there pins the source commit and each file's SHA-256. They are credited in `NOTICE`,
  `assets/ATTRIBUTION.md` and Help ▸ About ▸ Data (`lightcraft_film::CREDITS`).
- Share-alike covers the data only. Any change to those files must be marked (a `CHANGELOG.txt` next to them) and
  stay CC BY-SA 4.0. LightCraft's code, including the film model, is MIT OR Apache-2.0.
- **No spektrafilm code is used**, nor any other film-simulation implementation: the model above is written from
  published colour science (Hunt, *The Reproduction of Colour*; Giorgianni & Madden, *Digital Color Management*)
  and the data sheets.
- The author asks, without binding, that the name "spektrafilm" not be used in product branding (the UI says
  "Film Simulation"), that the data not be resold, and that they not be used to train commercial AI. We follow these.
- CIE 1931 colour-matching functions (CIE, CC-BY-4.0, via rawtoaces-data) and the CIE daylight basis functions
  (CIE 15:2004, via colour-science, BSD-3-Clause) are in `crates/film/src/cie_table.rs`, generated by
  `tools/film_cie_table.py`.

## Limits (phase 1)

- **CPU only.** The GPU path declines film renders, so a photo with film on renders on the CPU.
- No halation, film grain or interlayer (DIR) couplers yet. Effects ▸ Grain still applies its own grain.
- An ideal enlarger with a per-layer neutral balance, not dichroic filters.
- Generic spectral upsampling from Rec.2020, not the camera's measured sensitivities.
- Five stocks.

The follow-ups are tracked in [gaps.md](gaps.md) (film simulation entry).

## Developer notes

| Stage | Where |
|---|---|
| Spectral grid, CIE tables, illuminants, enlarger lamp | `crates/film/src/spectrum.rs`, `cie_table.rs` |
| Profile parsing and validation (`Stock`, `Kind`, `Curve`) | `crates/film/src/profile.rs` |
| Built-in stocks, lookup and lists | `crates/film/src/stocks.rs` |
| Upsampling basis | `crates/film/src/basis.rs` |
| 1-D tables and 3-D grids | `crates/film/src/grid.rs` |
| Exposure, print, scan, neutral balance, cache, `Look::apply` | `crates/film/src/look.rs` |
| Settings, controls, preset group | `crates/develop/src/settings.rs` (`Film`), `controls.rs` (`film.*`), `presets.rs` |
| Commands | `crates/engine/src/cmd/develop.rs` (`develop.film`, `develop.filmStocks`) |
| Render: film stage in place of the tone map | `crates/pipeline/src/finish.rs` (`film_look`, `FinishParams.film`) |
| GPU decline | `crates/engine/src/media.rs` |
| Edit ▸ Film section | `crates/ui-egui/src/panels/edit.rs` (`film_section`) |
| Credits | `crates/film/src/lib.rs` (`CREDITS`), Help ▸ About ▸ Data |

**Adding a stock:**

1. Copy the profile JSON **unmodified** from spektrafilm into `assets/film/spektrafilm/` (a modified file needs a
   `CHANGELOG.txt` entry and stays CC BY-SA).
2. Add its `include_str!` to `BUILTIN` in `crates/film/src/stocks.rs`.
3. Add a row with its SHA-256 to `assets/film/spektrafilm/README.md`, and name it in `CREDITS`, `NOTICE` and the
   `assets/ATTRIBUTION.md` row if they list the stocks.
4. Pass the three stock tests (`cargo test -p lightcraft-film stocks`): `every_builtin_stock_parses`,
   `every_stock_has_rgb_layers_and_cmy_dyes` and `dyes_reproduce_the_midscale_neutral`. Then check the look tests
   (`look::tests`) for the new stock, e.g. that a negative prints mid-grey neutral.
