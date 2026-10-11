//! Film simulation (issue #988): a scene-linear Rec.2020 colour is exposed onto a virtual film stock (measured
//! spectral sensitivities and characteristic curves), printed onto a virtual paper and scanned under D50, or — for
//! slide film — viewed directly on a D50 light box. The model is our own (`docs/film-simulation.md`); the stock data
//! are spektrafilm profiles by Andrea Volpato, CC BY-SA 4.0, vendored unmodified in `assets/film/spektrafilm/`.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod basis;
mod cie_table;
mod grid;
mod look;
mod profile;
pub mod spectrum;
mod stocks;

pub use look::{Look, LookParams, look};
pub use profile::{Curve, Kind, Stock};
pub use stocks::{DEFAULT_PAPER, StockInfo, films, papers, stock};

/// Credits for the data compiled into this crate (Help ▸ About ▸ Data). Required by the stock data's licence.
pub const CREDITS: &str = "Film simulation stock data: spektrafilm by Andrea Volpato \
(https://github.com/andreavolpato/spektrafilm), licensed CC BY-SA 4.0 \
(https://creativecommons.org/licenses/by-sa/4.0/), unmodified. Kodak Portra 400, Ektar 100, Gold 200, \
Fujifilm Velvia 100 and Kodak Portra Endura, processed by the author from Kodak and Fujifilm data sheets. \
LightCraft's film model is its own; it uses no spektrafilm code.\n\n\
CIE 1931 colour-matching functions (CIE, CC-BY-4.0, via rawtoaces-data) and CIE daylight basis functions \
(CIE 15:2004, via colour-science, BSD-3-Clause).";

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

#[cfg(test)]
mod tests {
    #[test]
    fn credits_carry_the_cc_by_sa_attribution() {
        for must in ["spektrafilm", "Andrea Volpato", "https://github.com/andreavolpato/spektrafilm", "CC BY-SA 4.0", "unmodified"] {
            assert!(super::CREDITS.contains(must), "{must}");
        }
    }
}
