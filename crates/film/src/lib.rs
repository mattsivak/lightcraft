//! Film simulation (issue #988): a scene-linear Rec.2020 colour is exposed onto a virtual film stock (measured
//! spectral sensitivities and characteristic curves), printed onto a virtual paper and scanned under D50, or — for
//! slide film — viewed directly on a D50 light box. The model is our own (`docs/film-simulation.md`); the stock data
//! are spektrafilm profiles by Andrea Volpato, CC BY-SA 4.0, vendored unmodified in `assets/film/spektrafilm/`.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

#[allow(dead_code)] // used by `look` from Task 5 on
mod basis;
mod cie_table;
mod profile;
pub mod spectrum;
mod stocks;

pub use profile::{Curve, Kind, Stock};
pub use stocks::{DEFAULT_PAPER, StockInfo, films, papers, stock};

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
