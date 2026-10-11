//! Film simulation in the per-pixel stage (issue #988).

use lightcraft_develop::{DevelopSettings, Film, Treatment};
use lightcraft_raster::Rgb32f;

use crate::{DeepSamples, OutputDepth, RenderRequest, SourceInfo, render};

fn sunset() -> Rgb32f {
    let lib = lightcraft_scenes::demo_library();
    let s = lib.iter().find(|s| s.kind == lightcraft_scenes::Kind::OceanSunset).map(|s| s.render(240, 160));
    s.unwrap_or_else(|| lib[0].render(240, 160))
}

fn raw() -> SourceInfo {
    SourceInfo { raw: true, ..Default::default() }
}

fn film(stock: &str) -> DevelopSettings {
    DevelopSettings { film: Film { enabled: true, stock: stock.into(), ..Film::default() }, ..Default::default() }
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
    let (w, h) = (r.image.width, r.image.height);
    let px = r.image.data[(h / 2) * w + w / 2];
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
    match r.deep.map(|d| d.samples) {
        Some(DeepSamples::F32(v)) => {
            let mx = v.iter().copied().fold(0.0f32, f32::max);
            let n = v.iter().filter(|c| **c > 1.0 + 1e-4).count();
            assert!(v.iter().all(|c| c.is_finite() && *c <= 1.0 + 1e-4), "film output stays at most SDR white: max {mx}, {n} of {}", v.len());
        }
        other => panic!("expected float samples, got {other:?}"),
    }
    let mut bw = film("kodak_portra_400");
    bw.treatment = Treatment::Bw;
    let r = render(&src, &raw(), &bw, &RenderRequest::fit(120, 120));
    let (w, h) = (r.image.width, r.image.height);
    let px = r.image.data[(h / 2) * w + w / 2];
    assert!(px[0] == px[1] && px[1] == px[2], "B&W still converts: {px:?}");
}
