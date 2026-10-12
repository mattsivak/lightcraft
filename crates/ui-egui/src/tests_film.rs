//! Headless tests of the Edit panel's Film section (issue #988).

use std::time::Duration;

use serde_json::json;

use crate::headless::Headless;
use crate::{LightcraftApp, Services};

const T: Duration = Duration::from_secs(20);
const SETTLE: Duration = Duration::from_secs(120);

/// Detail view with the Edit panel and only its Film section open (sections start collapsed).
fn edit_panel() -> Headless {
    let services = Services { png: None, ..Default::default() };
    let app = LightcraftApp::new(lightcraft_engine::Session::with_demo(), services);
    let mut h = Headless::new(app, [1200.0, 1600.0], 2.0);
    let r = h.request("ui.set", json!({"view": "detail", "right": "edit", "openSections": ["film"]}), T);
    assert_eq!(r["ok"], true, "{r}");
    h
}

fn on_screen(h: &Headless, id: &str) -> bool {
    h.app.widgets.iter().any(|(w, _)| w == id)
}

fn film(h: &mut Headless, params: serde_json::Value) {
    let r = h.request("engine.execute", json!({"command": "develop.film", "params": params}), T);
    assert_eq!(r["ok"], true, "{r}");
    // the first frame after the change may still lay out the previous settings
    h.snapshot(SETTLE);
    h.snapshot(SETTLE);
}

/// Edit ▸ Film keeps to the panel's width: a long stock or paper name ("Kodak Professional Portra Endura") is cut
/// short instead of widening the section, so the film sliders line up with every other slider.
#[test]
fn film_section_never_widens_the_panel() {
    let mut h = edit_panel();
    let r = h.request("ui.set", json!({"openSections": ["light", "film"]}), T);
    assert_eq!(r["ok"], true, "{r}");
    film(&mut h, json!({"enabled": true, "stock": "kodak_ektar_100", "output": "print"}));
    let right = |h: &Headless, id: &str| h.app.widgets.iter().find(|(w, _)| w == id).map(|(_, r)| r.right()).unwrap();
    let edge = right(&h, "slider:light.exposure");
    for id in ["slider:film.exposure", "slider:film.printExposure", "slider:film.printMagenta", "slider:film.printYellow"] {
        assert!((right(&h, id) - edge).abs() < 0.5, "{id} ends at {} but Light ▸ Exposure at {edge}", right(&h, id));
    }
    for id in ["combo:filmStock", "combo:filmPaper"] {
        assert!(right(&h, id) <= edge + 0.5, "{id} ends at {}, past the sliders at {edge}", right(&h, id));
    }
}

/// Edit ▸ Film: choosing a negative turns film simulation on as a lab scan with the Lab scan / Print switch; Print
/// adds the paper and print controls; a slide film has neither. Contrast, Whites and Blacks stay usable.
#[test]
fn film_section_shows_the_controls_for_the_stock() {
    let mut h = edit_panel();
    let r = h.request("ui.set", json!({"openSections": ["light", "film"]}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.snapshot(SETTLE);
    assert!(on_screen(&h, "combo:filmStock"));
    assert!(!on_screen(&h, "slider:film.exposure"), "film off: no sliders");
    film(&mut h, json!({"enabled": true, "stock": "kodak_gold_200"}));
    for id in ["slider:film.exposure", "button:filmOutput-scan", "button:filmOutput-print"] {
        assert!(on_screen(&h, id), "{id} not on screen");
    }
    assert!(!on_screen(&h, "combo:filmPaper") && !on_screen(&h, "slider:film.printExposure"), "a lab scan has no paper");
    let r = h.request("ui.clickWidget", json!({"id": "button:filmOutput-print"}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.snapshot(SETTLE);
    h.snapshot(SETTLE);
    for id in ["combo:filmPaper", "slider:film.printExposure", "slider:film.printMagenta", "slider:film.printYellow"] {
        assert!(on_screen(&h, id), "{id} not on screen after choosing Print");
    }
    film(&mut h, json!({"stock": "fujifilm_velvia_100"}));
    assert!(on_screen(&h, "slider:film.exposure"));
    assert!(!on_screen(&h, "button:filmOutput-print") && !on_screen(&h, "combo:filmPaper"), "a slide is viewed directly");
    // the Light sliders shape the light before the film: not greyed out
    let r = h.request("ui.clickWidget", json!({"id": "slider:light.contrast", "fx": 0.8}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.snapshot(SETTLE);
    let id = h.app.session.active().unwrap();
    assert!(h.app.session.develop_of(id).unwrap().light.contrast > 10.0, "Contrast moves with film on");
}
