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
        // a non-boolean `enabled` is refused, not read as "toggle" (which would turn film on here)
        json!({"enabled": "false"}),
        json!({"enabled": 0, "stock": "kodak_ektar_100"}),
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
