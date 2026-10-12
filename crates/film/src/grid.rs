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
        // wasm32 can't spawn threads, so there a single "thread" bakes inline
        let threads = if cfg!(target_arch = "wasm32") { 1 } else { std::thread::available_parallelism().map_or(1, |v| v.get()).clamp(1, n) };
        let per = n.div_ceil(threads) * n * n;
        let fill = |t: usize, chunk: &mut [[f32; 3]]| {
            for (j, out) in chunk.iter_mut().enumerate() {
                let idx = t * per + j;
                let (x, y, z) = (idx % n, (idx / n) % n, idx / (n * n));
                let p = [lo[0] + x as f64 * step[0], lo[1] + y as f64 * step[1], lo[2] + z as f64 * step[2]];
                *out = f(p).map(|v| if v.is_finite() { v as f32 } else { 0.0 });
            }
        };
        if threads == 1 {
            fill(0, &mut data);
        } else {
            let fill = &fill;
            let ok = std::thread::scope(|s| {
                let mut ok = true;
                let mut handles = Vec::with_capacity(threads);
                for (t, chunk) in data.chunks_mut(per).enumerate() {
                    // `Builder::spawn_scoped` reports a failed spawn as an error, where `Scope::spawn` would panic
                    match std::thread::Builder::new().spawn_scoped(s, move || fill(t, chunk)) {
                        Ok(h) => handles.push(h),
                        Err(_) => ok = false,
                    }
                }
                // every handle joined (not short-circuiting), so a panicking worker is an error instead of a panic
                // in `scope`
                for h in handles {
                    ok &= h.join().is_ok();
                }
                ok
            });
            if !ok {
                return Err(FilmError::Profile("baking the film look failed".into()));
            }
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
