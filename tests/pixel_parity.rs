//! Does the native renderer draw what `hydra-synth` drew?
//!
//! `glsl_parity` proves we generate the same shader text. That is not the same
//! claim as the same picture: a different rasteriser, a different float
//! precision and a flipped `gl_FragCoord` origin could all agree on the source
//! and disagree on the pixels.
//!
//! So the reference frames beside this file were rendered by `hydra-synth`
//! itself, on the same chains. Each is 128×128 RGBA, deflated: a megabyte raw,
//! a tenth of that on disk. The recipe that regenerates them is not part of
//! this repository.

use rustel_hydra::glsl::parse_chain;
use rustel_hydra::native::NativeRenderer;

/// Chains chosen to be still.
///
/// hydra advances `time` on its own clock, so a chain that moves cannot be
/// compared frame to frame. `osc`, `noise`, `voronoi`, `gradient`, `rotate`
/// and the scrolls read `time` through a rate, and every rate here is zero.
/// That includes `osc`'s `sync`: its default gives a phase offset that reads
/// as a rendering difference.
const CHAINS: &[(&str, &str)] = &[
    ("osc", "osc(20, 0, 1.2).out()"),
    ("osc_kaleid", "osc(10, 0, 0.8).kaleid(5).out()"),
    ("noise", "noise(3, 0).out()"),
    ("shape", "shape(4, 0.3, 0.01).out()"),
    ("gradient", "gradient(0).out()"),
    ("voronoi", "voronoi(8, 0, 0.3).out()"),
    (
        "osc_rotate_color",
        "osc(30, 0, 0.5).rotate(0.7, 0).color(1, 0.6, 0.2).out()",
    ),
    (
        "shape_mult_osc",
        "shape(6, 0.4, 0.02).mult(osc(15, 0)).out()",
    ),
    (
        "osc_modulate_noise",
        "osc(12, 0, 0.9).modulate(noise(2, 0), 0.4).out()",
    ),
    ("osc_pixelate", "osc(25, 0, 1.0).pixelate(20, 20).out()"),
    ("noise_thresh", "noise(5, 0).thresh(0.5, 0.04).out()"),
    (
        "osc_invert_contrast",
        "osc(18, 0, 0.6).invert(1).contrast(1.6).out()",
    ),
    (
        "osc_luma_saturate",
        "osc(14, 0, 0.7).luma(0.4, 0.1).saturate(2).out()",
    ),
    (
        "shape_diff_osc",
        "shape(3, 0.5, 0.05).diff(osc(8, 0)).out()",
    ),
    (
        "modulate_rotate",
        "osc(16, 0, 0.4).modulateRotate(shape(4, 0.3, 0.02), 1.5, 0).out()",
    ),
    (
        "osc_scroll_repeat",
        "osc(22, 0, 0.9).scrollX(0.2, 0).repeat(2, 2).out()",
    ),
];

// The voronoi hash amplifies small GPU sin differences into different cells.
// Its reference pixels are comparable only on a matching renderer; GLSL parity
// still checks the shader on every host. The other chains, including polynomial
// simplex noise, do not use this transcendental hash.
const SIN_HASHED: &[&str] = &["voronoi"];

// Voronoi reference compatibility depends on the renderer's sin implementation,
// not the OS. Even two Linux software renderers produced different cells.
fn held_to_reference(name: &str) -> bool {
    !SIN_HASHED.contains(&name)
}

/// How many distinct colours a frame holds, which is how much picture is in
/// it: a blank frame answers 1. What is left to assert about a chain whose
/// reference cannot be reproduced - that it still drew one.
fn distinct_colours(frame: &[u8]) -> usize {
    let mut seen = std::collections::HashSet::new();
    for pixel in frame.as_chunks::<4>().0 {
        seen.insert([pixel[0], pixel[1], pixel[2]]);
    }
    seen.len()
}

/// Pearson correlation over the colour channels, which is what the audio
/// corpus judges by for the same reason: two renderers agreeing on shape
/// matters more than agreeing on the last bit of a float.
fn correlation(ours: &[u8], theirs: &[u8]) -> f64 {
    let pairs: Vec<(f64, f64)> = ours
        .as_chunks::<4>()
        .0
        .iter()
        .zip(theirs.as_chunks::<4>().0)
        .flat_map(|(a, b)| (0..3).map(move |c| (f64::from(a[c]), f64::from(b[c]))))
        .collect();
    let n = pairs.len() as f64;
    let (mx, my) = pairs
        .iter()
        .fold((0.0, 0.0), |(x, y), (a, b)| (x + a / n, y + b / n));
    let mut num = 0.0;
    let (mut dx, mut dy) = (0.0, 0.0);
    for (a, b) in &pairs {
        num += (a - mx) * (b - my);
        dx += (a - mx).powi(2);
        dy += (b - my).powi(2);
    }
    if dx == 0.0 || dy == 0.0 {
        return if dx == dy { 1.0 } else { 0.0 };
    }
    num / (dx.sqrt() * dy.sqrt())
}

#[test]
fn the_native_renderer_draws_what_hydra_drew() {
    let Ok(mut renderer) = NativeRenderer::new(128, 128) else {
        eprintln!("skipped: nothing to render on");
        return;
    };
    let mut worst: Option<(&str, f64)> = None;
    let mut checked = 0;

    for (name, code) in CHAINS {
        let reference = include_reference(name);
        let node = parse_chain(code).unwrap_or_else(|e| panic!("{name}: {e}"));
        renderer
            .draw(&node, 0.0)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let ours = renderer.read(0).expect("read");
        assert_eq!(ours.len(), reference.len(), "{name}: frame sizes differ");

        let r = correlation(&ours, &reference);
        let mean_delta = ours
            .as_chunks::<4>()
            .0
            .iter()
            .zip(reference.as_chunks::<4>().0)
            .flat_map(|(a, b)| (0..3).map(move |c| (f64::from(a[c]) - f64::from(b[c])).abs()))
            .sum::<f64>()
            / (ours.len() as f64 / 4.0 * 3.0);
        eprintln!("PIXELS {name}: correlation {r:.4}, mean channel delta {mean_delta:.1}/255");
        if !held_to_reference(name) {
            // Not comparable to the reference on this GPU, so what is left to
            // claim is the weaker thing that still means something: it drew a
            // picture rather than a wash or a blank frame.
            let colours = distinct_colours(&ours);
            assert!(
                colours > 64,
                "`{name}` drew only {colours} distinct colours, and its \
                 reference frame cannot be reproduced on this GPU to say more"
            );
            eprintln!("PIXELS {name}: drawn, not held to the reference here");
        } else if worst.is_none_or(|(_, w)| r < w) {
            worst = Some((name, r));
        }
        checked += 1;
    }

    let (name, r) = worst.expect("at least one chain");
    assert!(checked >= 8, "only {checked} chains compared");
    // Not a tolerance. Every one of these is byte-identical to what
    // `hydra-synth` renders, so anything less is a regression worth reading
    // rather than noise to absorb.
    assert!(
        r > 0.9999,
        "`{name}` correlates {r:.4} against hydra-synth's own render"
    );
}

fn include_reference(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/reference")
        .join(format!("{name}.rgba.z"));
    let packed = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    miniz_oxide::inflate::decompress_to_vec_zlib(&packed)
        .unwrap_or_else(|e| panic!("{}: {e:?}", path.display()))
}
