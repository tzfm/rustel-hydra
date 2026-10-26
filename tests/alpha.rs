//! The renderer preserves Hydra's premultiplied alpha from `luma`, `mask`
//! and `layer`. The terminal can then composite shapes over its background.
//!
//! These tests measure the rendered output because valid shader source alone
//! does not prove that transparency survives the complete pipeline.

mod common;

use std::time::{Duration, Instant};

use common::renderer_is_available;
use rustel_hydra::HydraHost;

/// The share of a rendered frame that is fully opaque, as a percentage.
fn opaque_percent(code: &str) -> u8 {
    let mut host = HydraHost::new();
    host.preview(Some(code))
        .expect("a snippet the parser accepts");
    let frames = host.preview_frames();
    let until = Instant::now() + Duration::from_secs(25);
    while Instant::now() < until {
        if let Some((_, _, pixels)) = frames.take() {
            // Wait for a frame that is a picture: a device that has only just
            // opened delivers black before it delivers a sketch.
            let lit = pixels.iter().step_by(4).filter(|value| **value > 8).count();
            if lit > pixels.len() / 400 {
                let count = pixels.len() / 4;
                let opaque = pixels
                    .iter()
                    .skip(3)
                    .step_by(4)
                    .filter(|a| **a == 255)
                    .count();
                return (opaque * 100 / count.max(1)) as u8;
            }
        }
        std::thread::sleep(Duration::from_millis(40));
    }
    panic!("no lit frame for `{code}`");
}

#[test]
fn a_transform_that_writes_alpha_reaches_the_terminal_with_it() {
    if !renderer_is_available() {
        println!("no adapter and no software rasteriser - skipping");
        return;
    }

    // An oscillator fills the frame. Every pixel is opaque, and a backdrop of
    // it covers the screen - which is what it does upstream too.
    assert_eq!(
        opaque_percent("osc(20, 0.1, 1.2).out()"),
        100,
        "osc writes vec4(r, g, b, a) with a of 1"
    );

    // `luma` keeps the bright half and drops the rest to nothing:
    // `vec4(rgb * a, a * c0.a)`. Roughly half the frame must come back thin.
    let luma = opaque_percent("osc(20, 0.1, 1.2).luma().out()");
    assert!(
        (10..90).contains(&luma),
        "luma leaves a shape, not a full frame: {luma}% opaque"
    );
}
