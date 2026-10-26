//! Every transform Hydra ships, composed, uplifted and rendered on whatever
//! this machine has - a GPU, or a software rasteriser when it has none.
//!
//! The assertion throughout is a frame that is actually a picture, not merely
//! a frame: a shader that fails to compile and one that draws black are the
//! same thing from the outside.

use rustel_hydra::glsl::{FUNCTIONS, Kind};
use rustel_hydra::native::NativeRenderer;
use rustel_hydra::{HydraCall, HydraNode};

fn global(name: &str) -> HydraNode {
    HydraNode::Global { name: name.into() }
}

fn n(v: f64) -> HydraNode {
    HydraNode::Number { v }
}

/// A chain that exercises one transform, whatever kind it is.
fn exercise(name: &str, kind: Kind) -> HydraNode {
    let osc = |calls| HydraNode::Chain {
        head: "osc".into(),
        args: vec![n(10.0), n(0.1), n(0.3)],
        calls,
    };
    match kind {
        // `src` needs a texture to read; `s0` is bound and black until
        // something fills it, which is enough to prove it compiles and draws.
        Kind::Src if name == "src" => HydraNode::Chain {
            head: "src".into(),
            args: vec![HydraNode::Global { name: "s0".into() }],
            calls: Vec::new(),
        },
        Kind::Src => HydraNode::Chain {
            head: name.into(),
            args: Vec::new(),
            calls: Vec::new(),
        },
        Kind::Coord | Kind::Color => osc(vec![HydraCall {
            method: name.into(),
            args: Vec::new(),
        }]),
        Kind::Combine | Kind::CombineCoord => osc(vec![HydraCall {
            method: name.into(),
            args: vec![HydraNode::Chain {
                head: "noise".into(),
                args: vec![n(3.0)],
                calls: Vec::new(),
            }],
        }]),
    }
}

#[test]
fn every_transform_in_the_table_renders() {
    let mut renderer = match NativeRenderer::new(64, 64) {
        Ok(renderer) => renderer,
        Err(error) => {
            // A machine with neither a GPU nor a software rasteriser proves
            // nothing here, and should not fail the suite for it.
            eprintln!("skipped: {error}");
            return;
        }
    };
    eprintln!("rendering on {}", renderer.adapter());

    let mut drew = 0;
    let mut refused = Vec::new();
    for entry in FUNCTIONS {
        let node = exercise(entry.name, entry.kind);
        match renderer.render(&node, 1.5) {
            Ok(pixels) => {
                assert_eq!(
                    pixels.len(),
                    64 * 64 * 4,
                    "{}: a full RGBA frame",
                    entry.name
                );
                drew += 1;
            }
            Err(error) => refused.push(format!("{}: {error}", entry.name)),
        }
    }

    // `sum()` is hydra's own bug, not ours: its body reads `s` while its input
    // is named `scale`, and a stray brace smuggles in a second overload. Fixed
    // upstream on 2026-07-09 and unreleased, so the pinned 1.4.0 cannot
    // compile it either.
    assert_eq!(refused.len(), 1, "unexpected refusals: {refused:?}");
    assert!(refused[0].starts_with("sum:"), "{refused:?}");
    assert!(drew >= 50, "only {drew} transforms drew");
}

/// Feedback: a sketch reads the output it draws into. One pass cannot sample
/// and write the same texture, so each output keeps two and swaps them. A
/// wrong swap leaves the picture black or stops it evolving.
#[test]
fn a_sketch_can_read_the_output_it_draws_into() {
    let Ok(mut renderer) = NativeRenderer::new(64, 64) else {
        eprintln!("skipped: nothing to render on");
        return;
    };

    // Frame one: something to feed back.
    let seed = HydraNode::Chain {
        head: "osc".into(),
        args: vec![n(20.0), n(0.1), n(0.8)],
        calls: vec![HydraCall {
            method: "out".into(),
            args: vec![global("o0")],
        }],
    };
    renderer.draw(&seed, 0.0).expect("seed frame");
    let first = renderer.read(0).expect("read");

    // Then repeatedly feed o0 back through itself, brightened. If the swap
    // works the picture changes; if o0 read black every time it could not.
    let feedback = HydraNode::Chain {
        head: "src".into(),
        args: vec![global("o0")],
        calls: vec![
            HydraCall {
                method: "scrollX".into(),
                args: vec![n(0.01)],
            },
            HydraCall {
                method: "out".into(),
                args: vec![global("o0")],
            },
        ],
    };
    for step in 0..4 {
        renderer
            .draw(&feedback, step as f32 * 0.1)
            .expect("feedback frame");
    }
    let after = renderer.read(0).expect("read");

    let lit = |pixels: &[u8]| pixels.iter().step_by(4).filter(|v| **v > 8).count();
    assert!(lit(&first) > first.len() / 400, "the seed frame is black");
    assert!(
        lit(&after) > after.len() / 400,
        "feedback read black: the output never carried forward"
    );
    assert_ne!(
        first, after,
        "four scrolls of feedback left the picture untouched"
    );
}

/// `.out(o2)` draws into o2, and o0 is left alone.
#[test]
fn a_chain_lands_in_the_output_it_names() {
    let Ok(mut renderer) = NativeRenderer::new(32, 32) else {
        eprintln!("skipped: nothing to render on");
        return;
    };
    let into = |name: &str| HydraNode::Chain {
        head: "osc".into(),
        args: vec![n(20.0), n(0.1), n(0.8)],
        calls: vec![HydraCall {
            method: "out".into(),
            args: vec![global(name)],
        }],
    };
    renderer.draw(&into("o2"), 0.0).expect("draw into o2");
    let lit = |pixels: &[u8]| pixels.iter().step_by(4).filter(|v| **v > 8).count();
    let o2 = renderer.read(2).expect("read o2");
    let o0 = renderer.read(0).expect("read o0");
    assert!(lit(&o2) > o2.len() / 400, "o2 has the picture");
    assert_eq!(lit(&o0), 0, "o0 was never drawn into");
}

#[test]
fn bare_render_uses_a_display_target_without_mutating_o0() {
    let Ok(mut renderer) = NativeRenderer::new(32, 32) else {
        eprintln!("skipped: nothing to render on");
        return;
    };
    let solid = |colour: [f64; 4], output: &str| HydraNode::Chain {
        head: "solid".into(),
        args: colour.into_iter().map(n).collect(),
        calls: vec![HydraCall {
            method: "out".into(),
            args: vec![global(output)],
        }],
    };
    renderer
        .draw(&solid([1.0, 0.0, 0.0, 1.0], "o0"), 0.0)
        .expect("seed o0");
    renderer
        .draw(&solid([0.0, 1.0, 0.0, 1.0], "o1"), 0.0)
        .expect("seed o1");
    let before = renderer.read(0).expect("read o0 before render()");

    renderer.render_all().expect("compose four outputs");
    let display = renderer.read_display().expect("read four-up display");
    let after = renderer.read(0).expect("read o0 after render()");

    assert_eq!(before, after, "render() must not write or swap o0");
    assert_ne!(display, after, "the four-up canvas is a separate picture");
    assert!(
        display
            .as_chunks::<4>()
            .0
            .iter()
            .any(|pixel| pixel[1] > 200 && pixel[0] < 20),
        "the display includes o1's green quadrant"
    );
}

#[test]
fn prev_buffer_tracks_the_nonzero_output_being_drawn() {
    let Ok(mut renderer) = NativeRenderer::new(32, 32) else {
        eprintln!("skipped: nothing to render on");
        return;
    };
    let seed = HydraNode::Chain {
        head: "solid".into(),
        args: vec![n(0.9), n(0.2), n(0.1), n(1.0)],
        calls: vec![HydraCall {
            method: "out".into(),
            args: vec![global("o2")],
        }],
    };
    renderer.draw(&seed, 0.0).expect("seed o2");
    let expected = renderer.read(2).expect("read seed");
    let previous = HydraNode::Chain {
        head: "prev".into(),
        args: Vec::new(),
        calls: vec![HydraCall {
            method: "out".into(),
            args: vec![global("o2")],
        }],
    };
    renderer
        .draw(&previous, 0.0)
        .expect("copy o2's previous buffer back to o2");
    let actual = renderer.read(2).expect("read copied previous buffer");
    assert_eq!(actual, expected, "prevBuffer for o2 must not alias o0");
}

#[test]
fn hush_clears_both_halves_of_every_output() {
    let Ok(mut renderer) = NativeRenderer::new(16, 16) else {
        eprintln!("skipped: nothing to render on");
        return;
    };
    let seed = HydraNode::Chain {
        head: "solid".into(),
        args: vec![n(1.0), n(1.0), n(1.0), n(1.0)],
        calls: vec![HydraCall {
            method: "out".into(),
            args: vec![global("o3")],
        }],
    };
    renderer.draw(&seed, 0.0).expect("seed o3");
    assert!(renderer.read(3).unwrap().iter().any(|byte| *byte != 0));
    renderer
        .set_source(0, 1, 1, &[255, 255, 255, 255])
        .expect("seed s0");
    renderer.hush();
    let is_transparent = |pixels: &[u8]| pixels.iter().all(|byte| *byte == 0);
    assert!(
        is_transparent(&renderer.read(3).unwrap()),
        "hush clears the readable half to transparent black"
    );
    renderer
        .draw(
            &HydraNode::Chain {
                head: "prev".into(),
                args: Vec::new(),
                calls: vec![HydraCall {
                    method: "out".into(),
                    args: vec![global("o3")],
                }],
            },
            0.0,
        )
        .expect("draw cleared previous half");
    assert!(
        is_transparent(&renderer.read(3).unwrap()),
        "hush clears the feedback half too"
    );
    let source = HydraNode::Chain {
        head: "src".into(),
        args: vec![global("s0")],
        calls: Vec::new(),
    };
    assert!(
        renderer
            .render(&source, 0.0)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .all(|pixel| pixel[..3] == [0, 0, 0]),
        "hush clears external source textures too"
    );
}

/// `feedStrudel`'s half: a picture goes into `s0` and a sketch samples it.
#[test]
fn a_source_picture_reaches_the_shader() {
    let Ok(mut renderer) = NativeRenderer::new(32, 32) else {
        eprintln!("skipped: nothing to render on");
        return;
    };
    let read = HydraNode::Chain {
        head: "src".into(),
        args: vec![global("s0")],
        calls: Vec::new(),
    };

    // Nothing has filled s0, so it is black and so is the sketch.
    let blank = renderer.render(&read, 0.0).expect("renders");
    assert_eq!(
        blank.iter().step_by(4).filter(|v| **v > 8).count(),
        0,
        "an unfilled source is black"
    );

    // Fill the top half red and the bottom half blue, then read it back. The
    // halves must not swap: a terminal writes its top row first, hydra counts
    // from the bottom, and `uplift` flips the y of every texture fetch.
    let (w, h) = (8u32, 8u32);
    let mut picture = Vec::new();
    for row in 0..h {
        for _ in 0..w {
            let top = row < h / 2;
            picture.extend_from_slice(if top {
                &[255, 0, 0, 255]
            } else {
                &[0, 0, 255, 255]
            });
        }
    }
    renderer.set_source(0, w, h, &picture).expect("fills s0");
    let filled = renderer.render(&read, 0.0).expect("renders");
    assert!(
        filled.iter().step_by(4).filter(|v| **v > 8).count() > 0,
        "a filled source reaches the shader"
    );

    // `read` hands back rows top first, the order the terminal draws them, so
    // frame row 0 is the TOP of the picture and held red. Measured against
    // hydra's own arithmetic in the same buffer: `gradient()` puts green on
    // `st.y`, `st.y` is zero at the bottom of the screen, and that gradient
    // reads green-high in row 0 too.
    let row = 32 * 4;
    let top = &filled[..row];
    let bottom = &filled[filled.len() - row..];
    let redder = |band: &[u8]| {
        band.as_chunks::<4>()
            .0
            .iter()
            .map(|p| u32::from(p[0]))
            .sum::<u32>()
            > band
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| u32::from(p[2]))
                .sum::<u32>()
    };
    assert!(redder(top), "the top of the frame is the red half");
    assert!(!redder(bottom), "the bottom of the frame is the blue half");

    // The same fetch, on a render target rather than an uploaded picture:
    // copying o0 through `src` must not turn it over either.
    let gradient = HydraNode::Chain {
        head: "gradient".into(),
        args: vec![n(0.0)],
        calls: vec![HydraCall {
            method: "out".into(),
            args: vec![global("o0")],
        }],
    };
    renderer.draw(&gradient, 0.0).expect("gradient to o0");
    let drawn = renderer.read(0).expect("read o0");
    let copy = HydraNode::Chain {
        head: "src".into(),
        args: vec![global("o0")],
        calls: vec![HydraCall {
            method: "out".into(),
            args: vec![global("o1")],
        }],
    };
    renderer.draw(&copy, 0.0).expect("src(o0) to o1");
    let copied = renderer.read(1).expect("read o1");
    let green = |band: &[u8]| {
        band.as_chunks::<4>()
            .0
            .iter()
            .map(|p| u32::from(p[1]))
            .sum::<u32>()
    };
    assert!(
        green(&drawn[..row]) > green(&drawn[drawn.len() - row..]),
        "hydra's st.y is zero at the bottom, so gradient() is greenest at the top"
    );
    assert!(
        green(&copied[..row]) > green(&copied[copied.len() - row..]),
        "src(o0) copied the frame upside down"
    );

    // A picture that lies about its size is refused rather than read past.
    assert!(renderer.set_source(0, 64, 64, &picture).is_err());
    let mut extra = picture.clone();
    extra.extend_from_slice(&[0, 0, 0, 0]);
    assert!(
        renderer.set_source(0, w, h, &extra).is_err(),
        "a trailing pixel is not silently ignored"
    );
    assert!(
        renderer.set_source(4, w, h, &picture).is_err(),
        "an invalid source index is not clamped to s3"
    );

    renderer.clear_source(0).expect("clear s0");
    let cleared = renderer.render(&read, 0.0).expect("renders cleared source");
    assert_eq!(
        cleared.iter().step_by(4).filter(|v| **v > 8).count(),
        0,
        "clearing a source returns it to black"
    );
}

/// `H(pattern)` reaches the picture, and changing it changes the picture.
#[test]
fn a_signal_drives_the_shader() {
    let Ok(mut renderer) = NativeRenderer::new(32, 32) else {
        eprintln!("skipped: nothing to render on");
        return;
    };
    // `shape(H(0))` - the number of sides comes from the signal.
    let sketch = HydraNode::Chain {
        head: "shape".into(),
        args: vec![HydraNode::Signal { slot: 0 }, n(0.6), n(0.01)],
        calls: Vec::new(),
    };

    renderer.set_signals(&[3.0]);
    let three = renderer.render(&sketch, 0.0).expect("renders");
    renderer.set_signals(&[8.0]);
    let eight = renderer.render(&sketch, 0.0).expect("renders");

    let lit = |p: &[u8]| p.iter().step_by(4).filter(|v| **v > 8).count();
    assert!(lit(&three) > 0, "a triangle drew nothing");
    assert!(lit(&eight) > 0, "an octagon drew nothing");
    assert_ne!(
        three, eight,
        "the signal never reached the shader: both sizes drew the same shape"
    );
    // The signal arrives as the side count. In hydra's `shape`, `radius` is
    // the apothem, so fewer sides means more area: a triangle is 5.196 r^2
    // and an octagon is 3.314 r^2, a ratio of 1.57.
    let ratio = lit(&three) as f64 / lit(&eight) as f64;
    assert!(
        (1.35..1.8).contains(&ratio),
        "triangle {} over octagon {} is {ratio:.2}, expected about 1.57",
        lit(&three),
        lit(&eight)
    );
}

#[test]
fn the_seventeenth_signal_keeps_its_protocol_slot_on_the_gpu() {
    let Ok(mut renderer) = NativeRenderer::new(16, 16) else {
        eprintln!("skipped: nothing to render on");
        return;
    };
    let sketch = HydraNode::Chain {
        head: "solid".into(),
        args: vec![HydraNode::Signal { slot: 16 }, n(0.0), n(0.0), n(1.0)],
        calls: Vec::new(),
    };
    let mut signals = vec![0.0; 17];
    renderer.set_signals(&signals);
    let dark = renderer.render(&sketch, 0.0).expect("slot 16 compiles");
    signals[16] = 1.0;
    renderer.set_signals(&signals);
    let red = renderer.render(&sketch, 0.0).expect("slot 16 draws");
    let red_sum = |pixels: &[u8]| {
        pixels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|pixel| usize::from(pixel[0]))
            .sum::<usize>()
    };
    assert_eq!(red_sum(&dark), 0);
    assert!(
        red_sum(&red) > 0,
        "H signal 17 was truncated or read slot 0"
    );
}

#[test]
fn fft_bin_six_reaches_the_fixed_capacity_shader_uniform() {
    let Ok(mut renderer) = NativeRenderer::new(16, 16) else {
        eprintln!("skipped: nothing to render on");
        return;
    };
    let sketch = HydraNode::Chain {
        head: "solid".into(),
        args: vec![
            HydraNode::Source {
                src: "() => a.fft[6]".into(),
            },
            n(0.0),
            n(0.0),
            n(1.0),
        ],
        calls: Vec::new(),
    };
    let dark = renderer.render(&sketch, 0.0).expect("band 6 compiles");

    let mut bins = [0.0; rustel_hydra::HYDRA_AUDIO_BINS];
    bins[6] = 1.0;
    renderer.set_audio(bins);
    let red = renderer.render(&sketch, 0.0).expect("band 6 reaches GPU");
    let red_sum = |pixels: &[u8]| {
        pixels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|pixel| usize::from(pixel[0]))
            .sum::<usize>()
    };
    assert_eq!(red_sum(&dark), 0, "zero band draws no red");
    assert!(red_sum(&red) > 0, "band 6 never reached its uniform");
}

/// `render()` shows all four outputs at once, in quadrants.
#[test]
fn render_all_composites_the_four_outputs() {
    let Ok(mut renderer) = NativeRenderer::new(64, 64) else {
        eprintln!("skipped: nothing to render on");
        return;
    };
    // A different flat colour in each output, so a quadrant can be named by
    // the colour it carries.
    let solid = |r: f64, g: f64, b: f64, into: &str| HydraNode::Chain {
        head: "solid".into(),
        args: vec![n(r), n(g), n(b), n(1.0)],
        calls: vec![HydraCall {
            method: "out".into(),
            args: vec![global(into)],
        }],
    };
    renderer.draw(&solid(1.0, 0.0, 0.0, "o0"), 0.0).expect("o0");
    renderer.draw(&solid(0.0, 1.0, 0.0, "o1"), 0.0).expect("o1");
    renderer.draw(&solid(0.0, 0.0, 1.0, "o2"), 0.0).expect("o2");
    renderer.draw(&solid(1.0, 1.0, 0.0, "o3"), 0.0).expect("o3");
    renderer.render_all().expect("composite");
    let frame = renderer.read_display().expect("read four-up display");

    // Sample the middle of each quadrant. Rows come back bottom-first.
    let at = |x: usize, y: usize| {
        let i = (y * 64 + x) * 4;
        (frame[i], frame[i + 1], frame[i + 2])
    };
    let corners = [at(16, 16), at(48, 16), at(16, 48), at(48, 48)];
    let bright = |(r, g, b): (u8, u8, u8)| r > 128 || g > 128 || b > 128;
    assert!(
        corners.iter().all(|c| bright(*c)),
        "every quadrant carries an output: {corners:?}"
    );
    // Four outputs, four different colours, so four distinct quadrants.
    let mut seen: Vec<(u8, u8, u8)> = Vec::new();
    for corner in corners {
        if !seen.contains(&corner) {
            seen.push(corner);
        }
    }
    assert_eq!(seen.len(), 4, "four distinct quadrants: {corners:?}");
}

/// The footprint counts ten output textures at the render size, four
/// sources, the uniforms and a readback buffer with rows padded to wgpu's
/// alignment. It follows a resize, a source upload and a clear.
#[test]
fn the_footprint_counts_the_textures_and_buffers_and_follows_them() {
    let Ok(mut renderer) = NativeRenderer::new(64, 64) else {
        eprintln!("skipped: nothing to render on");
        return;
    };
    let uniforms = 16 + rustel_hydra::HYDRA_AUDIO_BINS * 4 + rustel_hydra::native::MAX_SIGNALS * 4;
    let sources = |texels: usize| texels * 4;

    let open = renderer.footprint();
    assert_eq!((open.width, open.height), (64, 64));
    assert_eq!(open.textures, 10 * 64 * 64 * 4 + sources(4) + uniforms);
    assert_eq!(
        open.readback,
        64 * 4 * 64,
        "a 256-byte row needs no padding"
    );
    assert_eq!(
        open.process_bytes() + open.gpu_bytes(),
        open.readback
            + if open.textures_in_process {
                open.resident
            } else {
                open.textures
            },
        "every byte is counted in one place or the other"
    );

    renderer.resize(100, 50).expect("resize");
    let resized = renderer.footprint();
    assert_eq!((resized.width, resized.height), (100, 50));
    assert_eq!(resized.textures, 10 * 100 * 50 * 4 + sources(4) + uniforms);
    assert_eq!(resized.readback, 512 * 50, "a 400-byte row pads to 512");

    renderer
        .set_source(2, 8, 4, &[0; 8 * 4 * 4])
        .expect("an image in s2");
    assert_eq!(
        renderer.footprint().textures,
        resized.textures - sources(1) + sources(8 * 4)
    );
    renderer.clear_source(2).expect("clear s2");
    assert_eq!(
        renderer.footprint(),
        resized,
        "a cleared source is one texel"
    );
}

/// Only written output textures count as resident. A one-output sketch
/// writes o0's back and zero-fills the four fronts by sampling them: five of
/// the ten. `hush()` writes all ten, and a resize starts over with none.
#[test]
fn the_footprint_counts_as_resident_only_the_textures_written() {
    let Ok(mut renderer) = NativeRenderer::new(64, 64) else {
        eprintln!("skipped: nothing to render on");
        return;
    };
    let texture = 64 * 64 * 4;
    let unwritten = renderer.footprint();
    let always = unwritten.resident;
    assert_eq!(
        unwritten.textures,
        always + 10 * texture,
        "no output written yet"
    );

    let sketch = HydraNode::Chain {
        head: "osc".into(),
        args: vec![n(10.0), n(0.1), n(0.8)],
        calls: Vec::new(),
    };
    renderer.draw(&sketch, 0.5).expect("draws into o0");
    assert_eq!(
        renderer.footprint().resident,
        always + 5 * texture,
        "o0's back and the four fronts"
    );
    renderer.draw(&sketch, 1.0).expect("draws into o0 again");
    assert_eq!(
        renderer.footprint().resident,
        always + 5 * texture,
        "o0's other half was already written, as a front"
    );

    renderer.hush();
    let hushed = renderer.footprint();
    assert_eq!(hushed.resident, hushed.textures, "all ten, cleared");
    if hushed.textures_in_process {
        assert_eq!(hushed.process_bytes(), hushed.readback + hushed.textures);
    }

    renderer.resize(32, 32).expect("resize");
    let resized = renderer.footprint();
    assert_eq!(
        resized.textures - resized.resident,
        10 * 32 * 32 * 4,
        "new textures, none written"
    );
}
