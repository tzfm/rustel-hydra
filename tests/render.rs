//! The host turns sketches into frames and reports each failure once.
//!
//! Covers the worker thread, composed shader, wgpu, readback and event
//! channel. Each test owns a host, thread and device, so the drawing checks
//! share one test and a check that needs a fresh host has its own.

mod common;

use std::time::{Duration, Instant};

use common::{default_sketch, renderer_is_available, sketch};
use rustel_hydra::glsl::parse_chain;
use rustel_hydra::{
    HYDRA_HEIGHT, HYDRA_SIGNAL_SAMPLES, HYDRA_SIGNAL_STEP_MS, HYDRA_WIDTH, HydraEvent, HydraHost,
    HydraOptions, HydraProgram, HydraSignalFrame, HydraStatement, HydraTuiCell, HydraTuiFrame,
};

/// Wait for a frame of the given width that is actually a picture.
fn lit_frame(host: &HydraHost, width: u16) -> (u16, u16) {
    let frames = host.frames();
    let mut seen: Vec<(u16, u16, usize)> = Vec::new();
    let until = Instant::now() + Duration::from_secs(25);
    while Instant::now() < until {
        if let Some((got, height, pixels)) = frames.take() {
            let lit = pixels.iter().step_by(4).filter(|v| **v > 8).count();
            if !seen.iter().any(|s| s.0 == got && s.1 == height) {
                seen.push((got, height, lit));
            }
            if got != width {
                std::thread::sleep(Duration::from_millis(40));
                continue;
            }
            assert_eq!(
                pixels.len(),
                usize::from(got) * usize::from(height) * 4,
                "a full RGBA frame"
            );
            if lit > pixels.len() / 400 {
                return (got, height);
            }
        }
        std::thread::sleep(Duration::from_millis(40));
    }
    // Say what actually arrived rather than only that the wanted thing did
    // not: a refused shader, a black picture and a frame at the wrong size
    // are indistinguishable from here otherwise.
    let said: Vec<String> = host
        .take_events()
        .iter()
        .map(|e| format!("{e:?}"))
        .collect();
    panic!("no lit frame at width {width}; saw {seen:?}; renderer said: {said:?}");
}

/// The same, for a stream with no host to ask.
fn lit_frame_from(frames: &rustel_hydra::HydraFrames, width: u16) -> (u16, u16) {
    let until = Instant::now() + Duration::from_secs(25);
    while Instant::now() < until {
        if let Some((got, height, pixels)) = frames.take()
            && got == width
        {
            let lit = pixels.iter().step_by(4).filter(|value| **value > 8).count();
            if lit > pixels.len() / 400 {
                return (got, height);
            }
        }
        std::thread::sleep(Duration::from_millis(40));
    }
    panic!("no lit frame at width {width}");
}

#[test]
fn a_sketch_renders_resizes_and_stops() {
    if !renderer_is_available() {
        eprintln!("skipped: no GPU and no software rasteriser");
        return;
    }
    let mut host = HydraHost::new();
    let frames = host.frames();
    assert!(!frames.wanted(), "nothing has been asked for yet");

    host.set_drawing(true);
    host.apply(default_sketch()).expect("install a sketch");
    assert_eq!(
        lit_frame(&host, HYDRA_WIDTH as u16),
        (HYDRA_WIDTH as u16, HYDRA_HEIGHT as u16)
    );
    assert!(frames.wanted(), "there is a picture to draw");

    // The renderer supplies ten more frames within the render timeout.
    // Elapsed time reports throughput without setting a shared-runner FPS limit.
    let started = Instant::now();
    let mut count = 0;
    while count < 10 && started.elapsed() < Duration::from_secs(25) {
        if frames.take().is_some() {
            count += 1;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    eprintln!("received {count} frames in {:?}", started.elapsed());
    assert_eq!(count, 10, "rendering stalled after {count} frames");

    // The two streams a live set feeds it are accepted without complaint.
    host.signals(&HydraSignalFrame {
        step: HYDRA_SIGNAL_STEP_MS,
        slots: vec![
            (0..HYDRA_SIGNAL_SAMPLES)
                .map(|index| serde_json::json!(index as f64 / 20.0))
                .collect(),
        ],
    });
    host.tui_sink()
        .publish(&HydraTuiFrame {
            cols: 4,
            rows: 2,
            cell_width: 8,
            cell_height: 16,
            cells: vec![HydraTuiCell::default(); 8],
        })
        .expect("valid terminal frame");

    // A surface that changes size keeps drawing.
    host.apply(sketch(960, 540)).expect("grow");
    assert_eq!(lit_frame(&host, 960), (960, 540));
    host.apply(sketch(256, 144)).expect("shrink");
    assert_eq!(lit_frame(&host, 256), (256, 144));

    // A stopped set wipes the picture rather than leaving the last frame up.
    host.set_drawing(false);
    assert!(!frames.wanted(), "a stopped set has no picture");
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        frames.take().is_none(),
        "and no frames arrive while stopped"
    );

    // Playing again brings it back.
    host.set_drawing(true);
    assert_eq!(lit_frame(&host, 256), (256, 144));

    // `noise()` must actually draw: it collapses to black under `mediump` on
    // some drivers with no error, which would quietly empty most of the
    // sketches anyone reaches for.
    let mut noisy = sketch(320, 180);
    noisy.statements = vec![rustel_hydra::HydraStatement::Evaluate {
        node: rustel_hydra::HydraNode::Chain {
            head: "noise".into(),
            args: vec![rustel_hydra::HydraNode::Number { v: 3.0 }],
            calls: vec![rustel_hydra::HydraCall {
                method: "out".into(),
                args: Vec::new(),
            }],
        },
    }];
    host.apply(noisy).expect("install a noise sketch");
    lit_frame(&host, 320);

    // The shelf draws through a second renderer, so browsing snippets does
    // not take the screen away from the score. Both streams must deliver at
    // once, each at its own size.
    let shelf = host.preview_frames();
    assert!(!shelf.wanted(), "nothing is being browsed yet");
    host.preview(Some("osc(20, 0.1, 1.2).kaleid(5).out()"))
        .expect("preview a snippet");
    assert!(shelf.wanted(), "the shelf has a picture of its own");
    // Different sizes from the same moment: the shelf's square thumbnail and
    // the score's 16:9 backdrop cannot both come from one canvas.
    assert_eq!(lit_frame_from(&shelf, 320), (320, 320));
    assert_eq!(lit_frame(&host, 320), (320, 180));

    // Closing the shelf takes its picture with it and leaves the score's.
    host.preview(None).expect("stop previewing");
    assert!(!shelf.wanted(), "the shelf is closed");
    std::thread::sleep(Duration::from_millis(300));
    assert!(shelf.take().is_none(), "and no frames arrive from it");
    assert_eq!(lit_frame(&host, 320), (320, 180));

    // A snippet runs as the source text it is. The score path is a recording
    // of nodes, because the score realm records and does not evaluate.
    let mut source = HydraProgram::from_source("osc(20, 0.1, 1.2).kaleid(5).out()");
    source.options.width = 288;
    source.options.height = 288;
    assert!(!source.is_empty(), "a source sketch is something to draw");
    host.apply(source).expect("run a sketch built from source");
    lit_frame(&host, 288);

    // The terminal asks for frames the size it will actually read. On the
    // cell path that is one pixel per character, so a full render crossing the
    // wire is a megabyte encoded to use forty kilobytes of it - measured at
    // 5.2ms a frame in base64 alone, against 0.05ms once it is scaled first.
    host.apply(sketch(640, 360)).expect("a full-size render");
    host.set_frame_size(200, 50);
    let until = Instant::now() + Duration::from_secs(20);
    let mut delivered = None;
    while Instant::now() < until {
        if let Some((width, height, pixels)) = frames.take()
            && width == 200
        {
            delivered = Some((width, height, pixels.len()));
            break;
        }
        std::thread::sleep(Duration::from_millis(40));
    }
    assert_eq!(
        delivered,
        Some((200, 50, 200 * 50 * 4)),
        "the wire carries what the terminal reads, not what hydra rendered"
    );

    // `contextType: 'webgl2'` must not take the picture down. hydra-synth's
    // regl only ever asks for WebGL 1, so claiming a WebGL 2 context first
    // made regl's own `getContext` return null and threw in `_initRegl`,
    // before a single frame. The option is accepted and ignored.
    let mut two = sketch(288, 162);
    two.options.context_type = "webgl2".into();
    host.apply(two).expect("install a webgl2-flavoured sketch");
    host.set_frame_size(288, 162);
    assert_eq!(lit_frame(&host, 288), (288, 162));
    host.apply(sketch(320, 180)).expect("back to the default");
    host.set_frame_size(200, 50);

    // Smoothing must actually reach the picture, not merely survive being
    // switched on. A high-frequency sketch is the case that shows it: fine
    // stripes decimated to a 200-wide grid alias hard, and averaging the
    // source rectangle instead flattens them. Assert on that flattening,
    // because a test that only checks the frame is still lit passes whether
    // or not the flag ever arrived.
    let mut stripes = sketch(640, 360);
    stripes.statements = vec![rustel_hydra::HydraStatement::Evaluate {
        node: rustel_hydra::HydraNode::Chain {
            head: "osc".into(),
            args: vec![
                // Fine enough that a stripe is narrower than the box each
                // destination pixel averages: at 640 wide into 200, that box
                // is three columns by seven rows. Measured across candidates,
                // this one flattens to 0.29 of its point-sampled spread,
                // against 0.98 for a switch that never arrived.
                rustel_hydra::HydraNode::Number { v: 2000.0 },
                rustel_hydra::HydraNode::Number { v: 0.0 },
                rustel_hydra::HydraNode::Number { v: 0.0 },
            ],
            calls: vec![rustel_hydra::HydraCall {
                method: "out".into(),
                args: Vec::new(),
            }],
        },
    }];
    host.apply(stripes)
        .expect("install a high-frequency sketch");
    host.set_frame_size(200, 50);

    /// How much the picture varies, so that flatter reads as smaller.
    fn spread(pixels: &[u8]) -> f64 {
        let luma: Vec<f64> = pixels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| 0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2]))
            .collect();
        let mean = luma.iter().sum::<f64>() / luma.len() as f64;
        (luma.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / luma.len() as f64).sqrt()
    }

    let settle = |host: &HydraHost| {
        // Frames already in flight were produced under the old setting.
        let until = Instant::now() + Duration::from_millis(400);
        while Instant::now() < until {
            host.frames().take();
            std::thread::sleep(Duration::from_millis(20));
        }
        let until = Instant::now() + Duration::from_secs(20);
        while Instant::now() < until {
            if let Some((width, _, pixels)) = host.frames().take()
                && width == 200
            {
                return spread(&pixels);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        panic!("no frame to measure");
    };

    host.set_smoothing(false);
    let sharp = settle(&host);
    host.set_smoothing(true);
    let soft = settle(&host);
    host.set_smoothing(false);
    assert!(
        soft < sharp * 0.6,
        "averaging must visibly flatten a striped sketch: {soft:.1} against {sharp:.1}"
    );

    // ...and the picture is the same way up as it always was. `gradient()`
    // ramps green along y, so row zero being the bright end is what the
    // terminal's sampler has always been handed.
    let mut ramp = sketch(64, 64);
    ramp.statements = vec![rustel_hydra::HydraStatement::Evaluate {
        node: rustel_hydra::HydraNode::Chain {
            head: "gradient".into(),
            args: vec![rustel_hydra::HydraNode::Number { v: 0.0 }],
            calls: vec![rustel_hydra::HydraCall {
                method: "out".into(),
                args: Vec::new(),
            }],
        },
    }];
    host.apply(ramp).expect("install a gradient");
    host.set_frame_size(64, 64);
    let until = Instant::now() + Duration::from_secs(20);
    let mut ends = None;
    while Instant::now() < until {
        if let Some((width, height, pixels)) = frames.take()
            && width == 64
            && height == 64
        {
            let green = |row: usize| u32::from(pixels[row * 64 * 4 + 1]);
            ends = Some((green(0), green(63)));
            break;
        }
        std::thread::sleep(Duration::from_millis(40));
    }
    let (top, bottom) = ends.expect("a gradient frame");
    assert!(
        top > bottom + 100,
        "row zero is the bright end of the ramp, as it was before scaling: {top} vs {bottom}"
    );

    // And a score that draws nothing stops the whole thing.
    host.apply(HydraProgram::default())
        .expect("an empty sketch is not an error");
    let until = Instant::now() + Duration::from_secs(5);
    while frames.wanted() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(!frames.wanted(), "an empty sketch stops drawing");
}

/// `feedStrudel`: the terminal itself becomes the texture `s0` samples.
#[test]
fn the_terminal_reaches_a_sketch_that_asked_for_it() {
    if !renderer_is_available() {
        eprintln!("skipped: no GPU and no software rasteriser");
        return;
    }
    use rustel_hydra::{HydraTuiCell, HydraTuiFrame};

    let mut host = HydraHost::new();
    host.set_drawing(true);

    // A sketch that draws nothing BUT the feed, so what comes back is the
    // terminal or it is black.
    let mut program = rustel_hydra::HydraProgram::from_source("src(s0).out()");
    program.options.feed_strudel = true;
    program.options.width = 128;
    program.options.height = 128;
    host.apply(program).expect("apply");
    host.set_frame_size(128, 128);

    let sink = host.tui_sink();
    assert!(sink.wanted(), "the sketch asked for the feed");

    // A grid of bright letters on black.
    let (cols, rows) = (16u16, 8u16);
    let cells = (0..usize::from(cols) * usize::from(rows))
        .map(|index| HydraTuiCell {
            code: u32::from(b'A') + (index % 26) as u32,
            fg: [255, 255, 255],
            bg: [0, 0, 0],
            flags: 0,
        })
        .collect();
    sink.publish(&HydraTuiFrame {
        cols,
        rows,
        cell_width: 8,
        cell_height: 16,
        cells,
    })
    .expect("valid terminal frame");

    let frames = host.frames();
    let until = Instant::now() + Duration::from_secs(20);
    let mut lit = 0;
    while Instant::now() < until {
        if let Some((width, _, pixels)) = frames.take()
            && width == 128
        {
            lit = pixels.iter().step_by(4).filter(|value| **value > 8).count();
            if lit > 0 {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(40));
    }
    assert!(
        lit > 0,
        "a sketch reading s0 drew nothing: the terminal never reached it"
    );
}

/// Wait for what the host says it holds to come to something, and hand
/// it back; say what it read instead if it never does.
fn settles(
    host: &HydraHost,
    within: Duration,
    wanted: impl Fn(&rustel_hydra::HydraMemory) -> bool,
) -> rustel_hydra::HydraMemory {
    let until = Instant::now() + within;
    loop {
        let memory = host.memory();
        if wanted(&memory) {
            return memory;
        }
        assert!(Instant::now() < until, "the host held {memory:?}");
        std::thread::sleep(Duration::from_millis(40));
    }
}

/// The memory breakdown counts a renderer at its render size while it draws
/// and follows a resize. It stops counting a renderer once it is retired, a
/// few seconds after its picture stops, or once the host is closed.
#[test]
fn a_renderer_is_counted_while_it_is_open_and_not_once_it_is_retired() {
    if !renderer_is_available() {
        eprintln!("skipped: no GPU and no software rasteriser");
        return;
    }
    let mut host = HydraHost::new();
    assert!(host.memory().is_empty(), "nothing before a sketch");

    host.set_drawing(true);
    host.apply(sketch(256, 144)).expect("install a sketch");
    lit_frame(&host, 256);
    let small = settles(&host, Duration::from_secs(5), |memory| {
        memory.score.is_some()
    })
    .score
    .expect("the score's renderer");
    // Rendered above the delivery and averaged down, in proportion.
    assert_eq!(small.width * 144, small.height * 256, "{small:?}");
    assert!(
        small.textures >= 10 * small.width as usize * small.height as usize * 4,
        "ten output textures at the render size: {small:?}"
    );
    assert_eq!(host.memory().theme, None, "no theme was installed");

    host.apply(sketch(640, 360)).expect("grow");
    lit_frame(&host, 640);
    let grown = settles(&host, Duration::from_secs(5), |memory| {
        memory.score.is_some_and(|score| score.width != small.width)
    })
    .score
    .expect("the score's renderer");
    assert_eq!(grown.width * 360, grown.height * 640, "{grown:?}");

    // Stopped, the theme's picture takes over, and the score's renderer is
    // kept only for its linger.
    host.set_drawing(false);
    host.theme(Some("osc(10, 0.1, 1.2).out()"))
        .expect("install a theme sketch");
    let theme_only = settles(&host, Duration::from_secs(20), |memory| {
        memory.score.is_none() && memory.theme.is_some()
    });
    assert_eq!(
        theme_only.renderers().count(),
        1,
        "only the theme's renderer is left: {theme_only:?}"
    );

    host.close();
    assert!(host.memory().is_empty(), "{:?}", host.memory());
}

/// A raw program for `code` at the given size.
fn from_source(code: &str, width: u32, height: u32) -> HydraProgram {
    HydraProgram {
        options: HydraOptions {
            width,
            height,
            ..HydraOptions::default()
        },
        ..HydraProgram::from_source(code)
    }
}

/// A recorded program with one statement per chain.
fn recording(width: u32, height: u32, chains: &[&str]) -> HydraProgram {
    HydraProgram {
        statements: chains
            .iter()
            .map(|chain| HydraStatement::Evaluate {
                node: parse_chain(chain).unwrap_or_else(|error| panic!("{chain}: {error}")),
            })
            .collect(),
        ..sketch(width, height)
    }
}

/// Collects failures until the shelf has delivered `passes` frames since the
/// first failure arrived. The shelf draws once per worker pass and a take sees
/// only the latest frame, so `passes` is a lower bound on the worker passes
/// waited through. Panics if the frames do not arrive.
fn failures_across(host: &mut HydraHost, passes: usize) -> Vec<String> {
    host.set_preview_frame_size(32, 32);
    host.preview(Some("osc(20, 0.1, 1.2).out()"))
        .expect("count passes on the shelf");
    let shelf = host.preview_frames();
    let mut said = Vec::new();
    let mut counted: Option<usize> = None;
    let until = Instant::now() + Duration::from_secs(25);
    loop {
        said.extend(
            host.take_events()
                .into_iter()
                .filter_map(|event| match event {
                    HydraEvent::Failed { message } => Some(message),
                    _ => None,
                }),
        );
        if counted.is_some_and(|counted| counted >= passes) {
            break;
        }
        if !said.is_empty() {
            *counted.get_or_insert(0) += usize::from(shelf.take().is_some());
        }
        assert!(
            Instant::now() < until,
            "the host said {said:?} and the shelf counted {counted:?} passes after it"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    host.preview(None).expect("clear the shelf");
    said
}

/// A persistent parse, compose or shader failure is reported once, although
/// every frame is still read back.
#[test]
fn a_persistent_failure_is_said_once_not_every_frame() {
    if !renderer_is_available() {
        eprintln!("skipped: no GPU and no software rasteriser");
        return;
    }
    let mut host = HydraHost::new();
    host.set_drawing(true);
    host.set_frame_size(64, 64);
    for (code, why) in [
        ("osc(", "sketch:"),
        (
            "thisIsNotAHydraGenerator().out()",
            "thisIsNotAHydraGenerator",
        ),
        ("osc(10, 0.1, 0.3).sum().out()", "shader"),
    ] {
        host.apply(from_source(code, 64, 64))
            .expect("apply a sketch that fails");
        let said = failures_across(&mut host, 12);
        assert!(
            said.len() == 1 && said[0].contains(why),
            "`{code}` must be said exactly once, for its own failure: {said:?}"
        );
    }
}

/// Two chains with different persistent failures are each reported once.
#[test]
fn chains_that_fail_differently_are_each_said_once() {
    if !renderer_is_available() {
        eprintln!("skipped: no GPU and no software rasteriser");
        return;
    }
    let mut host = HydraHost::new();
    host.set_drawing(true);
    host.set_frame_size(64, 64);
    host.apply(recording(
        64,
        64,
        &["osc().foo().out(o0)", "osc().bar().out(o1)"],
    ))
    .expect("apply a score whose two chains fail");
    let said = failures_across(&mut host, 12);
    assert!(
        said.len() == 2
            && said.iter().any(|message| message.contains("foo"))
            && said.iter().any(|message| message.contains("bar")),
        "each chain's failure must be said exactly once: {said:?}"
    );
}

/// A clean frame re-arms the latch within one program: each spell of an
/// oversized delivery is reported once.
#[test]
fn a_clean_frame_rearms_the_said_once_latch() {
    if !renderer_is_available() {
        eprintln!("skipped: no GPU and no software rasteriser");
        return;
    }
    let mut host = HydraHost::new();
    host.set_drawing(true);
    host.apply(sketch(64, 64)).expect("install a sketch");
    for spell in ["the first", "a later"] {
        // Discard any frame held from the previous spell.
        host.frames().take();
        host.set_frame_size(64, 64);
        assert_eq!(lit_frame(&host, 64), (64, 64));
        host.set_frame_size(8192, 64);
        let said = failures_across(&mut host, 12);
        assert_eq!(
            said.len(),
            1,
            "{spell} spell of too wide must be said exactly once: {said:?}"
        );
    }
}

/// A new program re-arms the latch: broken, empty, then broken again reports
/// the failure twice.
#[test]
fn a_new_program_rearms_the_said_once_latch() {
    if !renderer_is_available() {
        eprintln!("skipped: no GPU and no software rasteriser");
        return;
    }
    let mut host = HydraHost::new();
    host.set_drawing(true);
    host.set_frame_size(64, 64);
    for run in ["the first", "the restored"] {
        host.apply(from_source("thisIsNotAHydraGenerator().out()", 64, 64))
            .expect("apply a sketch that cannot compose");
        let said = failures_across(&mut host, 12);
        assert_eq!(
            said.len(),
            1,
            "{run} broken sketch must be said exactly once: {said:?}"
        );
        host.apply(HydraProgram::default())
            .expect("an empty sketch is not an error");
    }
}

/// A failing theme is reported once per install, through its own latch.
#[test]
fn a_failing_theme_is_said_once_not_every_frame() {
    if !renderer_is_available() {
        eprintln!("skipped: no GPU and no software rasteriser");
        return;
    }
    let mut host = HydraHost::new();
    for install in ["installed", "installed again"] {
        host.theme(Some("osc(10, 0.1, 0.3).sum().out()"))
            .expect("install a theme that will not compile");
        let said = failures_across(&mut host, 12);
        assert!(
            said.len() == 1 && said[0].contains("shader"),
            "a theme {install} must be said exactly once: {said:?}"
        );
        host.theme(None).expect("remove the theme");
    }
}

/// A raw program reads back the output its chain draws to.
#[test]
fn a_raw_sketch_reads_the_output_its_chain_drew() {
    if !renderer_is_available() {
        eprintln!("skipped: no GPU and no software rasteriser");
        return;
    }
    let mut host = HydraHost::new();
    host.set_drawing(true);
    host.apply(from_source("osc(20, 0.1, 1.2).out(o1)", 128, 128))
        .expect("apply a raw sketch bound for o1");
    host.set_frame_size(128, 128);
    assert_eq!(lit_frame(&host, 128), (128, 128));
}
