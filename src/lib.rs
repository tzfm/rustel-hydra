//! Hydra visuals, drawn behind the code in the terminal.
//!
//! Hydra's vocabulary is a table of GLSL functions and a composer that splices
//! them into one fragment shader. [`glsl`] is that table and that composer, in
//! Rust; [`native`] runs the result on the GPU, offscreen, and hands back
//! pixels the terminal draws.
//!
//! The default `rustel` command and `rustel-runtime` builds include the
//! `hydra` feature. A minimal host can omit the GPU stack by disabling default
//! features and selecting only the capabilities it needs.
//!
//! # Fidelity is checked, not claimed
//!
//! `hydra-synth` can emit the GLSL for any chain without a GPU. So
//! `tests/glsl_parity.rs` compares our shader against its, **character for
//! character**, across its whole table plus multi-step chains, nested sources
//! and signals; and `tests/pixel_parity.rs` compares the rendered frames. The
//! function bodies in [`glsl::table`] are generated from the library's own
//! source rather than retyped, so a version bump regenerates them.
//!
//! # Shape
//!
//! * A score's Hydra calls are *recorded*, not executed. [`HydraProgram`] is
//!   that recording, and it crosses a thread boundary as data.
//! * [`HydraHost`] owns one worker thread, spawned the first time a score asks
//!   to draw. It exists because a frame costs a few milliseconds of GPU
//!   round-trip and the Session must not wait on it.
//! * Two pictures are drawn: the score's, behind the code, and the snippet
//!   shelf's thumbnail. Separate renderers, so browsing does not take the
//!   screen from a playing set.
//! * `H(pattern)` compiles to a uniform, which is what a `() => …` argument
//!   compiles to, so the shader is the one Hydra would write.
//!
//! No part of this crate runs on the audio callback, the scheduler, or the
//! QuickJS realm.

pub mod glsl;
mod host;
mod names;
pub mod native;
mod program;

pub use host::{
    HydraEvent, HydraFrames, HydraHost, HydraHostError, HydraInputLease, HydraInputSink,
    HydraMemory, HydraTuiSink,
};
pub use names::hydra_names;
pub use program::{
    HYDRA_DEFAULT_AUDIO_BINS, HYDRA_MAX_AUDIO_BINS, HYDRA_MAX_OUTPUT_EDGE, HYDRA_MAX_OUTPUT_PIXELS,
    HYDRA_PROGRAM_VERSION, HYDRA_SOURCE_SLOTS, HydraCall, HydraNode, HydraOptions, HydraProgram,
    HydraProgramError, HydraSource, HydraStatement, MAX_HYDRA_ARGS, MAX_HYDRA_CALLS,
    MAX_HYDRA_DEPTH, MAX_HYDRA_NAME_BYTES, MAX_HYDRA_NODES, MAX_HYDRA_SIGNALS,
    MAX_HYDRA_SOURCE_BYTES, MAX_HYDRA_STATEMENTS, MAX_HYDRA_TEXT_BYTES,
};

use serde::{Deserialize, Serialize};

/// Frames a second Hydra renders.
///
/// A terminal redraws far more slowly than a display, and every frame costs a
/// readback and a picture's worth of bytes across a channel. Thirty is smooth
/// to the eye and leaves the machine alone.
pub const HYDRA_FRAMES_PER_SECOND: u32 = 30;

/// Hydra's fallback render size.
///
/// A host-provided delivery size takes precedence. Studio always derives that
/// size from its current layout; headless and custom hosts can use this
/// fallback.
pub const HYDRA_WIDTH: u32 = 640;
pub const HYDRA_HEIGHT: u32 = 360;

/// Default compatibility value for the legacy strength option.
///
/// The renderer records and validates this value but leaves final compositing
/// to its host. Studio uses its persisted visuals opacity instead.
pub const HYDRA_STRENGTH: f32 = 0.45;

/// Frames of terminal grid the window will accept per second.
///
/// The terminal redraws faster than this when a set is busy; the texture Hydra
/// samples does not need every one of those, and each frame is a full grid.
pub const HYDRA_TUI_FRAMES_PER_SECOND: u32 = 30;

/// Fixed GPU/protocol capacity for audio bands.
///
/// Hydra starts with [`HYDRA_DEFAULT_AUDIO_BINS`] active bands and
/// `a.setBins(n)` changes their count and spectrum spacing. The uniform stays
/// at this maximum size so programs do not need a new pipeline layout.
pub const HYDRA_AUDIO_BINS: usize = HYDRA_MAX_AUDIO_BINS;

/// How far apart the samples in a [`HydraSignalFrame`] stand, in milliseconds.
///
/// One display frame at 60 Hz. The window indexes the schedule by elapsed time
/// rather than asking the engine anything, which is what keeps `H(pattern)`
/// free at frame rate.
pub const HYDRA_SIGNAL_STEP_MS: f64 = 1000.0 / 60.0;

/// Samples in one [`HydraSignalFrame`]: a third of a second at 60 Hz.
///
/// Sent about four times a second, so a frame always overlaps its successor
/// and a late refresh shows the last known value rather than a hole.
pub const HYDRA_SIGNAL_SAMPLES: usize = 20;

/// Values a score streams into the window for its `H(pattern)` thunks.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct HydraSignalFrame {
    /// Milliseconds between consecutive samples.
    pub step: f64,
    /// One list of samples per declared signal slot, in slot order.
    pub slots: Vec<Vec<serde_json::Value>>,
}

/// What the engine can say about the sound it is making, in the shape
/// hydra-synth's own microphone analyser would have said it.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct HydraAudioFrame {
    /// Fixed-capacity backing for `a.fft`. Only the configured prefix is
    /// populated; Hydra's default prefix has four bins.
    pub bins: [f32; HYDRA_AUDIO_BINS],
    pub rms: f32,
    /// The wider spectrum the terminal's own scopes are drawn from.
    pub spectrum: Vec<f32>,
}

/// One terminal cell, on its way to becoming a texture.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HydraTuiCell {
    /// The first scalar of the cell's symbol. Zero means blank.
    pub code: u32,
    pub fg: [u8; 3],
    pub bg: [u8; 3],
    /// Bit 0 bold, 1 italic, 2 underlined, 3 reversed, 4 dim.
    pub flags: u8,
}

/// One frame of the terminal, on its way to becoming a texture.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HydraTuiFrame {
    pub cols: u16,
    pub rows: u16,
    /// Cell size in pixels when the terminal knows it; zero when it does not,
    /// and the window picks something legible.
    pub cell_width: u16,
    pub cell_height: u16,
    /// Row-major, `cols * rows` long.
    pub cells: Vec<HydraTuiCell>,
}

#[cfg(test)]
mod tests {
    use super::*;

    mod numeric_arguments {
        use super::*;

        fn with_arg(arg: HydraNode) -> HydraProgram {
            sketch(
                vec![HydraStatement::Evaluate {
                    node: HydraNode::Chain {
                        head: "osc".into(),
                        args: vec![arg],
                        calls: vec![],
                    },
                }],
                0,
            )
        }

        #[test]
        fn numeric_nodes_and_list_marks_must_be_finite() {
            for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                assert_eq!(
                    with_arg(HydraNode::Number { v: value }).validate(),
                    Err(HydraProgramError::NonFiniteNumber("numeric argument"))
                );
                for (field, expected) in [
                    (0, "list speed"),
                    (1, "list smoothness"),
                    (2, "list offset"),
                ] {
                    let mut marks = [1.0, 0.0, 0.0];
                    marks[field] = value;
                    assert_eq!(
                        with_arg(HydraNode::List {
                            v: vec![HydraNode::Number { v: 1.0 }],
                            speed: marks[0],
                            smooth: marks[1],
                            ease: "linear".into(),
                            offset: marks[2],
                        })
                        .validate(),
                        Err(HydraProgramError::NonFiniteNumber(expected))
                    );
                }
            }
        }

        #[test]
        fn finite_numeric_nodes_and_marks_survive_the_wire() {
            let program = with_arg(HydraNode::List {
                v: vec![HydraNode::Number { v: -0.0 }, HydraNode::Number { v: 1e-5 }],
                speed: -2.0,
                smooth: 0.5,
                ease: "linear".into(),
                offset: -0.25,
            });
            program.validate().expect("finite values validate");
            let wire = serde_json::to_value(&program).expect("serialize");
            let restored: HydraProgram = serde_json::from_value(wire).expect("deserialize");
            assert_eq!(restored, program);
        }
    }

    fn sketch(statements: Vec<HydraStatement>, signals: u32) -> HydraProgram {
        HydraProgram {
            version: HYDRA_PROGRAM_VERSION,
            options: HydraOptions::default(),
            statements,
            signals,
            raw: None,
        }
    }

    #[test]
    fn typed_sources_survive_the_wire_and_make_a_source_only_program_nonempty() {
        let program = sketch(
            vec![
                HydraStatement::ConfigureSource {
                    slot: 0,
                    source: HydraSource::Camera { device: None },
                },
                HydraStatement::ConfigureSource {
                    slot: 3,
                    source: HydraSource::Camera { device: Some(2) },
                },
                HydraStatement::ConfigureSource {
                    slot: 1,
                    source: HydraSource::ImageUrl {
                        url: "https://i.imgur.com/zFttbWq.jpg".into(),
                    },
                },
                HydraStatement::ClearSource { slot: 1 },
            ],
            0,
        );
        assert!(!program.is_empty(), "configuring a source is an effect");
        program.validate().expect("the source slots are valid");
        let json = serde_json::to_string(&program).expect("serialize");
        let back: HydraProgram = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(program, back);
    }

    #[test]
    fn typed_audio_bin_configuration_survives_the_wire() {
        let program = sketch(vec![HydraStatement::ConfigureAudio { bins: 8 }], 0);
        assert!(!program.is_empty(), "analyser configuration is an effect");
        program
            .validate()
            .expect("eight bins fit the native uniform");
        let json = serde_json::to_string(&program).expect("serialize");
        assert!(json.contains(r#""s":"audio","bins":8"#), "{json}");
        let back: HydraProgram = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(program, back);
    }

    #[test]
    fn audio_bin_configuration_is_strictly_bounded() {
        for (bins, found) in [(0, 0), (17, 17), (u8::MAX, usize::from(u8::MAX))] {
            let program = sketch(vec![HydraStatement::ConfigureAudio { bins }], 0);
            assert_eq!(
                program.validate(),
                Err(HydraProgramError::OutOfRange {
                    what: "audio bins",
                    min: 1,
                    max: HYDRA_MAX_AUDIO_BINS,
                    found,
                })
            );
        }
        for bins in [
            1,
            HYDRA_DEFAULT_AUDIO_BINS as u8,
            HYDRA_MAX_AUDIO_BINS as u8,
        ] {
            sketch(vec![HydraStatement::ConfigureAudio { bins }], 0)
                .validate()
                .unwrap_or_else(|error| panic!("{bins} bins: {error}"));
        }
    }

    #[test]
    fn unimplemented_timing_assignments_are_refused_at_the_wire() {
        let program = sketch(
            vec![HydraStatement::Assign {
                name: "speed".into(),
                value: HydraNode::Number { v: 2.0 },
            }],
            0,
        );
        assert_eq!(
            program.validate(),
            Err(HydraProgramError::UnsupportedSetting("speed".into()))
        );
    }

    #[test]
    fn default_and_null_options_are_valid_bounded_fallbacks() {
        for json in [
            serde_json::json!({
                "version": HYDRA_PROGRAM_VERSION,
                "statements": [],
                "signals": 0,
            }),
            serde_json::json!({
                "version": HYDRA_PROGRAM_VERSION,
                "options": null,
                "statements": [],
                "signals": 0,
            }),
        ] {
            let program: HydraProgram = serde_json::from_value(json).expect("fallback options");
            assert_eq!(program.options, HydraOptions::default());
            program.validate().expect("default geometry is bounded");
        }
    }

    #[test]
    fn output_geometry_has_edge_and_aggregate_allocation_bounds() {
        let with_size = |width, height| HydraProgram {
            options: HydraOptions {
                width,
                height,
                ..HydraOptions::default()
            },
            ..HydraProgram::default()
        };
        for (width, height) in [(0, 1), (1, 0), (4097, 1), (1, 4097), (4096, 1025)] {
            assert!(matches!(
                with_size(width, height).validate(),
                Err(HydraProgramError::OutputDimensions { .. })
            ));
        }
        with_size(4096, 1024)
            .validate()
            .expect("the exact area boundary is accepted");
    }

    #[test]
    fn scalar_output_options_must_be_finite_and_bounded() {
        let mut program = HydraProgram::default();
        program.options.strength = f32::NAN;
        assert!(matches!(
            program.validate(),
            Err(HydraProgramError::InvalidOption(_))
        ));
        program.options = HydraOptions::default();
        program.options.pixel_ratio = f64::INFINITY;
        assert!(matches!(
            program.validate(),
            Err(HydraProgramError::InvalidOption(_))
        ));
        program.options = HydraOptions::default();
        program.options.fps = 0;
        assert!(matches!(
            program.validate(),
            Err(HydraProgramError::InvalidOption(_))
        ));
    }

    #[test]
    fn source_slots_and_url_bytes_are_bounded_at_the_wire() {
        let invalid_slot = sketch(
            vec![HydraStatement::ConfigureSource {
                slot: HYDRA_SOURCE_SLOTS as u8,
                source: HydraSource::Camera { device: None },
            }],
            0,
        );
        assert!(matches!(
            invalid_slot.validate(),
            Err(HydraProgramError::TooMany {
                what: "source slot index",
                ..
            })
        ));

        let invalid_clear = sketch(
            vec![HydraStatement::ClearSource {
                slot: HYDRA_SOURCE_SLOTS as u8,
            }],
            0,
        );
        assert!(matches!(
            invalid_clear.validate(),
            Err(HydraProgramError::TooMany {
                what: "source slot index",
                ..
            })
        ));

        let image = |url: String| {
            sketch(
                vec![HydraStatement::ConfigureSource {
                    slot: 0,
                    source: HydraSource::ImageUrl { url },
                }],
                0,
            )
        };
        assert_eq!(
            image(String::new()).validate(),
            Err(HydraProgramError::Empty("source URL"))
        );
        assert!(matches!(
            image("x".repeat(MAX_HYDRA_TEXT_BYTES + 1)).validate(),
            Err(HydraProgramError::TooLong {
                what: "source URL",
                ..
            })
        ));
    }

    #[test]
    fn a_name_that_is_not_an_identifier_is_refused() {
        let program = sketch(
            vec![HydraStatement::Evaluate {
                node: HydraNode::Chain {
                    head: "osc(1);globalThis.x=1;(".into(),
                    args: Vec::new(),
                    calls: Vec::new(),
                },
            }],
            0,
        );
        assert!(matches!(
            program.validate(),
            Err(HydraProgramError::Name(_))
        ));
    }

    #[test]
    fn a_signal_slot_the_score_never_declared_is_refused() {
        let program = sketch(
            vec![HydraStatement::Evaluate {
                node: HydraNode::Signal { slot: 3 },
            }],
            0,
        );
        assert_eq!(program.validate(), Err(HydraProgramError::Signal(3)));
    }
}
