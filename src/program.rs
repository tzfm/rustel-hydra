//! The wire form of one score's Hydra sketch.
//!
//! A score does not hand the renderer JavaScript to run. The score realm
//! records what it was asked to draw - which generator, which arguments, which
//! methods, in which order - and that recording crosses the thread boundary as
//! data. The renderer composes a shader from it and runs no JavaScript.
//!
//! Every list here has a ceiling. The recording is written by score text, so
//! it is exactly as trustworthy as the person typing, and exactly as bounded
//! as the editor event protocol next door in `rustel-runtime`.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Current schema version of a recorded sketch.
pub const HYDRA_PROGRAM_VERSION: u16 = 3;

/// Drawing statements one sketch may carry.
pub const MAX_HYDRA_STATEMENTS: usize = 64;

/// Nodes in one statement, counted across the whole tree.
pub const MAX_HYDRA_NODES: usize = 2_048;

/// How deeply one statement may nest chains inside arguments.
pub const MAX_HYDRA_DEPTH: usize = 24;

/// Arguments to one generator or method call.
pub const MAX_HYDRA_ARGS: usize = 16;

/// `List`'s wire defaults: an unmarked array steps at speed 1 with the
/// linear easing, as hydra-synth's `getValue` reads it.
const fn list_speed() -> f64 {
    1.0
}
fn list_ease() -> String {
    "linear".to_owned()
}

/// Method calls in one chain.
pub const MAX_HYDRA_CALLS: usize = 128;

/// A generator, method, or global name.
pub const MAX_HYDRA_NAME_BYTES: usize = 64;

/// Source text retained for one `() => …` argument.
pub const MAX_HYDRA_SOURCE_BYTES: usize = 4 * 1024;

/// Text retained for one string argument.
pub const MAX_HYDRA_TEXT_BYTES: usize = 4 * 1024;

/// Distinct `H(pattern)` signals one score may stream.
pub const MAX_HYDRA_SIGNALS: usize = 64;

/// External texture inputs exposed as `s0` through `s3`.
///
/// This is part of the recording protocol, not a renderer implementation
/// detail: a recording with any other slot must be refused before it reaches
/// hardware or a source service.
pub const HYDRA_SOURCE_SLOTS: usize = 4;

/// The analyser shape hydra-synth starts with before `a.setBins(...)`.
pub const HYDRA_DEFAULT_AUDIO_BINS: usize = 4;

/// Largest analyser shape the native shader protocol can address.
///
/// The GPU uniform has this fixed capacity even when a sketch selects fewer
/// bands, so changing it is a protocol change rather than a renderer detail.
pub const HYDRA_MAX_AUDIO_BINS: usize = 16;

/// Largest width or height accepted for one native output frame.
pub const HYDRA_MAX_OUTPUT_EDGE: u32 = 4_096;

/// Largest area accepted for one native output frame.
///
/// The renderer owns ten RGBA render textures (two halves for four outputs
/// and two halves for the display compositor), so this bounds their aggregate
/// storage to forty mebi-pixels rather than permitting a 4096² allocation.
pub const HYDRA_MAX_OUTPUT_PIXELS: u64 = 4 * 1024 * 1024;

/// What went wrong with a recording, said plainly enough to print.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HydraProgramError {
    /// The recording was written by a different protocol version.
    Version(u16),
    /// A list exceeded its documented ceiling.
    TooMany {
        what: &'static str,
        limit: usize,
        found: usize,
    },
    /// A string exceeded its documented ceiling.
    TooLong {
        what: &'static str,
        limit: usize,
        found: usize,
    },
    /// A name was not something that can be looked up on a JavaScript object.
    Name(String),
    /// Statements nested past the depth ceiling.
    TooDeep(usize),
    /// A `H(...)` slot outside the range the score declared.
    Signal(u32),
    /// A required string carried no value.
    Empty(&'static str),
    /// A scalar setting fell outside the protocol's supported range.
    OutOfRange {
        what: &'static str,
        min: usize,
        max: usize,
        found: usize,
    },
    /// A recorded setting has no honest native implementation.
    UnsupportedSetting(String),
    /// Output dimensions would exceed the native renderer's bounded texture
    /// and readback allocation.
    OutputDimensions {
        width: u32,
        height: u32,
        max_edge: u32,
        max_pixels: u64,
    },
    /// A scalar option is non-finite or outside its documented range.
    InvalidOption(&'static str),
    /// A recorded numeric argument cannot be represented on the JSON wire.
    NonFiniteNumber(&'static str),
}

impl fmt::Display for HydraProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Version(found) => write!(
                f,
                "hydra sketch is protocol version {found}; this build speaks {HYDRA_PROGRAM_VERSION}"
            ),
            Self::TooMany { what, limit, found } => {
                write!(f, "hydra sketch has {found} {what}; the maximum is {limit}")
            }
            Self::TooLong { what, limit, found } => write!(
                f,
                "hydra sketch has a {what} of {found} bytes; the maximum is {limit}"
            ),
            Self::Name(name) => write!(
                f,
                "hydra sketch names {name:?}, which is not a JavaScript identifier"
            ),
            Self::TooDeep(limit) => write!(f, "hydra sketch nests deeper than {limit} levels"),
            Self::Signal(slot) => write!(
                f,
                "hydra sketch reads signal slot {slot}, which it never declared"
            ),
            Self::Empty(what) => write!(f, "hydra sketch has an empty {what}"),
            Self::OutOfRange {
                what,
                min,
                max,
                found,
            } => write!(
                f,
                "hydra sketch sets {what} to {found}; supported values are {min}..={max}"
            ),
            Self::UnsupportedSetting(name) => write!(
                f,
                "hydra sketch assigns {name}, which is not supported by native Hydra"
            ),
            Self::OutputDimensions {
                width,
                height,
                max_edge,
                max_pixels,
            } => write!(
                f,
                "hydra output dimensions {width}x{height} exceed the {max_edge}-pixel edge / {max_pixels}-pixel area limit"
            ),
            Self::InvalidOption(why) => write!(f, "hydra sketch has an invalid option: {why}"),
            Self::NonFiniteNumber(what) => {
                write!(f, "hydra sketch has a non-finite {what}")
            }
        }
    }
}

impl std::error::Error for HydraProgramError {}

/// Everything `initHydra({...})` can say.
///
/// `detectAudio` and `feedStrudel` control native inputs. `pixelRatio`,
/// `pixelated`, and `contextType` remain on the wire for source compatibility
/// but currently do not alter native rendering. `width` and `height` are the
/// fallback frame size when a host does not provide a delivery size.
/// `strength` is retained and validated for protocol compatibility but is not
/// applied by the renderer.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct HydraOptions {
    /// Fallback render size for a host that does not provide delivery
    /// dimensions. Studio normally overrides both values from its layout.
    pub width: u32,
    pub height: u32,
    /// Compatibility value in the range 0..=1. Native rendering currently
    /// leaves compositing strength to the host.
    pub strength: f32,
    pub detect_audio: bool,
    pub feed_strudel: bool,
    pub pixel_ratio: f64,
    pub pixelated: bool,
    pub context_type: String,
    /// How many frames a second cross the wire.
    ///
    /// Host-controlled, not a score option. Its default comes from
    /// [`crate::HYDRA_FRAMES_PER_SECOND`] so the host and renderer agree.
    #[serde(default = "default_fps")]
    pub fps: u32,
}

fn default_fps() -> u32 {
    crate::HYDRA_FRAMES_PER_SECOND
}

impl Default for HydraOptions {
    fn default() -> Self {
        Self {
            width: crate::HYDRA_WIDTH,
            height: crate::HYDRA_HEIGHT,
            strength: crate::HYDRA_STRENGTH,
            // `a.fft` reads the engine's output, so audio-reactive sketches
            // can start immediately without opening a microphone.
            detect_audio: true,
            feed_strudel: false,
            pixel_ratio: 1.0,
            pixelated: true,
            context_type: "webgl".to_owned(),
            fps: crate::HYDRA_FRAMES_PER_SECOND,
        }
    }
}

/// One value in a recorded call.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "c")]
pub enum HydraNode {
    #[serde(rename = "num")]
    Number { v: f64 },
    #[serde(rename = "str")]
    Text { v: String },
    #[serde(rename = "bool")]
    Bool { v: bool },
    #[serde(rename = "null")]
    Null,
    /// An array, which animates where a float goes: its index is
    /// `time * speed * bpm / 60 + offset`, with hydra's default 30 bpm.
    #[serde(rename = "list")]
    List {
        v: Vec<HydraNode>,
        /// `.fast(speed)`.
        #[serde(default = "list_speed")]
        speed: f64,
        /// `.smooth(k)`: 0 steps between entries; otherwise the value eases
        /// across a window `k` wide around each boundary.
        #[serde(default)]
        smooth: f64,
        /// `.ease(name)`: the curve across the smooth window, by hydra's
        /// easing name. The composer refuses a name it does not compile.
        #[serde(default = "list_ease")]
        ease: String,
        /// `.offset(o)`: added to the index.
        #[serde(default)]
        offset: f64,
    },
    /// A function written in the score, kept as its own source text. The
    /// composer compiles it into the shader, where it can read `time` and
    /// `a.fft[n]`. It cannot close over score variables.
    #[serde(rename = "fn")]
    Source { src: String },
    /// `H(pattern)`: the engine streams this pattern's value in.
    #[serde(rename = "sig")]
    Signal { slot: u32 },
    /// A bare Hydra global used as a value: `o0`, `s0`, `time`.
    #[serde(rename = "ref")]
    Global { name: String },
    /// `osc(10, 0.1).kaleid(4).out(o0)`
    #[serde(rename = "call")]
    Chain {
        head: String,
        #[serde(default)]
        args: Vec<HydraNode>,
        #[serde(default)]
        calls: Vec<HydraCall>,
    },
}

/// One `.method(...)` in a chain.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct HydraCall {
    pub method: String,
    #[serde(default)]
    pub args: Vec<HydraNode>,
}

/// One top-level thing the score asked to be drawn.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "s")]
pub enum HydraStatement {
    /// Evaluate a chain for its effect, which for Hydra means `.out()`.
    #[serde(rename = "eval")]
    Evaluate { node: HydraNode },
    /// Reserved wire shape for Hydra's settable globals. Validation currently
    /// refuses it because silently discarding clock semantics is worse than a
    /// clear unsupported error.
    #[serde(rename = "set")]
    Assign { name: String, value: HydraNode },
    /// Configure one of Hydra's four external texture inputs. Keeping this
    /// typed is important: a URL or camera request is an effect, not a GLSL
    /// chain, and the host must be able to permission-check it before doing
    /// any I/O.
    #[serde(rename = "source")]
    ConfigureSource { slot: u8, source: HydraSource },
    /// `sN.clear()`: release external acquisition and return the texture to
    /// transparent black.
    #[serde(rename = "clear_source")]
    ClearSource { slot: u8 },
    /// `a.setBins(n)`: choose how many contiguous regions of the analyser's
    /// spectrum become `a.fft[0]` through `a.fft[n - 1]`.
    #[serde(rename = "audio")]
    ConfigureAudio { bins: u8 },
}

/// The external inputs native Hydra can acquire without a browser DOM.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HydraSource {
    /// `sN.initCam()`; no device means the platform default camera.
    Camera { device: Option<u32> },
    /// `sN.initImage(url)`. The source loader accepts public HTTPS URLs;
    /// insecure, local and private-network targets are rejected before I/O.
    ImageUrl { url: String },
}

/// What one evaluated score wants drawn.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct HydraProgram {
    pub version: u16,
    /// Absent, or explicitly null, when the score asked for nothing to be
    /// drawn - which is the instruction to stop, and has to cross the wire as
    /// readily as a sketch does.
    #[serde(default, deserialize_with = "options_or_default")]
    pub options: HydraOptions,
    pub statements: Vec<HydraStatement>,
    /// How many `H(pattern)` slots the score declared. Slots are dense and
    /// numbered from zero in the order the score built them.
    #[serde(default)]
    pub signals: u32,
    /// Hydra source to draw instead of `statements`.
    ///
    /// A score never sets this: its chain is recorded rather than evaluated,
    /// which is what keeps the realms apart. The snippet shelf does, because a
    /// snippet is source - the text the reader is about to copy - and
    /// previewing anything else would preview the wrong thing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
}

impl Default for HydraProgram {
    fn default() -> Self {
        Self {
            version: HYDRA_PROGRAM_VERSION,
            options: HydraOptions::default(),
            statements: Vec::new(),
            signals: 0,
            raw: None,
        }
    }
}

/// `null` and a missing field both mean "no options", which is what a score
/// with nothing to draw records. Serde's own `default` covers only the second.
fn options_or_default<'de, D>(deserializer: D) -> Result<HydraOptions, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<HydraOptions>::deserialize(deserializer)?.unwrap_or_default())
}

impl HydraProgram {
    /// True when the score asked for nothing - the signal to stop drawing.
    pub fn is_empty(&self) -> bool {
        self.statements.is_empty() && self.raw.is_none()
    }

    /// A sketch from the shelf, run as the source it is.
    pub fn from_source(code: &str) -> Self {
        Self {
            raw: Some(code.to_owned()),
            ..Self::default()
        }
    }

    /// Accept a recording only if it is within every documented ceiling.
    ///
    /// The recording arrives from the score realm in the same process, but it
    /// is written by score text and reaches a thread that must not be able to
    /// be talked into an unbounded allocation or an unbounded string eval.
    pub fn validate(&self) -> Result<(), HydraProgramError> {
        if self.version != HYDRA_PROGRAM_VERSION {
            return Err(HydraProgramError::Version(self.version));
        }
        if self.signals as usize > MAX_HYDRA_SIGNALS {
            return Err(HydraProgramError::TooMany {
                what: "signals",
                limit: MAX_HYDRA_SIGNALS,
                found: self.signals as usize,
            });
        }
        let pixels = u64::from(self.options.width)
            .checked_mul(u64::from(self.options.height))
            .ok_or(HydraProgramError::OutputDimensions {
                width: self.options.width,
                height: self.options.height,
                max_edge: HYDRA_MAX_OUTPUT_EDGE,
                max_pixels: HYDRA_MAX_OUTPUT_PIXELS,
            })?;
        if self.options.width == 0
            || self.options.height == 0
            || self.options.width > HYDRA_MAX_OUTPUT_EDGE
            || self.options.height > HYDRA_MAX_OUTPUT_EDGE
            || pixels > HYDRA_MAX_OUTPUT_PIXELS
        {
            return Err(HydraProgramError::OutputDimensions {
                width: self.options.width,
                height: self.options.height,
                max_edge: HYDRA_MAX_OUTPUT_EDGE,
                max_pixels: HYDRA_MAX_OUTPUT_PIXELS,
            });
        }
        if !self.options.strength.is_finite() || !(0.0..=1.0).contains(&self.options.strength) {
            return Err(HydraProgramError::InvalidOption(
                "strength must be finite and between 0 and 1",
            ));
        }
        if !self.options.pixel_ratio.is_finite()
            || !(0.05..=4.0).contains(&self.options.pixel_ratio)
        {
            return Err(HydraProgramError::InvalidOption(
                "pixelRatio must be finite and between 0.05 and 4",
            ));
        }
        if self.options.fps == 0 || self.options.fps > 240 {
            return Err(HydraProgramError::InvalidOption(
                "fps must be between 1 and 240",
            ));
        }
        check_identifier(&self.options.context_type)?;
        if let Some(raw) = &self.raw {
            check_len("sketch source", raw, MAX_HYDRA_SOURCE_BYTES)?;
        }
        if self.statements.len() > MAX_HYDRA_STATEMENTS {
            return Err(HydraProgramError::TooMany {
                what: "statements",
                limit: MAX_HYDRA_STATEMENTS,
                found: self.statements.len(),
            });
        }
        let mut nodes = 0_usize;
        for statement in &self.statements {
            match statement {
                HydraStatement::Evaluate { node } => self.check_node(node, 0, &mut nodes)?,
                HydraStatement::Assign { name, value } => {
                    check_identifier(name)?;
                    self.check_node(value, 0, &mut nodes)?;
                    return Err(HydraProgramError::UnsupportedSetting(name.clone()));
                }
                HydraStatement::ConfigureSource { slot, source } => {
                    if usize::from(*slot) >= HYDRA_SOURCE_SLOTS {
                        return Err(HydraProgramError::TooMany {
                            what: "source slot index",
                            limit: HYDRA_SOURCE_SLOTS - 1,
                            found: usize::from(*slot),
                        });
                    }
                    if let HydraSource::ImageUrl { url } = source {
                        if url.is_empty() {
                            return Err(HydraProgramError::Empty("source URL"));
                        }
                        check_len("source URL", url, MAX_HYDRA_TEXT_BYTES)?;
                    }
                }
                HydraStatement::ClearSource { slot } => {
                    if usize::from(*slot) >= HYDRA_SOURCE_SLOTS {
                        return Err(HydraProgramError::TooMany {
                            what: "source slot index",
                            limit: HYDRA_SOURCE_SLOTS - 1,
                            found: usize::from(*slot),
                        });
                    }
                }
                HydraStatement::ConfigureAudio { bins } => {
                    let found = usize::from(*bins);
                    if !(1..=HYDRA_MAX_AUDIO_BINS).contains(&found) {
                        return Err(HydraProgramError::OutOfRange {
                            what: "audio bins",
                            min: 1,
                            max: HYDRA_MAX_AUDIO_BINS,
                            found,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn check_node(
        &self,
        node: &HydraNode,
        depth: usize,
        nodes: &mut usize,
    ) -> Result<(), HydraProgramError> {
        if depth > MAX_HYDRA_DEPTH {
            return Err(HydraProgramError::TooDeep(MAX_HYDRA_DEPTH));
        }
        *nodes += 1;
        if *nodes > MAX_HYDRA_NODES {
            return Err(HydraProgramError::TooMany {
                what: "nodes",
                limit: MAX_HYDRA_NODES,
                found: *nodes,
            });
        }
        match node {
            HydraNode::Number { v } => {
                if !v.is_finite() {
                    return Err(HydraProgramError::NonFiniteNumber("numeric argument"));
                }
                Ok(())
            }
            HydraNode::Bool { .. } | HydraNode::Null => Ok(()),
            HydraNode::Text { v } => check_len("string", v, MAX_HYDRA_TEXT_BYTES),
            HydraNode::Source { src } => check_len("function source", src, MAX_HYDRA_SOURCE_BYTES),
            HydraNode::Signal { slot } => {
                if *slot >= self.signals {
                    return Err(HydraProgramError::Signal(*slot));
                }
                Ok(())
            }
            HydraNode::Global { name } => check_identifier(name),
            HydraNode::List {
                v,
                speed,
                smooth,
                ease,
                offset,
            } => {
                for (name, value) in [
                    ("list speed", speed),
                    ("list smoothness", smooth),
                    ("list offset", offset),
                ] {
                    if !value.is_finite() {
                        return Err(HydraProgramError::NonFiniteNumber(name));
                    }
                }
                check_len("easing name", ease, MAX_HYDRA_TEXT_BYTES)?;
                if v.len() > MAX_HYDRA_ARGS {
                    return Err(HydraProgramError::TooMany {
                        what: "list entries",
                        limit: MAX_HYDRA_ARGS,
                        found: v.len(),
                    });
                }
                for entry in v {
                    self.check_node(entry, depth + 1, nodes)?;
                }
                Ok(())
            }
            HydraNode::Chain { head, args, calls } => {
                check_identifier(head)?;
                if args.len() > MAX_HYDRA_ARGS {
                    return Err(HydraProgramError::TooMany {
                        what: "arguments",
                        limit: MAX_HYDRA_ARGS,
                        found: args.len(),
                    });
                }
                if calls.len() > MAX_HYDRA_CALLS {
                    return Err(HydraProgramError::TooMany {
                        what: "chained calls",
                        limit: MAX_HYDRA_CALLS,
                        found: calls.len(),
                    });
                }
                for argument in args {
                    self.check_node(argument, depth + 1, nodes)?;
                }
                for call in calls {
                    check_identifier(&call.method)?;
                    if call.args.len() > MAX_HYDRA_ARGS {
                        return Err(HydraProgramError::TooMany {
                            what: "arguments",
                            limit: MAX_HYDRA_ARGS,
                            found: call.args.len(),
                        });
                    }
                    for argument in &call.args {
                        self.check_node(argument, depth + 1, nodes)?;
                    }
                }
                Ok(())
            }
        }
    }
}

fn check_len(what: &'static str, value: &str, limit: usize) -> Result<(), HydraProgramError> {
    if value.len() > limit {
        return Err(HydraProgramError::TooLong {
            what,
            limit,
            found: value.len(),
        });
    }
    Ok(())
}

/// A name the renderer will look up on an object, so it must be a plain
/// identifier and nothing that could be read as an expression.
fn check_identifier(name: &str) -> Result<(), HydraProgramError> {
    check_len("name", name, MAX_HYDRA_NAME_BYTES)?;
    let mut characters = name.chars();
    let valid = match characters.next() {
        Some(first) if first.is_ascii_alphabetic() || first == '_' || first == '$' => {
            characters.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(HydraProgramError::Name(name.to_owned()))
    }
}
