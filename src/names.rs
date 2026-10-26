//! Hydra's own vocabulary, for the editor's checker.
//!
//! The score realm does not need this list: the recorder and the composer
//! have their own tables, so a Hydra function works whether or not it appears
//! here. The checker does: without it, a valid sketch gets a note such as
//! "unknown function `diff`" for each Hydra name.
//!
//! Being out of date therefore costs a spurious note, never a broken sketch.
//!
//! Extracted from the `hydra-synth` source rather than typed out. To
//! refresh, run over the matching `hydra-synth` tarball:
//!
//! ```text
//! grep -oE "name: '[A-Za-z_][A-Za-z0-9_]*',\\s*type: '[a-zA-Z]+'" \\
//!     src/glsl/glsl-functions.js
//! ```

/// Generators and transforms generated from `glsl-functions.js`.
const GLSL: &[&str] = &[
    "a",
    "add",
    "b",
    "blend",
    "brightness",
    "color",
    "colorama",
    "contrast",
    "diff",
    "g",
    "gradient",
    "hue",
    "invert",
    "kaleid",
    "layer",
    "luma",
    "mask",
    "modulate",
    "modulateHue",
    "modulateKaleid",
    "modulatePixelate",
    "modulateRepeat",
    "modulateRepeatX",
    "modulateRepeatY",
    "modulateRotate",
    "modulateScale",
    "modulateScrollX",
    "modulateScrollY",
    "mult",
    "noise",
    "osc",
    "pixelate",
    "posterize",
    "prev",
    "r",
    "repeat",
    "repeatX",
    "repeatY",
    "rotate",
    "saturate",
    "scale",
    "scroll",
    "scrollX",
    "scrollY",
    "shape",
    "shift",
    "solid",
    "src",
    "sub",
    "sum",
    "thresh",
    "voronoi",
];

/// The rest of hydra-synth's surface: outputs, sources, settings, and the
/// array modifiers.
const API: &[&str] = &[
    "afterUpdate",
    "clear",
    "ease",
    "fast",
    "fit",
    "fps",
    "hide",
    "hush",
    "init",
    "initCam",
    "initCanvas",
    "initImage",
    "initScreen",
    "initStream",
    "initVideo",
    "offset",
    "out",
    "render",
    "setBins",
    "setCutoff",
    "setMax",
    "setResolution",
    "setScale",
    "setSmooth",
    "show",
    "smooth",
    "tick",
    "update",
    "o0",
    "o1",
    "o2",
    "o3",
    "s0",
    "s1",
    "s2",
    "s3",
    "a",
    "bpm",
    "height",
    "mouse",
    "pixelRatio",
    "speed",
    "time",
    "width",
];

/// Every name a Hydra sketch may call, in one iterator.
pub fn hydra_names() -> impl Iterator<Item = &'static str> {
    GLSL.iter().copied().chain(API.iter().copied())
}
