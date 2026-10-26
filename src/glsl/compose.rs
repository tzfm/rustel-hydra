//! Fold a recorded chain into a fragment shader.
//!
//! A faithful port of `hydra-synth`'s `generate-glsl.js` and the shader
//! assembly at the end of `glsl-source.js`. The whitespace matters: it is
//! compared against what hydra itself emits, so the template literals over
//! there are reproduced here rather than tidied.

use std::fmt::Write as _;

use crate::glsl::table::{Function, Kind, UTILITY, function};
use crate::program::{HydraCall, HydraNode, MAX_HYDRA_SIGNALS};

#[derive(Debug, PartialEq)]
pub enum ComposeError {
    /// A name hydra does not answer to.
    Unknown(String),
    /// A chain that begins with something other than a source.
    NotASource(String),
    /// An argument this composer cannot yet turn into GLSL.
    Argument(String),
}

impl std::fmt::Display for ComposeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown(name) => write!(f, "hydra has no transform called `{name}`"),
            Self::NotASource(name) => write!(f, "`{name}` cannot begin a chain"),
            Self::Argument(what) => write!(f, "unsupported argument: {what}"),
        }
    }
}

/// Names that are values rather than transforms: the outputs, the sources,
/// and the clock.
///
/// A score's recording and a parsed snippet disagree about how to spell one.
/// The recorder emits `src(o0)`'s argument as a chain with the head `o0` and
/// nothing else - it cannot tell an uncalled generator from a called one by
/// the time it serialises - while the text parser emits `Global`. Rather than
/// teach one to imitate the other, both are read the same way here: a bare
/// head that names no transform but does name one of these is a value.
pub fn global_named(name: &str) -> bool {
    matches!(
        name,
        "o0" | "o1" | "o2" | "o3" | "s0" | "s1" | "s2" | "s3" | "time" | "prevBuffer"
    )
}

/// Read a node as a bare global, however it was spelled.
fn as_global(node: &HydraNode) -> Option<&str> {
    match node {
        HydraNode::Global { name } => Some(name),
        HydraNode::Chain { head, args, calls }
            if args.is_empty() && calls.is_empty() && global_named(head) =>
        {
            Some(head)
        }
        _ => None,
    }
}

/// One link of a chain: a transform and the arguments the score gave it.
struct Step<'a> {
    entry: &'static Function,
    args: &'a [HydraNode],
}

/// Which output a chain draws into, from its `.out(oN)`.
///
/// A chain with no `.out()` at all, or a bare `.out()`, goes to `o0` - which
/// is hydra's own default.
pub fn output_of(node: &HydraNode) -> usize {
    let HydraNode::Chain { calls, .. } = node else {
        return 0;
    };
    calls
        .iter()
        .find(|call| call.method == "out")
        .and_then(|call| {
            as_global(call.args.first()?)?
                .strip_prefix('o')
                .and_then(|index| index.parse::<usize>().ok())
                .filter(|index| *index < 4)
        })
        .unwrap_or(0)
}

/// Every `sN` and `oN` the composed shader samples, so the renderer knows
/// which textures have to be bound and - for an output - which one it must
/// not also be drawing into.
pub fn sampled(node: &HydraNode) -> Vec<String> {
    fn walk(node: &HydraNode, found: &mut Vec<String>) {
        match node {
            HydraNode::Chain { head, args, calls } => {
                let Some(entry) = function(head) else {
                    return;
                };
                if head == "src" || head == "prev" {
                    if let Some(name) = args.first().and_then(as_global) {
                        if !found.iter().any(|f| f == name) {
                            found.push(name.to_owned());
                        }
                    } else if head == "prev" && !found.iter().any(|f| f == "prevBuffer") {
                        found.push("prevBuffer".to_owned());
                    }
                }
                // Match `arguments`: excess arguments are ignored by Hydra
                // and `.out(...)` chooses a target rather than feeding the
                // shader. Neither position is a sampled texture.
                for (arg, (kind, _)) in args.iter().zip(call_inputs(entry)) {
                    if kind == "vec4" {
                        walk(arg, found);
                    }
                }
                for call in calls {
                    if call.method == "out" {
                        continue;
                    }
                    let Some(entry) = function(&call.method) else {
                        continue;
                    };
                    for (arg, (kind, _)) in call.args.iter().zip(call_inputs(entry)) {
                        if kind == "vec4" {
                            walk(arg, found);
                        }
                    }
                }
            }
            HydraNode::Global { name }
                if (name.starts_with('s') || name.starts_with('o')) && !found.contains(name) =>
            {
                found.push(name.clone());
            }
            _ => {}
        }
    }
    let mut found = Vec::new();
    walk(node, &mut found);
    found
}

/// Whether a valid drawable chain contains an explicit `src(source)` in an
/// argument position the shader actually consumes.
///
/// This deliberately does not treat a bare source name as sufficient. It is a
/// privacy boundary for camera-backed themes, where a source name hidden in
/// `.out(s0)`, `render(s0)`, or an ignored excess argument must not authorize
/// acquisition.
pub fn explicitly_samples(node: &HydraNode, source: &str) -> bool {
    fn renderable(node: &HydraNode) -> bool {
        let HydraNode::Chain { head, args, calls } = node else {
            return true;
        };
        let Some(entry) = function(head) else {
            return false;
        };
        if head == "sum" {
            return false;
        }
        if args
            .iter()
            .zip(call_inputs(entry))
            .any(|(arg, (kind, _))| kind == "vec4" && !renderable(arg))
        {
            return false;
        }
        calls.iter().all(|call| {
            if call.method == "out" {
                return true;
            }
            function(&call.method).is_some_and(|entry| {
                call.method != "sum"
                    && call
                        .args
                        .iter()
                        .zip(call_inputs(entry))
                        .all(|(arg, (kind, _))| kind != "vec4" || renderable(arg))
            })
        })
    }

    fn walk(node: &HydraNode, source: &str) -> bool {
        let HydraNode::Chain { head, args, calls } = node else {
            return false;
        };
        let Some(entry) = function(head) else {
            return false;
        };
        if head == "src" && args.first().and_then(as_global) == Some(source) {
            return true;
        }
        args.iter()
            .zip(call_inputs(entry))
            .any(|(arg, (kind, _))| kind == "vec4" && walk(arg, source))
            || calls.iter().any(|call| {
                if call.method == "out" {
                    return false;
                }
                function(&call.method).is_some_and(|entry| {
                    call.args
                        .iter()
                        .zip(call_inputs(entry))
                        .any(|(arg, (kind, _))| kind == "vec4" && walk(arg, source))
                })
            })
    }

    // A command-shaped or otherwise invalid chain cannot make a source
    // visible, even if it happens to contain a syntactic `src(s0)` argument.
    compose(node, "highp").is_ok() && renderable(node) && walk(node, source)
}

/// A shader, and the values a frame has to feed it.
#[derive(Debug, PartialEq)]
pub struct Composed {
    pub shader: String,
    /// One per `H(pattern)` in the chain, in the order the shader declares
    /// them: the uniform's name and the signal slot that fills it.
    pub signals: Vec<(String, u32)>,
}

/// Turn a recorded chain into the fragment shader hydra would have built.
pub fn compose(node: &HydraNode, precision: &str) -> Result<String, ComposeError> {
    Ok(compose_full(node, precision)?.shader)
}

/// The same, keeping the uniforms a frame must fill.
pub fn compose_full(node: &HydraNode, precision: &str) -> Result<Composed, ComposeError> {
    let steps = flatten(node)?;
    let mut used: Vec<&'static Function> = Vec::new();
    let mut signals: Vec<(String, u32)> = Vec::new();
    let body = generate(&steps, "c", "st", &mut used, &mut signals)?;

    let mut shader = String::new();
    // The leading newline and two-space indents are hydra's own template.
    let _ = write!(shader, "\n  precision {precision} float;\n  ");
    // Uniforms come before hydra's own, in declaration order, and in hydra's
    // own six-space indent - a function argument compiles to exactly this.
    for (name, _) in &signals {
        let _ = write!(shader, "\n      uniform float {name};");
    }
    shader.push_str("\n  uniform float time;\n  uniform vec2 resolution;\n  varying vec2 uv;\n  uniform sampler2D prevBuffer;\n\n  ");
    for helper in UTILITY {
        let _ = write!(shader, "\n            {helper}\n          ");
    }
    shader.push_str("\n\n  ");
    for entry in &used {
        let _ = write!(shader, "\n            {}\n          ", definition(entry));
    }
    shader.push_str("\n\n  void main () {\n    vec2 st = gl_FragCoord.xy/resolution.xy;\n\n    ");
    shader.push_str(&body);
    shader.push_str("\n    gl_FragColor = c;\n  }\n  ");
    Ok(Composed { shader, signals })
}

/// The complete GLSL definition, as `processGlsl` writes it.
fn definition(entry: &Function) -> String {
    let args = entry
        .kind
        .leading_args()
        .iter()
        .map(|(kind, name)| format!("{kind} {name}"))
        .chain(
            entry
                .inputs
                .iter()
                .map(|input| format!("{} {}", input.kind, input.name)),
        )
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "\n  {} {}({}) {{\n      {}\n  }}\n",
        entry.kind.return_type(),
        entry.name,
        args,
        entry.glsl
    )
}

/// A recorded chain is a head and a list of method calls; hydra wants them as
/// one flat sequence beginning with the source.
fn flatten(node: &HydraNode) -> Result<Vec<Step<'_>>, ComposeError> {
    let HydraNode::Chain { head, args, calls } = node else {
        return Err(ComposeError::NotASource(format!("{node:?}")));
    };
    let entry = function(head).ok_or_else(|| ComposeError::Unknown(head.clone()))?;
    if entry.kind != Kind::Src {
        return Err(ComposeError::NotASource(head.clone()));
    }
    let mut steps = vec![Step { entry, args }];
    for HydraCall { method, args } in calls {
        // `.out()` is where a chain goes, not part of what it draws.
        if method == "out" {
            continue;
        }
        let entry = function(method).ok_or_else(|| ComposeError::Unknown(method.clone()))?;
        steps.push(Step { entry, args });
    }
    Ok(steps)
}

/// The body of `main`, built the way `generateGlsl` builds it.
///
/// hydra folds the chain into a nest of closures and then calls the outermost;
/// the same thing read forwards is: each step wraps what came before, so the
/// text is assembled from the end of the chain back to its start.
fn generate(
    steps: &[Step<'_>],
    colour: &str,
    uv: &str,
    used: &mut Vec<&'static Function>,
    signals: &mut Vec<(String, u32)>,
) -> Result<String, ComposeError> {
    let mut text = String::new();
    for (index, step) in steps.iter().enumerate() {
        if !used.iter().any(|seen| seen.name == step.entry.name) {
            used.push(step.entry);
        }
        let scoped = format!("{colour}{index}");
        let inputs = arguments(step, &scoped, uv, used, signals)?;
        let call = format!(
            "{}({}{})",
            step.entry.name,
            target(step, colour, uv),
            inputs.text
        );
        text = match step.entry.kind {
            Kind::Src => format!("{}\n         vec4 {colour} = {call};", inputs.prelude),
            Kind::Color | Kind::Combine => {
                format!(
                    "{}\n         {text}\n         {colour} = {call};",
                    inputs.prelude
                )
            }
            Kind::Coord | Kind::CombineCoord => {
                format!(
                    "{}\n         {uv} = {call};\n         {text}",
                    inputs.prelude
                )
            }
        };
    }
    Ok(text)
}

/// What a transform is applied to: a coordinate, or the colour so far.
fn target(step: &Step<'_>, colour: &str, uv: &str) -> String {
    match step.entry.kind {
        Kind::Src | Kind::Coord | Kind::CombineCoord => uv.to_owned(),
        Kind::Color | Kind::Combine => colour.to_owned(),
    }
}

/// The arguments a chain actually passes, which are not quite the declared
/// ones.
///
/// `processGlsl` builds the definition from the kind's leading arguments plus
/// the declared inputs, then keeps `inputs.slice(1)` as what a call supplies.
/// So the first leading argument is the thing being transformed - implicit at
/// the call site - and any others become real arguments. That is how
/// `mult(osc(20))` ends up as `mult(c, c1_i0, 1.)`: the second colour, then
/// the declared `amount` filled in from its default.
fn call_inputs(entry: &'static Function) -> Vec<(&'static str, Option<f64>)> {
    entry
        .kind
        .leading_args()
        .iter()
        .skip(1)
        .map(|(kind, _)| (*kind, None))
        .chain(entry.inputs.iter().map(|input| (input.kind, input.default)))
        .collect()
}

struct Arguments {
    /// Nested chains, generated before the call that uses them.
    prelude: String,
    /// `, a, b, c` - what follows the target in the call.
    text: String,
}

/// Turn a step's arguments into GLSL, filling in hydra's defaults.
fn arguments(
    step: &Step<'_>,
    scoped: &str,
    uv: &str,
    used: &mut Vec<&'static Function>,
    signals: &mut Vec<(String, u32)>,
) -> Result<Arguments, ComposeError> {
    let mut prelude = String::new();
    let mut text = String::new();
    // hydra names an input's variable after the step, not the argument slot it
    // came from, so `scoped` is threaded through rather than recomputed.
    for (index, (kind, default)) in call_inputs(step.entry).into_iter().enumerate() {
        let given = step.args.get(index);
        let rendered = match given {
            None => match default {
                // Only a `float` is written with a forced decimal point.
                // hydra's `ensure_decimal_dot` is guarded on the type, so a
                // `vec4` default of 1 stays `1` - `sum()` is the one case.
                Some(value) if kind == "float" => decimal(value),
                Some(value) => plain(value),
                None => {
                    return Err(ComposeError::Argument(format!(
                        "`{}` needs a {kind} in position {index}",
                        step.entry.name
                    )));
                }
            },
            // Snippet text reaches here through the parser, which already
            // refuses such a literal as written; a score's JSON does not pass
            // the parser, and carries 1e300 as readily as 0.5. This arm is
            // where both front ends meet, so the bound holds for both.
            Some(HydraNode::Number { v }) if kind == "float" => {
                if !shader_float(*v) {
                    return Err(ComposeError::Argument(format!(
                        "in `{}`, {}",
                        step.entry.name,
                        out_of_range(&format!("{v:e}"))
                    )));
                }
                decimal(*v)
            }
            Some(HydraNode::Number { .. }) => {
                return Err(ComposeError::Argument(format!(
                    "`{}` was given a number where it needs a {kind}",
                    step.entry.name
                )));
            }
            // A bare global, however it was spelled. This arm must come
            // before the nested-chain arm below, because a recorded `o0` is
            // structurally a chain and would otherwise be compiled as one.
            Some(node) if as_global(node).is_some() => {
                let name = as_global(node).expect("just matched");
                if (kind == "sampler2D"
                    && step.entry.name == "src"
                    && matches!(
                        name,
                        "s0" | "s1" | "s2" | "s3" | "o0" | "o1" | "o2" | "o3" | "prevBuffer"
                    ))
                    || (kind == "float" && name == "time")
                {
                    name.to_owned()
                } else if kind == "vec4"
                    && (name.starts_with('s') || name.starts_with('o') || name == "prevBuffer")
                {
                    // hydra-synth accepts `.modulate(o0)` and other combine
                    // inputs as shorthand for `.modulate(src(o0))`. The score
                    // recorder deliberately keeps the bare global typed, so
                    // make that implicit source chain here rather than
                    // weakening numeric argument checks.
                    let source = HydraNode::Chain {
                        head: "src".into(),
                        args: vec![node.clone()],
                        calls: Vec::new(),
                    };
                    let nested = flatten(&source)?;
                    let inner_colour = format!("{scoped}_i{index}");
                    let inner_uv = format!("{uv}_{scoped}_i{index}");
                    let inner = generate(&nested, &inner_colour, &inner_uv, used, signals)?;
                    prelude = format!("vec2 {inner_uv} = {uv};{prelude}\n         {inner}");
                    inner_colour
                } else {
                    return Err(ComposeError::Argument(format!(
                        "`{}` was given `{name}`, which is not a {kind}",
                        step.entry.name
                    )));
                }
            }
            Some(node @ HydraNode::Chain { .. }) if kind == "vec4" => {
                // A chain given as an argument is generated into its own
                // variable first. hydra wraps each such input around the ones
                // before it, so the last one's declaration ends up outermost -
                // reproduced here by folding the accumulated text inside.
                let nested = flatten(node)?;
                let inner_colour = format!("{scoped}_i{index}");
                let inner_uv = format!("{uv}_{scoped}_i{index}");
                let inner = generate(&nested, &inner_colour, &inner_uv, used, signals)?;
                prelude = format!("vec2 {inner_uv} = {uv};{prelude}\n         {inner}");
                inner_colour
            }
            Some(node @ HydraNode::Chain { .. }) => {
                return Err(ComposeError::Argument(format!(
                    "`{}` was given a texture chain where it needs a {kind}: {node:?}",
                    step.entry.name
                )));
            }
            // `src(s0)` and `src(o1)` name a texture uniform; the name is
            // what GLSL uses, and the renderer binds it.
            // `H(pattern)` is a value that changes every frame, which is a
            // uniform. hydra reaches the same shape from the other side: a
            // `() => …` argument becomes `uniform float <input><n>`, where n
            // counts every uniform declared so far. Matching that naming is
            // what keeps a signal chain byte-identical to hydra's own.
            Some(HydraNode::Signal { slot }) if kind == "float" => {
                if *slot as usize >= MAX_HYDRA_SIGNALS {
                    return Err(ComposeError::Argument(format!(
                        "signal slot {slot} is outside 0..{MAX_HYDRA_SIGNALS}"
                    )));
                }
                let name = format!("{}{}", input_name(step.entry, index), signals.len());
                signals.push((name.clone(), *slot));
                name
            }
            Some(HydraNode::Signal { .. }) => {
                return Err(ComposeError::Argument(format!(
                    "`{}` was given a signal where it needs a {kind}",
                    step.entry.name
                )));
            }
            // `osc(() => Math.sin(time))` is idiomatic Hydra, and there is
            // no JavaScript here to run it. Upstream evaluates the callback
            // once a frame and passes a uniform; this compiles the expression
            // into the shader instead, where `time` and the audio bands
            // already live. The value stops being sampled and becomes exact.
            Some(HydraNode::Source { src }) if kind == "float" => {
                match crate::glsl::expr::compile(src) {
                    Ok(glsl) => glsl,
                    Err(error) => {
                        return Err(ComposeError::Argument(format!(
                            "`{}` was given a function this cannot compile: {error}. \
                         Arithmetic on `time` and `a.fft[0..15]` works; for a \
                         value from the music, `H(pattern)` reads the transport",
                            step.entry.name
                        )));
                    }
                }
            }
            Some(HydraNode::Source { .. }) => {
                return Err(ComposeError::Argument(format!(
                    "`{}` was given a callback where it needs a {kind}",
                    step.entry.name
                )));
            }
            Some(HydraNode::Text { .. }) => {
                return Err(ComposeError::Argument(format!(
                    "`{}` was given a string, which no Hydra transform takes",
                    step.entry.name
                )));
            }
            // A list where a float goes is hydra's animated argument, compiled
            // into the shader around `time`. hydra's `bpm` stays at its
            // default 30, since a score's assignment to it is refused.
            Some(list @ HydraNode::List { .. }) if kind == "float" => {
                list_expression(list, step.entry.name)?
            }
            Some(other) => {
                return Err(ComposeError::Argument(format!(
                    "`{}` was given something this cannot compile: {other:?}",
                    step.entry.name
                )));
            }
        };
        let _ = write!(text, ", {rendered}");
    }
    Ok(Arguments { prelude, text })
}

/// Whether a number can be written into the shader as a float literal.
///
/// A GLSL `float` is an f32, not an f64. naga's GLSL lexer reads every
/// literal's text with `parse::<f32>()`, so anything past `f32::MAX` -
/// `1e39`, `1e300`, a forty-digit integer, all finite as f64 - lexes as
/// infinity and fails `LiteralError::Infinity` at pipeline build: composition
/// succeeds, then every frame retries the uncached build and fails with a GPU
/// diagnostic that names nothing about the literal.
///
/// `as` rounds to nearest, so a value that rounds to `f32::MAX` - its own
/// shortest spelling `3.4028235e38` among them - stays legal, exactly as the
/// lexer treats it. The one f64 this refuses that naga would take is the
/// rounding midpoint 2^128 − 2^103 itself, which `as` sends to infinity on the
/// tie while its printed decimal sits a hair below; refusing it is the safe
/// side of a bound no sketch will be written against.
pub(crate) fn shader_float(value: f64) -> bool {
    (value as f32).is_finite()
}

/// The refusal for a literal [`shader_float`] rejects, worded once so the
/// parser, the composer and the callback compiler all say the same thing.
pub(crate) fn out_of_range(literal: &str) -> String {
    format!(
        "`{literal}` is out of range for a shader float, whose largest magnitude is {:e}",
        f32::MAX
    )
}

/// A number the way JavaScript prints it: `10`, not `10.0`.
fn plain(value: f64) -> String {
    // A shader float is an f32: past its range the literal lexes as infinity
    // and fails every pipeline build, per frame and uncached. The parser and
    // the composer's float arm refuse such numbers by name; this is the one
    // place every f64 becomes GLSL text, so a third path trips here instead.
    debug_assert!(shader_float(value), "a shader float cannot hold {value:e}");
    if value == value.trunc() && value.abs() < 1e15 {
        format!("{}", value.trunc() as i64)
    } else {
        format!("{value}")
    }
}

/// An easing curve, from the expression for `t` to the expression for the
/// eased value.
type Curve = fn(&str) -> String;

/// hydra's easing table, in hydra's order. The curves are hydra-synth's
/// easing-functions.js, whose `--t` reads `(t - 1)` here.
const EASINGS: [(&str, Curve); 14] = [
    ("linear", |t| t.to_owned()),
    ("easeInQuad", |t| power(t, 2)),
    ("easeOutQuad", |t| format!("(({t})*(2.0-({t})))")),
    ("easeInOutQuad", |t| {
        format!("(({t})<0.5 ? 2.0*({t})*({t}) : -1.0+(4.0-2.0*({t}))*({t}))")
    }),
    ("easeInCubic", |t| power(t, 3)),
    ("easeOutCubic", |t| {
        format!("({}+1.0)", power(&less_one(t), 3))
    }),
    ("easeInOutCubic", |t| {
        format!(
            "(({t})<0.5 ? 4.0*{} : 4.0*{}+1.0)",
            power(t, 3),
            power(&less_one(t), 3)
        )
    }),
    ("easeInQuart", |t| power(t, 4)),
    ("easeOutQuart", |t| {
        format!("(1.0-{})", power(&less_one(t), 4))
    }),
    ("easeInOutQuart", |t| {
        format!(
            "(({t})<0.5 ? 8.0*{} : 1.0-8.0*{})",
            power(t, 4),
            power(&less_one(t), 4)
        )
    }),
    ("easeInQuint", |t| power(t, 5)),
    ("easeOutQuint", |t| {
        format!("(1.0+{})", power(&less_one(t), 5))
    }),
    ("easeInOutQuint", |t| {
        format!(
            "(({t})<0.5 ? 16.0*{} : 1.0+16.0*{})",
            power(t, 5),
            power(&less_one(t), 5)
        )
    }),
    ("sin", |t| {
        format!("((1.0+sin(3.14159265358979*({t})-1.5707963267949))/2.0)")
    }),
];

/// `t` multiplied by itself `n` times.
fn power(t: &str, n: usize) -> String {
    format!("({})", vec![format!("({t})"); n].join("*"))
}

/// `t - 1`, for the curves hydra writes with `--t`.
fn less_one(t: &str) -> String {
    format!("({t})-1.0")
}

/// Whether `names` are exactly the easings [`EASINGS`] compiles, in order.
/// `rustel-runtime` checks the score recorder's list against this at compile
/// time.
pub const fn compiles_exactly_these_easings(names: &[&str]) -> bool {
    if names.len() != EASINGS.len() {
        return false;
    }
    let mut at = 0;
    while at < names.len() {
        let (name, known) = (names[at].as_bytes(), EASINGS[at].0.as_bytes());
        if name.len() != known.len() {
            return false;
        }
        let mut byte = 0;
        while byte < name.len() {
            if name[byte] != known[byte] {
                return false;
            }
            byte += 1;
        }
        at += 1;
    }
    true
}

/// One of hydra's easings applied to `t`, or a refusal naming the table.
fn ease_expression(ease: &str, t: &str) -> Result<String, ComposeError> {
    let Some((_, curve)) = EASINGS.iter().find(|(name, _)| *name == ease) else {
        let table = EASINGS.map(|(name, _)| name).join(", ");
        return Err(ComposeError::Argument(format!(
            "`{ease}` is not one of hydra's easings: {table}"
        )));
    };
    Ok(curve(t))
}

/// The animated value of a `[…]` argument as one inline expression in the
/// `time` uniform: hydra-synth's `getValue`, stepping through the entries or
/// easing across `.smooth()`'s window, at hydra's default 30 bpm.
///
/// Each entry is selected by a triangular window rather than a dynamically
/// indexed array, which GLSL ES 1.00 does not promise. A negative index, or a
/// smoothing window that starts before 0, wraps Euclidean, where hydra reads
/// `undefined`.
fn list_expression(list: &HydraNode, function: &'static str) -> Result<String, ComposeError> {
    let HydraNode::List {
        v,
        speed,
        smooth,
        ease,
        offset,
    } = list
    else {
        return Err(ComposeError::Argument(format!(
            "`{function}` was given a list where it needs a float"
        )));
    };
    if v.is_empty() {
        return Err(ComposeError::Argument(format!(
            "`{function}` was given an empty list, which cannot animate"
        )));
    }
    for (mark, value) in [("fast", *speed), ("smooth", *smooth), ("offset", *offset)] {
        if !shader_float(value) {
            return Err(ComposeError::Argument(format!(
                "in `{function}`, the list's .{mark}: {}",
                out_of_range(&format!("{value:e}"))
            )));
        }
    }
    let mut entries = Vec::with_capacity(v.len());
    for entry in v {
        let HydraNode::Number { v: value } = entry else {
            return Err(ComposeError::Argument(format!(
                "`{function}` was given a list holding something other than a number"
            )));
        };
        if !shader_float(*value) {
            return Err(ComposeError::Argument(format!(
                "in `{function}`, {}",
                out_of_range(&format!("{value:e}"))
            )));
        }
        entries.push(decimal(*value));
    }
    let count = decimal(entries.len() as f64);
    let window = |index: &str| -> String {
        entries
            .iter()
            .enumerate()
            .map(|(at, value)| {
                format!(
                    "{value}*clamp(1.0-abs({index}-{}), 0.0, 1.0)",
                    decimal(at as f64)
                )
            })
            .collect::<Vec<_>>()
            .join("+")
    };
    // `time * speed * bpm / 60 + offset`, with bpm 30.
    let index = format!("(time*{}+{})", decimal(*speed * 0.5), decimal(*offset));
    if *smooth == 0.0 || entries.len() == 1 {
        let at = format!("mod(floor({index}),{count})");
        return Ok(format!("({})", window(&at)));
    }
    let back = format!("({index}-({}))", decimal(*smooth * 0.5));
    let share = format!("(min(mod({back},1.0)/({}), 1.0))", decimal(*smooth));
    let current = window(&format!("mod(floor({back}),{count})"));
    let next = window(&format!("mod(floor({back}+1.0),{count})"));
    let eased = ease_expression(ease, &share)?;
    Ok(format!("({eased}*(({next})-({current}))+({current}))"))
}

/// The declared name of the argument in this position, for naming a uniform
/// after it the way hydra does.
fn input_name(entry: &'static Function, index: usize) -> &'static str {
    let leading = entry.kind.leading_args().len().saturating_sub(1);
    if index < leading {
        // The implicit second colour of a `combine`; hydra never gives this
        // one a function, but a name is needed either way.
        return "input";
    }
    entry
        .inputs
        .get(index - leading)
        .map_or("input", |input| input.name)
}

/// hydra writes every `float` with a dot, so `10` becomes `10.`.
fn decimal(value: f64) -> String {
    let text = plain(value);
    if text.contains('.') {
        text
    } else {
        format!("{text}.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod animated_lists {
        //! Arrays as animated arguments: `[…]` where a float goes.

        use super::super::*;
        use wgpu::naga;

        /// A list node as the score recorder writes it.
        fn animated(values: &[f64], speed: f64, smooth: f64, ease: &str, offset: f64) -> HydraNode {
            HydraNode::List {
                v: values
                    .iter()
                    .map(|value| HydraNode::Number { v: *value })
                    .collect(),
                speed,
                smooth,
                ease: ease.to_owned(),
                offset,
            }
        }

        /// `solid(list, 2, 3)`, drawn to `o0`.
        fn solid(list: HydraNode) -> HydraNode {
            HydraNode::Chain {
                head: "solid".into(),
                args: vec![
                    list,
                    HydraNode::Number { v: 2.0 },
                    HydraNode::Number { v: 3.0 },
                ],
                calls: vec![HydraCall {
                    method: "out".into(),
                    args: vec![HydraNode::Global { name: "o0".into() }],
                }],
            }
        }

        /// hydra-synth's `getValue` at its default 30 bpm, with a negative index
        /// wrapped where hydra reads `undefined`.
        fn get_value(
            entries: &[f64],
            speed: f64,
            smooth: f64,
            ease: fn(f64) -> f64,
            offset: f64,
            seconds: f64,
        ) -> f64 {
            let modulo = |n: f64, d: f64| ((n % d) + d) % d;
            let length = entries.len() as f64;
            let index = seconds * speed * 30.0 / 60.0 + offset;
            if smooth == 0.0 {
                return entries[modulo(index, length).floor() as usize];
            }
            let back = index - smooth / 2.0;
            let current = entries[modulo(back, length).floor() as usize];
            let next = entries[modulo(back + 1.0, length).floor() as usize];
            let t = (modulo(back, 1.0) / smooth).min(1.0);
            ease(t) * (next - current) + current
        }

        /// The value naga's constant evaluator gives `expression` with `time` held
        /// at `seconds`.
        fn folded(expression: &str, seconds: f64) -> f32 {
            let source = format!(
                "#version 450\nconst float time = {seconds:?};\nconst float v = {expression};\nvoid main() {{}}\n"
            );
            let module = naga::front::glsl::Frontend::default()
                .parse(
                    &naga::front::glsl::Options::from(naga::ShaderStage::Fragment),
                    &source,
                )
                .unwrap_or_else(|error| panic!("naga folds {expression}: {error:?}"));
            let (_, value) = module
                .constants
                .iter()
                .find(|(_, constant)| constant.name.as_deref() == Some("v"))
                .expect("the folded constant");
            match module.global_expressions[value.init] {
                naga::Expression::Literal(naga::Literal::F32(value)) => value,
                ref other => panic!("{expression} did not fold to a float: {other:?}"),
            }
        }

        /// Every shape of list, and every easing in the table, composes into a
        /// shader that naga parses and validates, as the renderer requires.
        #[test]
        fn an_animated_list_composes_into_a_shader_naga_accepts() {
            let shapes = [
                animated(&[0.8, 4.0], 1.0, 1.0, "linear", 0.0),
                animated(&[1.0, 20.0], 0.1, 1.0, "linear", 0.0),
                animated(&[0.0, 1.0, 2.0], 2.0, 0.0, "linear", 0.0),
                animated(&[0.0, 1.0], -1.0, 0.0, "linear", -0.25),
                animated(&[0.0, 1.0], 1.0, -0.5, "linear", 0.0),
                animated(&[10.0], 1.0, 1.0, "linear", 0.0),
            ];
            let easings = EASINGS
                .iter()
                .map(|(name, _)| animated(&[-1.0, 1.0], 0.5, 2.0, name, 0.25));
            for list in shapes.into_iter().chain(easings) {
                let composed = compose_full(&solid(list.clone()), "highp")
                    .unwrap_or_else(|error| panic!("{list:?} composes: {error}"));
                let shader =
                    crate::native::uplift_with_signal_slots(&composed.shader, &composed.signals);
                let module = naga::front::glsl::Frontend::default()
                    .parse(
                        &naga::front::glsl::Options::from(naga::ShaderStage::Fragment),
                        &shader,
                    )
                    .unwrap_or_else(|error| panic!("naga parses {list:?}: {error:?}"));
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::default(),
                )
                .validate(&module)
                .unwrap_or_else(|error| panic!("naga validates {list:?}: {error}"));
            }
        }

        /// The composed expression takes the value hydra's `getValue` gives at the
        /// same time, stepped and smoothed, with speed, offset and easing marks, and a
        /// negative index wrapped.
        #[test]
        fn an_animated_list_walks_like_hydras_get_value() {
            let linear: fn(f64) -> f64 = |t| t;
            let in_quad: fn(f64) -> f64 = |t| t * t;
            let out_cubic: fn(f64) -> f64 = |t| (t - 1.0).powi(3) + 1.0;
            let sine: fn(f64) -> f64 =
                |t| (1.0 + (std::f64::consts::PI * t - std::f64::consts::FRAC_PI_2).sin()) / 2.0;
            // Entries, speed, smooth, easing name and curve, offset.
            type Walk<'a> = (&'a [f64], f64, f64, &'a str, fn(f64) -> f64, f64);
            let cases: [Walk; 9] = [
                (&[0.8, 4.0], 1.0, 1.0, "linear", linear, 0.0),
                (&[5.0, 3.0], 1.0, 1.0, "linear", linear, 0.0),
                (&[1.0, 20.0], 0.1, 1.0, "linear", linear, 0.0),
                (&[0.0, 10.0, 20.0], 1.0, 0.0, "linear", linear, 0.0),
                (&[0.0, 10.0, 20.0], 3.0, 0.0, "linear", linear, 0.5),
                (&[0.0, 1.0, 5.0], 2.0, 0.25, "sin", sine, 0.25),
                (&[-1.0, 1.0], 0.5, 2.0, "easeInQuad", in_quad, 0.0),
                (&[2.0, 7.0, 3.0], 1.0, 1.0, "easeOutCubic", out_cubic, 0.0),
                (&[0.0, 10.0], 1.0, -0.5, "linear", linear, 0.0),
            ];
            for (entries, speed, smooth, ease, curve, offset) in cases {
                let list = animated(entries, speed, smooth, ease, offset);
                let expression = list_expression(&list, "solid").expect("the list composes");
                for seconds in [0.3, 1.7, 2.9, 5.3, 11.1, 37.7] {
                    let expected = get_value(entries, speed, smooth, curve, offset, seconds);
                    let actual = f64::from(folded(&expression, seconds));
                    assert!(
                        (actual - expected).abs() < 1e-3,
                        "{list:?} at {seconds}s is {actual}, hydra gives {expected}"
                    );
                }
            }
        }

        /// A list that cannot animate is refused with the reason.
        #[test]
        fn lists_that_cannot_animate_are_refused_with_reasons() {
            let refusal = |list: HydraNode| {
                compose(&solid(list), "highp")
                    .expect_err("the list is refused")
                    .to_string()
            };
            assert!(
                refusal(animated(&[], 1.0, 0.0, "linear", 0.0))
                    .contains("an empty list, which cannot animate")
            );
            let textual = HydraNode::List {
                v: vec![HydraNode::Text { v: "no".into() }],
                speed: 1.0,
                smooth: 0.0,
                ease: "linear".into(),
                offset: 0.0,
            };
            assert!(refusal(textual).contains("something other than a number"));
            assert!(
                refusal(animated(&[0.0, 1.0], 1.0, 1.0, "warp", 0.0))
                    .contains("`warp` is not one of hydra's easings: linear, easeInQuad,")
            );
            assert!(
                refusal(animated(&[0.0, 1.0], 1e300, 0.0, "linear", 0.0))
                    .contains("the list's .fast: `1e300` is out of range for a shader float")
            );
            assert!(
                refusal(animated(&[0.0, 1.0], 1.0, f64::NAN, "linear", 0.0))
                    .contains("the list's .smooth")
            );
        }

        /// The easing check accepts the table's names in order and nothing else.
        #[test]
        fn the_easing_check_accepts_exactly_the_table() {
            let names = EASINGS.map(|(name, _)| name);
            assert!(compiles_exactly_these_easings(&names));
            assert!(!compiles_exactly_these_easings(&names[1..]));
            let mut misspelled = names;
            misspelled[13] = "sine";
            assert!(!compiles_exactly_these_easings(&misspelled));
            let mut reordered = names;
            reordered.swap(0, 1);
            assert!(!compiles_exactly_these_easings(&reordered));
        }
    }

    fn parsed(code: &str) -> HydraNode {
        crate::glsl::parse_chain(code).expect("test chain parses")
    }

    #[test]
    fn a_signal_slot_past_the_signal_block_is_refused() {
        let osc = |slot| HydraNode::Chain {
            head: "osc".into(),
            args: vec![HydraNode::Signal { slot }],
            calls: Vec::new(),
        };
        let last = MAX_HYDRA_SIGNALS as u32 - 1;
        assert!(compose_full(&osc(last), "highp").is_ok());
        assert!(matches!(
            compose_full(&osc(last + 1), "highp"),
            Err(ComposeError::Argument(_))
        ));
    }

    #[test]
    fn explicit_sampling_ignores_output_commands_and_unused_arguments() {
        let visible = parsed("osc(3).modulate(src(s0), 0.2).out()");
        assert!(explicitly_samples(&visible, "s0"));
        assert!(sampled(&visible).iter().any(|source| source == "s0"));

        for invisible in [
            "osc(3).out(s0)",
            "osc(3).out(src(s0))",
            "osc(src(s0), 0.1, 0).out()",
            "src(src(s0)).out()",
            "osc(3, 0.1, 0, s0).out()",
            "osc(3, 0.1, 0, src(s0)).out()",
            "osc(3).color(1, 1, 1, 1, s0).out()",
            "render(s0)",
            "render(src(s0))",
        ] {
            let node = parsed(invisible);
            assert!(
                !explicitly_samples(&node, "s0"),
                "control-only source must stay invisible: {invisible}"
            );
            assert!(
                !sampled(&node).iter().any(|source| source == "s0"),
                "sample inventory must match generated shader: {invisible}"
            );
        }

        for invalid in [
            "osc(src(s0), 0.1, 0).out()",
            "src(src(s0)).out()",
            "src(s0).blend(1).out()",
            "src(s0).blend(src(1)).out()",
        ] {
            assert!(
                compose(&parsed(invalid), "highp").is_err(),
                "a texture chain cannot fill a non-vec4 argument: {invalid}"
            );
        }

        assert!(
            compose(&parsed("src(s0).sum().out()"), "highp").is_ok(),
            "the parity composer preserves upstream's broken sum source"
        );
        assert!(
            !explicitly_samples(&parsed("src(s0).sum().out()"), "s0"),
            "a known GPU-refused chain cannot authorize camera acquisition"
        );
    }

    /// A chain the way a score's recording arrives: JSON straight into the
    /// node tree, with no parser in between to refuse anything.
    fn recorded(json: &str) -> HydraNode {
        serde_json::from_str(json).expect("test recording deserialises")
    }

    #[test]
    fn a_score_number_no_shader_float_can_hold_is_refused_by_name() {
        // JSON carries any finite f64, so the score path hands the composer
        // numbers the snippet parser would have refused, in every argument
        // position a float fills: a source's, and a method's.
        for (json, named) in [
            (
                r#"{"c":"call","head":"osc","args":[{"c":"num","v":1e300}]}"#,
                "in `osc`, `1e300`",
            ),
            (
                r#"{"c":"call","head":"osc","args":[{"c":"num","v":-1e39}]}"#,
                "in `osc`, `-1e39`",
            ),
            (
                r#"{"c":"call","head":"osc","args":[],"calls":[{"method":"rotate","args":[{"c":"num","v":1e39}]}]}"#,
                "in `rotate`, `1e39`",
            ),
            (
                r#"{"c":"call","head":"osc","args":[{"c":"num","v":3.4028236e38}]}"#,
                "in `osc`, `3.4028236e38`",
            ),
        ] {
            let node = recorded(json);
            let Err(ComposeError::Argument(error)) = compose(&node, "highp") else {
                panic!("a shader float cannot hold this: {json}");
            };
            assert!(error.contains(named), "names the value: {error}");
            assert!(
                error.contains("is out of range for a shader float"),
                "says why: {error}"
            );
        }

        // The same node built directly, as the score recorder's own types do.
        let node = HydraNode::Chain {
            head: "osc".into(),
            args: vec![HydraNode::Number { v: 1e300 }],
            calls: Vec::new(),
        };
        assert_eq!(
            compose(&node, "highp").map_err(|error| error.to_string()),
            Err(
                "unsupported argument: in `osc`, `1e300` is out of range for a shader \
             float, whose largest magnitude is 3.4028235e38"
                    .to_owned()
            )
        );
    }

    #[test]
    fn numbers_a_shader_float_can_hold_compose_as_before() {
        for (json, spelled) in [
            (
                r#"{"c":"call","head":"osc","args":[{"c":"num","v":3e38}]}"#,
                "osc(st, 300000000000000000000000000000000000000., 0.1, 0.)",
            ),
            (
                // `f32::MAX`'s shortest spelling rounds to it: legal.
                r#"{"c":"call","head":"osc","args":[{"c":"num","v":-3.4028235e38}]}"#,
                "osc(st, -340282350000000000000000000000000000000., 0.1, 0.)",
            ),
            (
                r#"{"c":"call","head":"osc","args":[{"c":"num","v":0.5}]}"#,
                "osc(st, 0.5, 0.1, 0.)",
            ),
        ] {
            let shader = compose(&recorded(json), "highp").expect("a shader float holds this");
            assert!(shader.contains(spelled), "{spelled} in {shader}");
        }

        // Only what reaches the shader is bounded: hydra ignores an argument
        // past the ones a transform declares, and so does this.
        compose(
            &recorded(
                r#"{"c":"call","head":"osc","args":[{"c":"num","v":1},{"c":"num","v":0.1},{"c":"num","v":0},{"c":"num","v":1e300}]}"#,
            ),
            "highp",
        )
        .expect("an ignored excess argument never becomes GLSL");
    }

    #[test]
    fn the_float_bound_agrees_with_the_f32_parse_the_shader_lexer_makes() {
        // naga's GLSL lexer reads each float literal's text with
        // `parse::<f32>()`. Sweep the f64s either side of the point where an
        // f32 rounds to infinity, and either side of `f32::MAX` itself, and
        // check the bound against that parse of the text this composer emits.
        //
        // At this size `decimal` spells a number as Rust's `Display` does
        // plus a point; that spelling is rebuilt here because `decimal`
        // itself asserts the very bound under test.
        let midpoint = f64::from(f32::MAX) + 2f64.powi(103);
        assert_eq!(
            decimal(f64::from(f32::MAX)),
            format!("{}.", f64::from(f32::MAX))
        );
        let mut disagreements = Vec::new();
        for centre in [midpoint, f64::from(f32::MAX)] {
            for step in -2_000_i64..=2_000 {
                let value = f64::from_bits(centre.to_bits().wrapping_add_signed(step));
                let lexed = format!("{value}.")
                    .parse::<f32>()
                    .expect("composed literals parse");
                if shader_float(value) != lexed.is_finite() {
                    disagreements.push(value);
                }
            }
        }
        // The bound never admits what the lexer reads as infinity. It refuses
        // exactly one value the lexer would take: the midpoint itself, where
        // `as` breaks the tie upward while its printed decimal sits below.
        assert_eq!(disagreements, vec![midpoint]);
        assert!(!shader_float(midpoint));
    }
}
