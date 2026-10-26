//! Bring hydra's GLSL up to the dialect a modern shader compiler accepts.
//!
//! `hydra-synth` writes GLSL ES 1.00, the WebGL 1 dialect: `varying`,
//! `gl_FragColor`, `texture2D`, and bare uniforms with no binding. naga wants
//! something closer to Vulkan GLSL. Most of the gap is in the preamble.
//! Hydra's function bodies otherwise keep their arithmetic. The pass rewrites
//! only the coordinate setup in `main`, texture fetches, and sampler
//! parameters and arguments.
//!
//! Every transform in `glsl::FUNCTIONS` except `sum()` parses and validates
//! after this pass. `sum()` fails because of a bug in hydra itself; see
//! `every_transform_in_the_table_renders` in `tests/native_render.rs`.

/// Where the frame's uniforms live in the bind group.
pub const GLOBALS_BINDING: u32 = 0;

/// Every texture a sketch can name, in the order their bindings are assigned.
/// `prevBuffer` is last because hydra declares it unconditionally.
pub const TEXTURES: [&str; 9] = ["s0", "s1", "s2", "s3", "o0", "o1", "o2", "o3", "prevBuffer"];

/// Where the per-frame signal values live.
pub const SIGNALS_BINDING: u32 = GLOBALS_BINDING + 1 + TEXTURES.len() as u32 * 2;

/// How many `H(pattern)` values one sketch may carry.
///
/// Sixteen `vec4`s, because a uniform array of scalars pads each element to
/// sixteen bytes in std140 and this way none of it is padding. This matches
/// the recording protocol's 64-signal ceiling.
pub const MAX_SIGNALS: usize = crate::MAX_HYDRA_SIGNALS;

/// Rewrite one composed shader for the native renderer.
///
/// `signals` are the uniform names [`crate::glsl::compose_full`] reported, in
/// declaration order; each becomes a slot in the signal block.
pub fn uplift_with_signals(frag: &str, signals: &[String]) -> String {
    let signals: Vec<(String, u32)> = signals
        .iter()
        .enumerate()
        .map(|(slot, name)| (name.clone(), slot as u32))
        .collect();
    uplift_with_signal_slots(frag, &signals)
}

/// Rewrite a composed shader while preserving each signal's protocol slot.
/// A statement may use only `H` slot 16, for example; occurrence order within
/// that one shader must not accidentally read slot zero. Every occurrence
/// aliases its slot, so repeated uses need no additional signal slots.
pub fn uplift_with_signal_slots(frag: &str, signals: &[(String, u32)]) -> String {
    let mut shader = uplift(frag);
    let block = format!(
        "layout(set=0, binding={SIGNALS_BINDING}) uniform _Signals {{ vec4 v[{}]; }} _signals;\n",
        MAX_SIGNALS / 4
    );
    let mut aliases = String::new();
    for (name, slot) in signals {
        // The declaration hydra wrote is replaced by a slot in the block.
        shader = shader.replace(&format!("uniform float {name};"), "");
        aliases.push_str(&format!(
            "#define {name} _signals.v[{}][{}]\n",
            slot / 4,
            slot % 4
        ));
    }
    // Only `main` reads a signal. An alias above the helpers would also
    // rewrite a helper local with the same name, such as `b0` in `_noise`.
    let shader = match shader.find("void main ()") {
        Some(main) => format!("{}\n{aliases}{}", &shader[..main], &shader[main..]),
        None => shader.replace("#version 450\n", &format!("#version 450\n{aliases}")),
    };
    shader.replace("#version 450\n", &format!("#version 450\n{block}"))
}

/// Rewrite one composed shader for the native renderer.
pub fn uplift(frag: &str) -> String {
    let mut body = frag.to_owned();

    // These become one uniform block, so the individual declarations go.
    for declaration in [
        "uniform float time;",
        "uniform vec2 resolution;",
        "varying vec2 uv;",
    ] {
        body = body.replace(declaration, "");
    }

    // A sampler in GLSL ES is one object. Vulkan splits it, and naga refuses
    // the combined form outright - "variable qualifier not implemented" - so
    // every `sampler2D` becomes a `texture2D` and a `sampler` bound beside it.
    //
    // Three places have to agree: the declarations, the one function in
    // hydra's table that takes a sampler as a parameter (`src`), and the call
    // sites that pass it a name.
    let mut samplers = String::new();
    for (slot, name) in TEXTURES.iter().enumerate() {
        let binding = GLOBALS_BINDING + 1 + slot as u32 * 2;
        body = body.replace(&format!("uniform sampler2D {name};"), "");
        samplers.push_str(&format!(
            "layout(set=0, binding={binding}) uniform texture2D _t_{name};\n\
             layout(set=0, binding={}) uniform sampler _s_{name};\n",
            binding + 1
        ));
        // `src(st, s0)` passes the name; it now passes both halves.
        body = body.replace(&format!(", {name})"), &format!(", _t_{name}, _s_{name})"));
    }

    // Put the coordinate system back the way hydra wrote it for.
    //
    // WebGL measures `gl_FragCoord.y` from the bottom of the frame. Vulkan
    // measures it from the top and has no option for the other. Without this
    // replacement `st.y` is inverted inside the shader, so rotate, kaleid and
    // every other angular transform turn the wrong way. A flip on readback
    // cannot correct that.
    //
    //     hydra st.y       Vulkan gl_FragCoord.y     texture v
    //   1 +--------+       0 +--------+            0 +--------+  top row
    //     |        |         |        |              |        |
    //   0 +--------+       h +--------+            1 +--------+  bottom row
    //
    // With h = resolution.y: st.y = (h - gl_FragCoord.y) / h, and `_sample`
    // below reads a texture at v = 1 - st.y.
    body = body.replace(
        "vec2 st = gl_FragCoord.xy/resolution.xy;",
        "vec2 st = vec2(gl_FragCoord.x, resolution.y - gl_FragCoord.y)/resolution.xy;",
    );

    body = body.replace("gl_FragColor", "_frag_out");
    body = body.replace("texture2D(", "texture(");

    // `vec4 src(vec2 _st, sampler2D tex)` and its body, which is the only
    // place hydra names a sampler parameter.
    //
    // Both fetches go through `_sample`, which flips the coordinate's y. The
    // `st` above is back on hydra's bottom-up axis, but a texture's v axis
    // runs the other way: v=0 is the first row uploaded, and for a render
    // target that row is `gl_FragCoord.y == 0`, which Vulkan puts at the top.
    // A fetch with `st` unaltered reads every picture upside down: `src(o0)`
    // inverts the frame it copies, and a camera picture is upside down.
    // Generators do not sample, so they look correct either way.
    body = body.replace("sampler2D tex)", "texture2D _t_tex, sampler _s_tex)");
    body = body.replace("texture(tex,", "_sample(_t_tex, _s_tex,");
    // `prev()` reads a global rather than a parameter.
    body = body.replace(
        "texture(prevBuffer,",
        "_sample(_t_prevBuffer, _s_prevBuffer,",
    );

    format!(
        "#version 450\n\
         layout(set=0, binding={GLOBALS_BINDING}) uniform _Globals {{\n\
         \x20   vec2 resolution;\n\
         \x20   float time;\n\
         \x20   vec4 audio0;\n\
         \x20   vec4 audio1;\n\
         \x20   vec4 audio2;\n\
         \x20   vec4 audio3;\n\
         }} _globals;\n\
         #define time _globals.time\n\
         #define resolution _globals.resolution\n\
         #define _audio _globals.audio0\n\
         #define _audio1 _globals.audio1\n\
         #define _audio2 _globals.audio2\n\
         #define _audio3 _globals.audio3\n\
         {samplers}\
         layout(location=0) in vec2 uv;\n\
         layout(location=0) out vec4 _frag_out;\n\
         vec4 _sample(texture2D _t, sampler _s, vec2 _uv) {{\n\
         \x20   return texture(sampler2D(_t, _s), vec2(_uv.x, 1.0 - _uv.y));\n\
         }}\n\
         {body}"
    )
}

/// Hydra's own four-up compositor, uplifted the same way everything else is.
///
/// The body between `void main` and the closing brace is `renderAll`'s from
/// `hydra-synth.js`, character for character: the quadrant arithmetic is
/// fiddly enough that reproducing it from the picture would be a way to get it
/// subtly wrong. `uv` here is the vertex stage's, so this one really does read
/// it, unlike every composed sketch.
pub fn render_all_shader() -> String {
    let mut samplers = String::new();
    for (slot, name) in TEXTURES.iter().enumerate() {
        if !name.starts_with('o') {
            continue;
        }
        let binding = GLOBALS_BINDING + 1 + slot as u32 * 2;
        samplers.push_str(&format!(
            "layout(set=0, binding={binding}) uniform texture2D _t_{name};\n\
             layout(set=0, binding={}) uniform sampler _s_{name};\n",
            binding + 1
        ));
    }
    format!(
        "#version 450\n\
         {samplers}\
         layout(location=0) in vec2 uv;\n\
         layout(location=0) out vec4 _frag_out;\n\
         void main () {{\n\
         \x20 vec2 st = vec2(1.0 - uv.x, uv.y);\n\
         \x20 st*= vec2(2);\n\
         \x20 vec2 q = floor(st).xy*(vec2(2.0, 1.0));\n\
         \x20 int quad = int(q.x) + int(q.y);\n\
         \x20 st.x += step(1., mod(st.y,2.0));\n\
         \x20 st.y += step(1., mod(st.x,2.0));\n\
         \x20 st = fract(st);\n\
         \x20 if(quad==0){{\n\
         \x20   _frag_out = texture(sampler2D(_t_o0, _s_o0), st);\n\
         \x20 }} else if(quad==1){{\n\
         \x20   _frag_out = texture(sampler2D(_t_o1, _s_o1), st);\n\
         \x20 }} else if (quad==2){{\n\
         \x20   _frag_out = texture(sampler2D(_t_o2, _s_o2), st);\n\
         \x20 }} else {{\n\
         \x20   _frag_out = texture(sampler2D(_t_o3, _s_o3), st);\n\
         \x20 }}\n\
         }}\n"
    )
}

/// The vertex half. One triangle covering the frame; hydra's fragment shaders
/// take their coordinates from `gl_FragCoord`, so this only has to exist.
pub const VERTEX: &str = r#"#version 450
layout(location=0) out vec2 uv;
void main() {
    vec2 p = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::naga;

    /// What `create_shader_module` does with a composed sketch, minus the
    /// GPU: naga's GLSL front end, then its validator.
    fn naga_accepts(shader: &str) -> Result<(), String> {
        let module = naga::front::glsl::Frontend::default()
            .parse(
                &naga::front::glsl::Options::from(naga::ShaderStage::Fragment),
                shader,
            )
            .map_err(|error| format!("{error:?}"))?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&module)
        .map_err(|error| format!("{error:?}"))?;
        Ok(())
    }

    #[test]
    fn a_signal_used_more_times_than_there_are_slots_compiles() {
        for slot in [0, MAX_SIGNALS as u32 - 1] {
            let call = crate::HydraCall {
                method: "rotate".into(),
                args: vec![crate::HydraNode::Signal { slot }],
            };
            let node = crate::HydraNode::Chain {
                head: "osc".into(),
                args: Vec::new(),
                calls: vec![call; MAX_SIGNALS + 1],
            };
            let composed =
                crate::glsl::compose_full(&node, "highp").expect("the composer admits it");
            let shader = uplift_with_signal_slots(&composed.shader, &composed.signals);
            naga_accepts(&shader).unwrap_or_else(|error| panic!("slot {slot}: {error}"));
        }
    }

    /// `solid` and `color` name inputs `b` and `a`, so a first signal there is
    /// `b0` or `a0`: the names of two locals in the `_noise` helper.
    #[test]
    fn a_signal_named_like_a_helper_local_compiles() {
        let number = |v: f64| crate::HydraNode::Number { v };
        let signal = || crate::HydraNode::Signal { slot: 0 };
        for args in [
            vec![number(0.0), number(0.0), signal()],
            vec![number(0.0), number(0.0), number(0.0), signal()],
        ] {
            let node = crate::HydraNode::Chain {
                head: "solid".into(),
                args,
                calls: Vec::new(),
            };
            let composed =
                crate::glsl::compose_full(&node, "highp").expect("the composer admits it");
            let shader = uplift_with_signal_slots(&composed.shader, &composed.signals);
            naga_accepts(&shader).unwrap_or_else(|error| panic!("{:?}: {error}", composed.signals));
        }
    }

    #[test]
    fn the_largest_float_literal_the_composer_admits_compiles_and_one_past_it_does_not() {
        // The composer's bound is the f32 range because naga reads a GLSL
        // float literal as an f32. The largest literals the composer admits,
        // from a chain and from a callback, must compile. The same shader
        // one step past the bound must not.
        for (sketch, admitted, refused) in [
            (
                "osc(3.4028235e38)",
                "340282350000000000000000000000000000000.",
                "340282357000000000000000000000000000000.",
            ),
            (
                "osc(-3e38)",
                "-300000000000000000000000000000000000000.",
                "-1000000000000000000000000000000000000000.",
            ),
            (
                "osc(() => 340282350000000000000000000000000000000)",
                "340282350000000000000000000000000000000.0",
                "340282357000000000000000000000000000000.0",
            ),
        ] {
            let node = crate::glsl::parse_chain(sketch).expect("the sketch parses");
            let composed =
                crate::glsl::compose_full(&node, "highp").expect("the composer admits it");
            let shader = uplift_with_signal_slots(&composed.shader, &composed.signals);
            assert!(
                shader.contains(admitted),
                "{sketch} spells {admitted}: {shader}"
            );
            naga_accepts(&shader).unwrap_or_else(|error| panic!("{sketch}: {error}"));

            // naga's front end evaluates the literal as it parses and
            // reports `LiteralError::Infinity`, whose message this is.
            let past = shader.replace(admitted, refused);
            let error = naga_accepts(&past).expect_err("naga reads it as an infinite f32");
            assert!(
                error.contains("Float literal is infinite"),
                "{sketch} past the bound: {error}"
            );
        }
    }
}
