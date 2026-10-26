//! Compare Rust-composed GLSL with the `hydra-synth` output, character for
//! character. `glsl-truth.json` contains the reference chains. The capture
//! harness is not part of this repository. Capture the file again when the
//! `hydra-synth` version changes.

use rustel_hydra::glsl::compose;
use rustel_hydra::{HydraCall, HydraNode};

/// Build the recorded shape of `head(args).m1(args).m2(args)`.
fn chain(head: &str, args: Vec<HydraNode>, calls: Vec<(&str, Vec<HydraNode>)>) -> HydraNode {
    HydraNode::Chain {
        head: head.into(),
        args,
        calls: calls
            .into_iter()
            .map(|(method, args)| HydraCall {
                method: method.into(),
                args,
            })
            .collect(),
    }
}

fn n(v: f64) -> HydraNode {
    HydraNode::Number { v }
}

fn truth() -> serde_json::Map<String, serde_json::Value> {
    let raw = include_str!("glsl-truth.json");
    serde_json::from_str::<serde_json::Value>(raw)
        .expect("the captured shaders parse")
        .as_object()
        .expect("an object of chain -> shader")
        .clone()
}

/// Report the first line that differs, because a 4,000-character diff is
/// unreadable and the first divergence is always the one that matters.
fn assert_same(label: &str, ours: &str, theirs: &str) {
    if ours == theirs {
        return;
    }
    let mut mine = ours.lines();
    let mut yours = theirs.lines();
    let mut line = 0;
    loop {
        line += 1;
        match (mine.next(), yours.next()) {
            (None, None) => break,
            (a, b) if a == b => continue,
            (a, b) => panic!(
                "{label}: line {line} differs\n  ours:   {a:?}\n  hydra:  {b:?}\n\
                 (ours {} chars, hydra {} chars)",
                ours.len(),
                theirs.len()
            ),
        }
    }
    panic!("{label}: same lines, different text");
}

/// Every transform hydra ships, exercised once with its own defaults.
///
/// Mechanical by design: a `src` alone, a `coord` or `color` after an `osc`,
/// and a `combine` or `combineCoord` fed a second chain. It is the breadth
/// check - the hand-written cases below are the depth one.
#[test]
fn every_transform_in_the_table_composes_the_same() {
    let truth = truth();
    let cases: Vec<(&str, HydraNode)> = vec![
        ("gradient()", chain("gradient", vec![], vec![])),
        ("noise()", chain("noise", vec![], vec![])),
        ("osc()", chain("osc", vec![], vec![])),
        (
            "osc(10,0.1,0.3).a()",
            chain("osc", vec![n(10.0), n(0.1), n(0.3)], vec![("a", vec![])]),
        ),
        (
            "osc(10,0.1,0.3).add(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("add", vec![chain("noise", vec![n(3.0)], vec![])])],
            ),
        ),
        (
            "osc(10,0.1,0.3).b()",
            chain("osc", vec![n(10.0), n(0.1), n(0.3)], vec![("b", vec![])]),
        ),
        (
            "osc(10,0.1,0.3).blend(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("blend", vec![chain("noise", vec![n(3.0)], vec![])])],
            ),
        ),
        (
            "osc(10,0.1,0.3).brightness()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("brightness", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).color()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("color", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).colorama()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("colorama", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).contrast()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("contrast", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).diff(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("diff", vec![chain("noise", vec![n(3.0)], vec![])])],
            ),
        ),
        (
            "osc(10,0.1,0.3).g()",
            chain("osc", vec![n(10.0), n(0.1), n(0.3)], vec![("g", vec![])]),
        ),
        (
            "osc(10,0.1,0.3).hue()",
            chain("osc", vec![n(10.0), n(0.1), n(0.3)], vec![("hue", vec![])]),
        ),
        (
            "osc(10,0.1,0.3).invert()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("invert", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).kaleid()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("kaleid", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).layer(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("layer", vec![chain("noise", vec![n(3.0)], vec![])])],
            ),
        ),
        (
            "osc(10,0.1,0.3).luma()",
            chain("osc", vec![n(10.0), n(0.1), n(0.3)], vec![("luma", vec![])]),
        ),
        (
            "osc(10,0.1,0.3).mask(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("mask", vec![chain("noise", vec![n(3.0)], vec![])])],
            ),
        ),
        (
            "osc(10,0.1,0.3).modulate(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("modulate", vec![chain("noise", vec![n(3.0)], vec![])])],
            ),
        ),
        (
            "osc(10,0.1,0.3).modulateHue(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("modulateHue", vec![chain("noise", vec![n(3.0)], vec![])])],
            ),
        ),
        (
            "osc(10,0.1,0.3).modulateKaleid(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("modulateKaleid", vec![chain("noise", vec![n(3.0)], vec![])])],
            ),
        ),
        (
            "osc(10,0.1,0.3).modulatePixelate(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![(
                    "modulatePixelate",
                    vec![chain("noise", vec![n(3.0)], vec![])],
                )],
            ),
        ),
        (
            "osc(10,0.1,0.3).modulateRepeat(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("modulateRepeat", vec![chain("noise", vec![n(3.0)], vec![])])],
            ),
        ),
        (
            "osc(10,0.1,0.3).modulateRepeatX(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![(
                    "modulateRepeatX",
                    vec![chain("noise", vec![n(3.0)], vec![])],
                )],
            ),
        ),
        (
            "osc(10,0.1,0.3).modulateRepeatY(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![(
                    "modulateRepeatY",
                    vec![chain("noise", vec![n(3.0)], vec![])],
                )],
            ),
        ),
        (
            "osc(10,0.1,0.3).modulateRotate(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("modulateRotate", vec![chain("noise", vec![n(3.0)], vec![])])],
            ),
        ),
        (
            "osc(10,0.1,0.3).modulateScale(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("modulateScale", vec![chain("noise", vec![n(3.0)], vec![])])],
            ),
        ),
        (
            "osc(10,0.1,0.3).modulateScrollX(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![(
                    "modulateScrollX",
                    vec![chain("noise", vec![n(3.0)], vec![])],
                )],
            ),
        ),
        (
            "osc(10,0.1,0.3).modulateScrollY(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![(
                    "modulateScrollY",
                    vec![chain("noise", vec![n(3.0)], vec![])],
                )],
            ),
        ),
        (
            "osc(10,0.1,0.3).mult(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("mult", vec![chain("noise", vec![n(3.0)], vec![])])],
            ),
        ),
        (
            "osc(10,0.1,0.3).pixelate()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("pixelate", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).posterize()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("posterize", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).r()",
            chain("osc", vec![n(10.0), n(0.1), n(0.3)], vec![("r", vec![])]),
        ),
        (
            "osc(10,0.1,0.3).repeat()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("repeat", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).repeatX()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("repeatX", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).repeatY()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("repeatY", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).rotate()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("rotate", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).saturate()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("saturate", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).scale()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("scale", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).scroll()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("scroll", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).scrollX()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("scrollX", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).scrollY()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("scrollY", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).shift()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("shift", vec![])],
            ),
        ),
        (
            "osc(10,0.1,0.3).sub(noise(3))",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("sub", vec![chain("noise", vec![n(3.0)], vec![])])],
            ),
        ),
        (
            "osc(10,0.1,0.3).sum()",
            chain("osc", vec![n(10.0), n(0.1), n(0.3)], vec![("sum", vec![])]),
        ),
        (
            "osc(10,0.1,0.3).thresh()",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.3)],
                vec![("thresh", vec![])],
            ),
        ),
        ("shape()", chain("shape", vec![], vec![])),
        ("solid()", chain("solid", vec![], vec![])),
        ("voronoi()", chain("voronoi", vec![], vec![])),
    ];
    assert!(cases.len() >= 50, "the sweep should cover the whole table");
    for (label, node) in cases {
        let theirs = truth
            .get(label)
            .and_then(serde_json::Value::as_str)
            .unwrap_or_else(|| panic!("no captured shader for `{label}`"));
        let ours = compose(&node, "highp").unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_same(label, &ours, theirs);
    }
}

/// `H(pattern)` compiles to the same shader a `() => ...` argument does.
/// hydra turns a function argument into `uniform float <input><n>`, where n
/// counts every uniform declared so far. A frame fills these uniform names.
#[test]
fn a_signal_compiles_to_the_uniform_a_function_argument_would() {
    use rustel_hydra::glsl::compose_full;
    let truth = truth();
    let signal = |slot| HydraNode::Signal { slot };

    let one = chain("osc", vec![signal(0), n(0.1), n(0.8)], vec![]);
    let composed = compose_full(&one, "highp").expect("composes");
    assert_same(
        "osc(() => 1, 0.1, 0.8)",
        &composed.shader,
        truth["osc(() => 1, 0.1, 0.8)"].as_str().unwrap(),
    );
    assert_eq!(composed.signals, vec![("frequency0".to_owned(), 0)]);

    // Two signals, and the counter runs across the whole chain.
    let two = chain("shape", vec![signal(3)], vec![("rotate", vec![signal(1)])]);
    let composed = compose_full(&two, "highp").expect("composes");
    assert_same(
        "shape(() => 3).rotate(() => 0.5)",
        &composed.shader,
        truth["shape(() => 3).rotate(() => 0.5)"].as_str().unwrap(),
    );
    assert_eq!(
        composed.signals,
        vec![("sides0".to_owned(), 3), ("angle1".to_owned(), 1)],
        "named in declaration order, each carrying the slot that fills it"
    );
}

#[test]
fn the_composer_emits_what_hydra_emits() {
    let truth = truth();
    let cases: Vec<(&str, HydraNode)> = vec![
        (
            "osc(10, 0.1, 0.8)",
            chain("osc", vec![n(10.0), n(0.1), n(0.8)], vec![]),
        ),
        (
            "noise(10, 0.1)",
            chain("noise", vec![n(10.0), n(0.1)], vec![]),
        ),
        (
            "osc(10, 0.1, 0.8).kaleid(5)",
            chain(
                "osc",
                vec![n(10.0), n(0.1), n(0.8)],
                vec![("kaleid", vec![n(5.0)])],
            ),
        ),
        (
            "osc(60).rotate(0.5).color(1, 0.5, 0.2)",
            chain(
                "osc",
                vec![n(60.0)],
                vec![
                    ("rotate", vec![n(0.5)]),
                    ("color", vec![n(1.0), n(0.5), n(0.2)]),
                ],
            ),
        ),
        (
            "shape(4, 0.3, 0.01).mult(osc(20))",
            chain(
                "shape",
                vec![n(4.0), n(0.3), n(0.01)],
                vec![("mult", vec![chain("osc", vec![n(20.0)], vec![])])],
            ),
        ),
        (
            "osc(10).modulate(noise(3), 0.5)",
            chain(
                "osc",
                vec![n(10.0)],
                vec![(
                    "modulate",
                    vec![chain("noise", vec![n(3.0)], vec![]), n(0.5)],
                )],
            ),
        ),
        (
            "gradient(1).brightness(0.2).contrast(1.5).invert(1)",
            chain(
                "gradient",
                vec![n(1.0)],
                vec![
                    ("brightness", vec![n(0.2)]),
                    ("contrast", vec![n(1.5)]),
                    ("invert", vec![n(1.0)]),
                ],
            ),
        ),
    ];

    for (label, node) in cases {
        let theirs = truth
            .get(label)
            .and_then(serde_json::Value::as_str)
            .unwrap_or_else(|| panic!("no captured shader for `{label}`"));
        let ours = compose(&node, "highp").unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_same(label, &ours, theirs);
    }
}

/// A callback such as `osc(() => Math.sin(time))` becomes GLSL, and what
/// cannot is refused by name. Upstream evaluates the callback in JavaScript
/// every frame. There is no JavaScript here, so the expression is compiled
/// into the shader, which already has `time` and the audio bands. The value
/// is exact and not sampled once a frame.
#[test]
fn a_callback_is_compiled_into_the_shader() {
    use rustel_hydra::glsl::compose;

    let animated = chain(
        "osc",
        vec![HydraNode::Source {
            src: "() => Math.sin(time)".into(),
        }],
        vec![],
    );
    let shader = compose(&animated, "highp").expect("a time expression compiles");
    assert!(
        shader.contains("osc(st, sin(time), 0.1, 0.)"),
        "the expression is inlined, not passed as a uniform"
    );

    let reactive = chain(
        "osc",
        vec![HydraNode::Source {
            src: "() => 0.8 + a.fft[0] * 1.4".into(),
        }],
        vec![],
    );
    let shader = compose(&reactive, "highp").expect("an audio expression compiles");
    assert!(
        shader.contains("_audio[0]"),
        "the band is read in the shader"
    );

    // What it cannot compile it names, along with what does work.
    let refused = chain(
        "osc",
        vec![HydraNode::Source {
            src: "() => Math.random()".into(),
        }],
        vec![],
    );
    let said = compose(&refused, "highp")
        .expect_err("an uncompilable callback is refused")
        .to_string();
    assert!(said.contains("osc"), "names the transform: {said}");
    assert!(said.contains("H(pattern)"), "names the way out: {said}");

    let with_string = chain("osc", vec![HydraNode::Text { v: "bd*4".into() }], vec![]);
    let said = compose(&with_string, "highp")
        .expect_err("a string is refused")
        .to_string();
    assert!(said.contains("string"), "{said}");
}
