# rustel-hydra

Hydra video synth in native Rust. GLSL shaders composed in code, rendered
offscreen through wgpu, pixels out. No browser, no window.

Vulkan, Metal and DX12 render offscreen natively. Lavapipe covers a
machine with no GPU at all. A host that has no adapter and no software
rasteriser cannot render, and the renderer reports that.

## Fidelity, checked

The GLSL function bodies in `glsl::table` are generated from
[hydra-synth](https://github.com/hydra-synth/hydra-synth)'s own source,
not retyped. `tests/glsl_parity.rs` compares this crate's composed shader
with hydra-synth's, character for character, across its whole function
table plus multi-step chains, nested sources and signals.
`tests/pixel_parity.rs` compares rendered frames against committed
reference images. When the hydra-synth version changes, capture
`tests/glsl-truth.json` again with a harness against that version.

## As a library

A sketch is recorded, not executed. `HydraProgram` is that recording, and
it crosses a thread boundary as data. `HydraHost` owns one worker thread
that renders it. The composer needs no GPU, so a host can build the
shader first:

```toml
[dependencies]
rustel-hydra = { git = "https://github.com/tzfm/rustel-hydra", tag = "v0.2.2" }
```

```rust
use rustel_hydra::glsl::compose;
use rustel_hydra::{HydraCall, HydraNode};

// osc(10, 0.1).kaleid(4)
let node = HydraNode::Chain {
    head: "osc".to_owned(),
    args: vec![HydraNode::Number { v: 10.0 }, HydraNode::Number { v: 0.1 }],
    calls: vec![HydraCall {
        method: "kaleid".to_owned(),
        args: vec![HydraNode::Number { v: 4.0 }],
    }],
};
let shader = compose(&node, "highp").unwrap();
```

`native::NativeRenderer` runs the composed shader offscreen and hands
back RGBA pixels.

## In rustel

Rustel draws a score's Hydra sketch behind the code in the studio. See
the [studio guide](https://github.com/tzfm/rustel/blob/main/docs/hydra.md).

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
The GLSL table is generated from hydra-synth, which is AGPL-3.0. Its
licence travels with this tree as
[LICENSE-hydra-synth](LICENSE-hydra-synth). This project is independent
and is not affiliated with hydra-synth.
