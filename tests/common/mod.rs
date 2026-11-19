//! Shared scaffolding for the renderer suites.

use std::sync::{Mutex, MutexGuard, PoisonError};

use rustel_hydra::{
    HYDRA_HEIGHT, HYDRA_PROGRAM_VERSION, HYDRA_WIDTH, HydraCall, HydraNode, HydraOptions,
    HydraProgram, HydraStatement,
};

/// Rendering requires a GPU adapter or software rasteriser, but no screen.
/// If neither is available, report the skipped rendering check.
#[allow(dead_code)]
pub fn renderer_is_available() -> bool {
    rustel_hydra::native::NativeRenderer::new(8, 8).is_ok()
}

static GPU: Mutex<()> = Mutex::new(());

/// Hold this while a test owns a renderer. The Vulkan loader on some Linux
/// hosts crashes when renderers in parallel threads name GPU objects, as
/// debug builds do. The host runs one renderer at a time, so tests do too.
#[allow(dead_code)]
pub fn gpu() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A bright moving grating at the requested render size. A black frame
/// indicates a rendering failure, such as a shader, buffer-swap or readback error.
#[allow(dead_code)]
pub fn sketch(width: u32, height: u32) -> HydraProgram {
    HydraProgram {
        version: HYDRA_PROGRAM_VERSION,
        options: HydraOptions {
            width,
            height,
            ..HydraOptions::default()
        },
        signals: 0,
        raw: None,
        statements: vec![HydraStatement::Evaluate {
            node: HydraNode::Chain {
                head: "osc".into(),
                args: vec![
                    HydraNode::Number { v: 20.0 },
                    HydraNode::Number { v: 0.1 },
                    HydraNode::Number { v: 1.2 },
                ],
                calls: vec![HydraCall {
                    method: "out".into(),
                    args: Vec::new(),
                }],
            },
        }],
    }
}

/// The same sketch at the size a score gets by default.
#[allow(dead_code)]
pub fn default_sketch() -> HydraProgram {
    sketch(HYDRA_WIDTH, HYDRA_HEIGHT)
}
