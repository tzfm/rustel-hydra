//! Running the composed shader on the GPU.
//!
//! No window, no surface, no swapchain: offscreen rendering is the ordinary
//! case for Vulkan, Metal and DX12, and a machine with no GPU falls through to
//! a software rasteriser rather than failing. On lavapipe with no GPU present,
//! a 640×360 frame draws and reads back in 4.7 ms; the terminal reads twenty a
//! second.

mod glyphs;
mod renderer;
mod uplift;

pub(crate) use glyphs::validate_tui_frame;
pub use glyphs::{GLYPH_HEIGHT, GLYPH_WIDTH, rasterize};
pub(crate) use renderer::OUTPUTS;
pub use renderer::{NativeError, NativeFootprint, NativeRenderer};
pub use uplift::{
    GLOBALS_BINDING, MAX_SIGNALS, SIGNALS_BINDING, TEXTURES, VERTEX, render_all_shader, uplift,
    uplift_with_signal_slots, uplift_with_signals,
};
