//! Turning a chain into a fragment shader.
//!
//! Hydra builds its shader by splicing GLSL bodies out of a table: [`table`]
//! holds those bodies verbatim and [`compose`] folds a chain into a shader the
//! way the library does.
//!
//! A score's chain arrives as structured nodes - see [`crate::HydraProgram`] -
//! so nothing here reads JavaScript. Snippet text, which nothing has evaluated,
//! goes through [`parse`] first.

pub mod compose;
pub mod expr;
pub mod parse;
pub mod table;

pub use compose::{
    ComposeError, Composed, compiles_exactly_these_easings, compose, compose_full,
    explicitly_samples, output_of, sampled,
};
pub use expr::{AUDIO_UNIFORM, ExprError};
pub use parse::{ParseError, parse_chain};
pub use table::{FUNCTIONS, Function, Input, Kind, UTILITY, function};
