//! The terminal, as a texture a sketch can sample.
//!
//! `feedStrudel` puts the editor into `s0`, and what a sketch samples has to
//! read as code: glyph shapes, not one flat colour a cell. At the size this
//! runs at - three by seven pixels a cell - rasterised glyphs still carry 2.4
//! times the variation of flat cells.
//!
//! So there are glyphs, but no font and no font library. [`ATLAS`] is printable
//! ASCII rasterised once, at development time, into an 8×16 coverage bitmap -
//! twelve kilobytes, which is a twenty-eighth of the smallest monospace font
//! that would otherwise have to be committed. A cell outside that range is
//! drawn as its background, which is what a box-drawing character mostly looks
//! like at this size anyway.

use crate::{HydraTuiCell, HydraTuiFrame};

/// One glyph's cell in the atlas.
pub const GLYPH_WIDTH: usize = 8;
pub const GLYPH_HEIGHT: usize = 16;

/// Coverage for `0x20..0x7f`, one byte a sample, row-major within each glyph.
static ATLAS: &[u8] = include_bytes!("glyphs.bin");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RasterLayout {
    width: usize,
    height: usize,
    bytes: usize,
    scale: usize,
}

/// Validate a terminal grid without allocating or indexing its cell buffer.
/// The raster is another Hydra source texture, so it observes the same edge
/// and area ceilings as an output frame.
pub(crate) fn validate_tui_frame(
    frame: &HydraTuiFrame,
    scale: usize,
) -> Result<RasterLayout, String> {
    let scale = scale.max(1);
    if frame.cols == 0 || frame.rows == 0 {
        return Err(format!(
            "Hydra terminal frame dimensions must be non-zero, got {}x{}",
            frame.cols, frame.rows
        ));
    }
    let width = usize::from(frame.cols)
        .checked_mul(scale)
        .ok_or_else(|| "Hydra terminal raster width overflows host memory".to_owned())?;
    let height = usize::from(frame.rows)
        .checked_mul(scale)
        .ok_or_else(|| "Hydra terminal raster height overflows host memory".to_owned())?;
    let width_u64 = u64::try_from(width)
        .map_err(|_| "Hydra terminal raster width does not fit u64".to_owned())?;
    let height_u64 = u64::try_from(height)
        .map_err(|_| "Hydra terminal raster height does not fit u64".to_owned())?;
    let pixels = width_u64
        .checked_mul(height_u64)
        .ok_or_else(|| "Hydra terminal raster pixel count overflows".to_owned())?;
    if width_u64 > u64::from(crate::HYDRA_MAX_OUTPUT_EDGE)
        || height_u64 > u64::from(crate::HYDRA_MAX_OUTPUT_EDGE)
        || pixels > crate::HYDRA_MAX_OUTPUT_PIXELS
    {
        return Err(format!(
            "Hydra terminal raster {width}x{height} exceeds the {}-pixel edge / {}-pixel area limit",
            crate::HYDRA_MAX_OUTPUT_EDGE,
            crate::HYDRA_MAX_OUTPUT_PIXELS
        ));
    }
    let expected_cells = usize::from(frame.cols)
        .checked_mul(usize::from(frame.rows))
        .ok_or_else(|| "Hydra terminal cell count overflows host memory".to_owned())?;
    if frame.cells.len() != expected_cells {
        return Err(format!(
            "Hydra terminal frame {}x{} requires exactly {expected_cells} cells, got {}",
            frame.cols,
            frame.rows,
            frame.cells.len()
        ));
    }
    let bytes = pixels
        .checked_mul(4)
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| "Hydra terminal RGBA byte count overflows host memory".to_owned())?;
    Ok(RasterLayout {
        width,
        height,
        bytes,
        scale,
    })
}

/// How much of a cell one glyph covers at a point, 0..=255.
fn coverage(code: u32, x: usize, y: usize) -> u8 {
    let Some(index) = code.checked_sub(0x20).filter(|c| *c < 95) else {
        return 0;
    };
    let at = index as usize * GLYPH_WIDTH * GLYPH_HEIGHT + y * GLYPH_WIDTH + x;
    ATLAS.get(at).copied().unwrap_or(0)
}

/// Draw a terminal frame as RGBA, `scale` pixels a cell each way.
///
/// Rows come out top-first, which is the order [`crate::native::NativeRenderer::set_source`]
/// wants. One is a useful scale in its own right: it is the flat-colour
/// version, and the glyph collapses to its mean.
pub fn rasterize(frame: &HydraTuiFrame, scale: usize) -> Result<(u32, u32, Vec<u8>), String> {
    let layout = validate_tui_frame(frame, scale)?;
    let mut pixels = Vec::new();
    pixels.try_reserve_exact(layout.bytes).map_err(|error| {
        format!(
            "could not reserve {} bytes for the Hydra terminal raster: {error}",
            layout.bytes
        )
    })?;
    pixels.resize(layout.bytes, 0);
    for row in 0..usize::from(frame.rows) {
        for column in 0..usize::from(frame.cols) {
            let cell = frame.cells[row * usize::from(frame.cols) + column];
            paint(&mut pixels, layout.width, column, row, layout.scale, cell);
        }
    }
    let width = u32::try_from(layout.width)
        .map_err(|_| "Hydra terminal raster width does not fit u32".to_owned())?;
    let height = u32::try_from(layout.height)
        .map_err(|_| "Hydra terminal raster height does not fit u32".to_owned())?;
    Ok((width, height, pixels))
}

/// One cell: its background, with its glyph blended over in its foreground.
fn paint(
    pixels: &mut [u8],
    width: usize,
    column: usize,
    row: usize,
    scale: usize,
    cell: HydraTuiCell,
) {
    for y in 0..scale {
        // Sample the glyph at the middle of each output pixel rather than its
        // corner: at three pixels a cell the difference is a stroke landing or
        // being missed entirely.
        let gy = (y * 2 + 1) * GLYPH_HEIGHT / (scale * 2);
        for x in 0..scale {
            let gx = (x * 2 + 1) * GLYPH_WIDTH / (scale * 2);
            let ink = u32::from(coverage(
                cell.code,
                gx.min(GLYPH_WIDTH - 1),
                gy.min(GLYPH_HEIGHT - 1),
            ));
            let at = ((row * scale + y) * width + column * scale + x) * 4;
            for channel in 0..3 {
                let back = u32::from(cell.bg[channel]);
                let fore = u32::from(cell.fg[channel]);
                pixels[at + channel] = ((back * (255 - ink) + fore * ink) / 255) as u8;
            }
            pixels[at + 3] = 255;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(code: u32) -> HydraTuiFrame {
        HydraTuiFrame {
            cols: 1,
            rows: 1,
            cell_width: 8,
            cell_height: 16,
            cells: vec![HydraTuiCell {
                code,
                fg: [255, 255, 255],
                bg: [0, 0, 0],
                flags: 0,
            }],
        }
    }

    #[test]
    fn a_glyph_puts_ink_where_a_space_does_not() {
        let ink = |code| {
            let (_, _, pixels) = rasterize(&grid(code), GLYPH_WIDTH).unwrap();
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| u32::from(p[0]))
                .sum::<u32>()
        };
        assert_eq!(ink(u32::from(b' ')), 0, "a space is all background");
        assert!(ink(u32::from(b'W')) > 0, "a letter is not");
        // `W` covers more of its cell than `.` does, which is the property a
        // sketch sampling this actually sees.
        assert!(ink(u32::from(b'W')) > ink(u32::from(b'.')) * 3);
    }

    #[test]
    fn a_cell_outside_ascii_is_its_background() {
        let (_, _, pixels) = rasterize(&grid(0x2588), 4).unwrap();
        assert!(
            pixels.as_chunks::<4>().0.iter().all(|p| p[0] == 0),
            "an unrasterised code leaves the background alone"
        );
    }

    #[test]
    fn the_frame_is_the_grid_times_the_scale() {
        let mut frame = grid(u32::from(b'x'));
        frame.cols = 5;
        frame.rows = 3;
        frame.cells = vec![HydraTuiCell::default(); 15];
        let (width, height, pixels) = rasterize(&frame, 3).unwrap();
        assert_eq!((width, height), (15, 9));
        assert_eq!(pixels.len(), 15 * 9 * 4);
    }

    #[test]
    fn malformed_or_hostile_grids_are_refused_before_allocation() {
        let huge = HydraTuiFrame {
            cols: u16::MAX,
            rows: u16::MAX,
            cells: Vec::new(),
            ..HydraTuiFrame::default()
        };
        let error = rasterize(&huge, 4).expect_err("huge raster is bounded");
        assert!(error.contains("exceeds"), "{error}");

        let mut wrong_cells = grid(u32::from(b'x'));
        wrong_cells.cols = 2;
        let error = rasterize(&wrong_cells, 4).expect_err("cell count must be exact");
        assert!(error.contains("exactly 2 cells, got 1"), "{error}");

        let empty = HydraTuiFrame::default();
        assert!(rasterize(&empty, 4).is_err());
        assert!(rasterize(&grid(u32::from(b'x')), usize::MAX).is_err());
    }
}
