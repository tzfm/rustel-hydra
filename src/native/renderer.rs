//! One device, one pipeline per sketch, four outputs.

use std::collections::HashMap;

use crate::glsl::{ComposeError, compose_full, output_of};
use crate::native::uplift::{
    MAX_SIGNALS, SIGNALS_BINDING, VERTEX, render_all_shader, uplift_with_signal_slots,
};
use crate::program::HydraNode;

#[derive(Debug)]
pub enum NativeError {
    NoAdapter(String),
    NoDevice(String),
    Compose(ComposeError),
    /// The shader would not compile. hydra ships one that does not - `sum()`
    /// in 1.4.0 reads `s` where its input is named `scale` - so this is a
    /// refusal a caller can carry on past, not a crash.
    Shader(String),
    /// A source picture that does not match the size it claims.
    Source(String),
    /// Output textures or their readback would exceed a checked geometry or
    /// device allocation limit.
    Geometry(String),
    Readback(String),
}

impl std::fmt::Display for NativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoAdapter(why) => write!(f, "no GPU or software rasteriser available: {why}"),
            Self::NoDevice(why) => write!(f, "could not open a rendering device: {why}"),
            Self::Compose(error) => write!(f, "{error}"),
            Self::Shader(why) => write!(f, "the shader would not compile: {why}"),
            Self::Source(why) => write!(f, "{why}"),
            Self::Geometry(why) => write!(f, "invalid Hydra output geometry: {why}"),
            Self::Readback(why) => write!(f, "reading the frame back failed: {why}"),
        }
    }
}

impl From<ComposeError> for NativeError {
    fn from(error: ComposeError) -> Self {
        Self::Compose(error)
    }
}

/// How many outputs hydra addresses: `o0`-`o3`.
pub const OUTPUTS: usize = 4;

/// How many source textures: `s0`-`s3`.
pub const SOURCES: usize = crate::HYDRA_SOURCE_SLOTS;

/// Four double-buffered outputs plus the double-buffered display compositor.
const OUTPUT_TEXTURE_COPIES: u64 = (OUTPUTS as u64 + 1) * 2;
const MAX_AGGREGATE_OUTPUT_PIXELS: u64 = crate::HYDRA_MAX_OUTPUT_PIXELS * OUTPUT_TEXTURE_COPIES;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OutputLayout {
    width: u32,
    height: u32,
    pixels: u64,
    aggregate_pixels: u64,
    row_bytes: usize,
    padded_row: u32,
    readback_bytes: u64,
    frame_bytes: usize,
}

/// Validate all CPU and GPU arithmetic before creating a texture or buffer.
fn output_layout(
    width: u32,
    height: u32,
    max_texture_dimension: u32,
    max_buffer_size: u64,
) -> Result<OutputLayout, NativeError> {
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| NativeError::Geometry(format!("{width}x{height} pixel count overflows")))?;
    let aggregate_pixels = pixels.checked_mul(OUTPUT_TEXTURE_COPIES).ok_or_else(|| {
        NativeError::Geometry(format!(
            "{width}x{height} across {OUTPUT_TEXTURE_COPIES} render textures overflows"
        ))
    })?;
    let row_bytes_u64 = u64::from(width)
        .checked_mul(4)
        .ok_or_else(|| NativeError::Geometry(format!("{width}-pixel RGBA row overflows")))?;
    let row_alignment = u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let padded_row_u64 = row_bytes_u64
        .checked_add(row_alignment - 1)
        .and_then(|row| row.checked_div(row_alignment))
        .and_then(|rows| rows.checked_mul(row_alignment))
        .ok_or_else(|| NativeError::Geometry(format!("{width}-pixel aligned row overflows")))?;
    let readback_bytes = padded_row_u64
        .checked_mul(u64::from(height))
        .ok_or_else(|| NativeError::Geometry(format!("{width}x{height} readback overflows")))?;
    let frame_bytes_u64 = pixels
        .checked_mul(4)
        .ok_or_else(|| NativeError::Geometry(format!("{width}x{height} RGBA frame overflows")))?;

    if width == 0 || height == 0 {
        return Err(NativeError::Geometry(format!(
            "dimensions must be non-zero, got {width}x{height}"
        )));
    }
    if width > crate::HYDRA_MAX_OUTPUT_EDGE
        || height > crate::HYDRA_MAX_OUTPUT_EDGE
        || pixels > crate::HYDRA_MAX_OUTPUT_PIXELS
        || aggregate_pixels > MAX_AGGREGATE_OUTPUT_PIXELS
    {
        return Err(NativeError::Geometry(format!(
            "{width}x{height} exceeds the {}-pixel edge / {}-pixel frame / {}-pixel aggregate limit",
            crate::HYDRA_MAX_OUTPUT_EDGE,
            crate::HYDRA_MAX_OUTPUT_PIXELS,
            MAX_AGGREGATE_OUTPUT_PIXELS
        )));
    }
    if width > max_texture_dimension || height > max_texture_dimension {
        return Err(NativeError::Geometry(format!(
            "{width}x{height} exceeds this adapter's {max_texture_dimension}-pixel texture dimension limit"
        )));
    }
    if readback_bytes > max_buffer_size {
        return Err(NativeError::Geometry(format!(
            "{width}x{height} needs a {readback_bytes}-byte readback buffer; this device permits {max_buffer_size}"
        )));
    }

    let row_bytes = usize::try_from(row_bytes_u64)
        .map_err(|_| NativeError::Geometry("RGBA row does not fit host memory".into()))?;
    let padded_row = u32::try_from(padded_row_u64)
        .map_err(|_| NativeError::Geometry("aligned readback row does not fit wgpu".into()))?;
    let frame_bytes = usize::try_from(frame_bytes_u64)
        .map_err(|_| NativeError::Geometry("RGBA frame does not fit host memory".into()))?;
    Ok(OutputLayout {
        width,
        height,
        pixels,
        aggregate_pixels,
        row_bytes,
        padded_row,
        readback_bytes,
        frame_bytes,
    })
}

fn reserve_frame(width: u32, height: u32, bytes: usize) -> Result<Vec<u8>, NativeError> {
    let mut pixels = Vec::new();
    pixels.try_reserve_exact(bytes).map_err(|error| {
        NativeError::Readback(format!(
            "could not reserve {bytes} bytes for a {width}x{height} RGBA frame: {error}"
        ))
    })?;
    Ok(pixels)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SourceLayout {
    bytes: usize,
    bytes_per_row: u32,
}

/// Validate score- or service-provided image metadata before it reaches wgpu.
/// In particular, all arithmetic is checked and the byte slice must describe
/// exactly one tightly packed RGBA image; accepting a prefix or suffix makes
/// it too easy for a stale width/height pair to upload the wrong frame.
fn source_layout(
    index: usize,
    width: u32,
    height: u32,
    bytes: usize,
    max_dimension: u32,
) -> Result<SourceLayout, NativeError> {
    if index >= SOURCES {
        return Err(NativeError::Source(format!(
            "hydra source index {index} is outside s0..s{}",
            SOURCES - 1
        )));
    }
    if width == 0 || height == 0 {
        return Err(NativeError::Source(format!(
            "s{index} needs non-zero dimensions, got {width}x{height}"
        )));
    }
    if width > max_dimension || height > max_dimension {
        return Err(NativeError::Source(format!(
            "s{index} image {width}x{height} exceeds this renderer's {max_dimension}-pixel dimension limit"
        )));
    }
    let bytes_per_row = width.checked_mul(4).ok_or_else(|| {
        NativeError::Source(format!(
            "s{index} image width {width} overflows its RGBA row size"
        ))
    })?;
    let expected = usize::try_from(bytes_per_row)
        .ok()
        .and_then(|row| usize::try_from(height).ok()?.checked_mul(row))
        .ok_or_else(|| {
            NativeError::Source(format!(
                "s{index} image dimensions {width}x{height} overflow their RGBA byte count"
            ))
        })?;
    if bytes != expected {
        return Err(NativeError::Source(format!(
            "s{index} wants exactly {expected} bytes for {width}x{height} RGBA, got {bytes}"
        )));
    }
    Ok(SourceLayout {
        bytes: expected,
        bytes_per_row,
    })
}

/// What one renderer has asked the GPU API for, by the size of each thing
/// it asked for.
///
/// The renderer is the only thing that knows which textures it holds and
/// at what size, and the studio's memory breakdown wants to know where the
/// visuals' share of the process went. So this is counted from the wgpu
/// objects themselves - the ten output textures (`o0`-`o3` and the
/// display, each double-buffered), `s0`-`s3`, the uniform buffers and the
/// readback buffer - rather than guessed from a render size.
///
/// It is an estimate. The driver's own memory is not in it:
/// the device, compiled pipelines, command buffers, the padding and
/// granularity its allocator adds, and on a software adapter the
/// rasteriser's threads and generated code. Nothing in the API reports
/// those, and they can be larger than everything counted here.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NativeFootprint {
    /// The size everything is drawn at, supersampling included.
    pub width: u32,
    pub height: u32,
    /// The output, display and source textures and the uniform buffers,
    /// as allocated.
    pub textures: usize,
    /// The same, less the output and display halves nothing has written
    /// yet. A software adapter's textures are process memory that takes
    /// pages only once something writes them. A one-output sketch never
    /// writes the back halves of the other outputs, and writes the display
    /// only through `render()` or `hush()`: half the ten. The process
    /// figure does not hold the unwritten halves.
    pub resident: usize,
    /// The buffer every frame is read back through: memory the CPU maps,
    /// so it is taken as the process's on every adapter.
    pub readback: usize,
    /// The textures are memory the process's own figure counts: a
    /// software rasteriser's, which it allocates on the heap, and Apple
    /// silicon's, whose footprint counts the graphics memory it shares.
    /// On any other GPU - a discrete card, and an integrated one on Linux
    /// or Windows too - they are the driver's buffers, which neither the
    /// anonymous and shared RSS nor the private working set includes, so
    /// adding them would take from the remainder what was never in it.
    pub textures_in_process: bool,
}

impl NativeFootprint {
    /// What counts toward the process: the readback, and the textures
    /// written so far where they are the process's.
    pub fn process_bytes(&self) -> usize {
        self.readback
            + if self.textures_in_process {
                self.resident
            } else {
                0
            }
    }

    /// What the driver holds for the textures outside the process's figure,
    /// at the size allocated: a GPU's memory is set aside when it is asked
    /// for, not when it is first drawn into.
    pub fn gpu_bytes(&self) -> usize {
        if self.textures_in_process {
            0
        } else {
            self.textures
        }
    }
}

/// One of hydra's outputs, double-buffered.
///
/// A sketch may read the output it is drawing into - `src(o0).modulate(…)` is
/// how every feedback effect in hydra works - and a texture cannot be sampled
/// and written in one pass. So each output keeps two, samples the front and
/// draws into the back, then swaps.
struct Output {
    textures: [wgpu::Texture; 2],
    views: [wgpu::TextureView; 2],
    front: usize,
    /// Which halves anything has written - a pass drawing into it, a hush
    /// clearing it, or wgpu zero-filling it the first time a bind group
    /// samples it. What the footprint counts as resident; a resize builds
    /// new outputs, and so starts it over.
    written: [bool; 2],
}

impl Output {
    fn new(device: &wgpu::Device, width: u32, height: u32, index: usize) -> Self {
        Self::named(device, width, height, &format!("o{index}"))
    }

    fn named(device: &wgpu::Device, width: u32, height: u32, label: &str) -> Self {
        let make = |half: usize| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("hydra {label}[{half}]")),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let textures = [make(0), make(1)];
        let views = [
            textures[0].create_view(&wgpu::TextureViewDescriptor::default()),
            textures[1].create_view(&wgpu::TextureViewDescriptor::default()),
        ];
        Self {
            textures,
            views,
            front: 0,
            written: [false; 2],
        }
    }

    /// What a sketch samples when it reads this output.
    fn front(&self) -> &wgpu::TextureView {
        &self.views[self.front]
    }

    /// What the next pass draws into.
    fn back(&self) -> &wgpu::TextureView {
        &self.views[1 - self.front]
    }

    fn swap(&mut self) {
        self.front = 1 - self.front;
    }

    /// The bytes of the halves written so far.
    fn written_bytes(&self) -> u64 {
        self.textures
            .iter()
            .zip(self.written)
            .filter(|(_, written)| *written)
            .map(|(texture, _)| texture_bytes(texture))
            .sum()
    }

    fn readable(&self) -> &wgpu::Texture {
        &self.textures[self.front]
    }
}

/// A texture's size in bytes: every texture here is `Rgba8Unorm`, four
/// bytes a texel.
fn texture_bytes(texture: &wgpu::Texture) -> u64 {
    let size = texture.size();
    u64::from(size.width) * u64::from(size.height) * 4
}

/// Whether an adapter's textures are memory the process's own figure
/// counts - `mem` is the anonymous and shared RSS on Linux, the private
/// working set on Windows and the physical footprint on macOS.
///
/// A software rasteriser's are: it allocates them on the heap. Apple
/// silicon's are: its GPU shares the memory, and the footprint counts
/// what the graphics stack holds for the process. Any other GPU's are not,
/// integrated or discrete: they are the kernel driver's buffers on Linux
/// and the video memory manager's on Windows - "shared GPU memory" there
/// on an integrated one - which the render targets are never mapped into
/// the process to be counted in. The GL backend also calls most real
/// cards `Other`, so only the two known cases count.
fn textures_count_in_process(device: wgpu::DeviceType) -> bool {
    device == wgpu::DeviceType::Cpu
        || (cfg!(target_os = "macos") && device == wgpu::DeviceType::IntegratedGpu)
}

#[derive(Clone, Copy)]
enum DrawTarget {
    Output(usize),
    Display,
}

/// A headless renderer for Hydra chains.
pub struct NativeRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// What the adapter reported, for the log line that tells a reader whether
    /// they are on hardware or on llvmpipe.
    adapter: String,
    /// The textures count toward the process's own figure: see
    /// [`NativeFootprint::textures_in_process`].
    textures_in_process: bool,
    width: u32,
    height: u32,
    /// One pipeline per composed shader, kept because compiling is the
    /// expensive part and a sketch changes far less often than a frame.
    pipelines: HashMap<String, wgpu::RenderPipeline>,
    /// Every pipeline this device ever compiled, evictions included. The
    /// driver keeps private memory per compiled shader that outlives the
    /// wgpu objects (measurably ~30 KB each under software Vulkan), so a
    /// long generating session must eventually swap the whole device - the
    /// host watches this and recreates the renderer past a threshold.
    pipelines_built: usize,
    /// Cache keys, oldest first, so the cache can be BOUNDED: every edited
    /// or generated chain is a new shader, and a session of generating
    /// used to keep every compiled pipeline it had ever seen.
    pipeline_order: Vec<String>,
    pipeline_layout: wgpu::PipelineLayout,
    globals: wgpu::Buffer,
    /// The values behind `H(pattern)`, one per declared signal.
    signals: wgpu::Buffer,
    /// `a.fft`, what the engine says it is playing.
    audio: [f32; crate::HYDRA_AUDIO_BINS],
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// `o0`-`o3`, each double-buffered for feedback.
    outputs: Vec<Output>,
    /// The canvas-only four-up picture produced by bare `render()`. It is not
    /// one of Hydra's addressable outputs, so composing it cannot perturb o0
    /// or any feedback chain.
    display: Output,
    /// `s0`-`s3`, and the textures behind them. Black until something fills
    /// one; `feedStrudel` fills `s0` with the terminal.
    sources: Vec<wgpu::Texture>,
    source_views: Vec<wgpu::TextureView>,
    readback: wgpu::Buffer,
    /// Tight CPU row/frame sizes, checked when the output is created.
    row_bytes: usize,
    frame_bytes: usize,
    /// Bytes in a readback row, rounded up to the alignment wgpu requires.
    padded_row: u32,
    /// Total mapped readback size, checked against device and host bounds.
    readback_bytes: usize,
}

impl NativeRenderer {
    /// Open a device and size the frame. Blocks; call it off the audio thread.
    pub fn new(width: u32, height: u32) -> Result<Self, NativeError> {
        pollster::block_on(Self::open(width, height))
    }

    async fn open(width: u32, height: u32) -> Result<Self, NativeError> {
        // Refuse protocol/resource-invalid geometry before probing adapters;
        // a hostile size must not make backend selection allocate anything.
        let _ = output_layout(width, height, crate::HYDRA_MAX_OUTPUT_EDGE, u64::MAX)?;
        // Prefer Vulkan: a Windows DX12 adapter produced black frames while
        // Vulkan rendered correctly. Try the remaining backends if it fails.
        // `WGPU_BACKEND` overrides this order through `Backends::from_env()`.
        let mut refused = Vec::new();
        for backends in [
            wgpu::Backends::VULKAN,
            wgpu::Backends::METAL,
            wgpu::Backends::DX12,
            wgpu::Backends::GL,
        ] {
            match Self::open_on(backends, width, height).await {
                Ok(renderer) => return Ok(renderer),
                Err(error) => refused.push(format!("{backends:?}: {error}")),
            }
        }
        Err(NativeError::NoAdapter(refused.join("; ")))
    }

    async fn open_on(
        backends: wgpu::Backends,
        width: u32,
        height: u32,
    ) -> Result<Self, NativeError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::from_env().unwrap_or(backends),
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|error| NativeError::NoAdapter(error.to_string()))?;
        let adapter_limits = adapter.limits();
        let _ = output_layout(
            width,
            height,
            adapter_limits.max_texture_dimension_2d,
            adapter_limits.max_buffer_size,
        )?;
        let (name, textures_in_process) = {
            let info = adapter.get_info();
            (
                format!("{} ({:?})", info.name, info.backend),
                textures_count_in_process(info.device_type),
            )
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|error| NativeError::NoDevice(error.to_string()))?;
        let device_limits = device.limits();
        let geometry = output_layout(
            width,
            height,
            device_limits.max_texture_dimension_2d,
            device_limits.max_buffer_size,
        )?;
        let readback_bytes = usize::try_from(geometry.readback_bytes).map_err(|_| {
            NativeError::Geometry("readback buffer does not fit host memory".into())
        })?;

        // Globals, then a texture and a sampler for every name a sketch can
        // reach: s0-s3 and o0-o3, in the order `uplift` assigns them. All are
        // declared whether or not a given shader uses them - wgpu allows a
        // layout wider than the shader, and one fixed layout means one
        // pipeline layout for every sketch.
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: crate::native::uplift::GLOBALS_BINDING,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for slot in 0..crate::native::uplift::TEXTURES.len() as u32 {
            let binding = crate::native::uplift::GLOBALS_BINDING + 1 + slot * 2;
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: binding + 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: SIGNALS_BINDING,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hydra"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hydra"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        // std140: `vec2` at 0, `float` at 8, then four `vec4`s beginning on a
        // sixteen-byte boundary. Sixteen bands cover shared sketches that use
        // `a.setBins(...)` while keeping one fixed pipeline layout.
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hydra globals"),
            size: (16 + crate::HYDRA_AUDIO_BINS * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let signals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hydra signals"),
            size: (MAX_SIGNALS * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hydra"),
            // hydra wraps: `fract(_st)` in `src` says so, and repeat is what
            // its `repeat`/`scroll` transforms expect at the edges.
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            // hydra-synth creates output textures with nearest sampling, and
            // regl's source-texture default is nearest too.
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let outputs: Vec<Output> = (0..OUTPUTS)
            .map(|index| Output::new(&device, width, height, index))
            .collect();
        let display = Output::named(&device, width, height, "display");
        // An empty source is a black texture rather than an unbound slot, so a
        // sketch that samples `s0` before anything fills it draws black
        // instead of failing to build a bind group.
        let sources: Vec<wgpu::Texture> = (0..SOURCES)
            .map(|index| Self::source_texture(&device, 1, 1, index))
            .collect();
        let source_views: Vec<wgpu::TextureView> = sources
            .iter()
            .map(|texture| texture.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect();

        let readback = Self::readback(&device, geometry);
        Ok(Self {
            device,
            queue,
            adapter: name,
            textures_in_process,
            width,
            height,
            pipelines: HashMap::new(),
            pipeline_order: Vec::new(),
            pipelines_built: 0,
            pipeline_layout,
            globals,
            signals,
            audio: [0.0; crate::HYDRA_AUDIO_BINS],
            layout,
            sampler,
            outputs,
            display,
            sources,
            source_views,
            readback,
            row_bytes: geometry.row_bytes,
            frame_bytes: geometry.frame_bytes,
            padded_row: geometry.padded_row,
            readback_bytes,
        })
    }

    fn source_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        index: usize,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("hydra s{index}")),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    /// Fill one of `s0`-`s3` with a picture a sketch can sample.
    ///
    /// This is where `feedStrudel` puts the terminal. `rgba` is row-major,
    /// **top row first** - the order a terminal grid is written in - and is
    /// uploaded as it stands.
    ///
    /// This function does not flip the picture. The picture is stored
    /// exactly as it arrives and turned the right way up where it is read:
    /// [`uplift`](crate::native::uplift) flips the coordinate every texture
    /// fetch is indexed by. That one flip also serves `o0`-`o3` and
    /// `prevBuffer`, which are render targets that nothing uploads. A second
    /// flip on upload would put these four sources back upside down while
    /// leaving the other five right.
    pub fn set_source(
        &mut self,
        index: usize,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<(), NativeError> {
        let layout = source_layout(
            index,
            width,
            height,
            rgba.len(),
            self.device.limits().max_texture_dimension_2d,
        )?;
        let size = self.sources[index].size();
        if size.width != width || size.height != height {
            self.sources[index] = Self::source_texture(&self.device, width, height, index);
            self.source_views[index] =
                self.sources[index].create_view(&wgpu::TextureViewDescriptor::default());
        }
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.sources[index],
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba[..layout.bytes],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(layout.bytes_per_row),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    /// Return one of `s0`-`s3` to a known transparent-black pixel.
    ///
    /// Replacing the texture also discards a potentially large decoded image
    /// immediately. The index is checked just as strictly as [`Self::set_source`].
    pub fn clear_source(&mut self, index: usize) -> Result<(), NativeError> {
        self.set_source(index, 1, 1, &[0, 0, 0, 0])
    }

    /// What a frame is read back through.
    fn readback(device: &wgpu::Device, geometry: OutputLayout) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hydra readback"),
            size: geometry.readback_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    /// Change the size everything is drawn and read at.
    ///
    /// Only the textures are rebuilt. Opening a device costs tens of
    /// milliseconds and there is no reason to pay it because a terminal was
    /// dragged - and the pipeline cache, which is the expensive part, is
    /// keyed on the shader and survives.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), NativeError> {
        let limits = self.device.limits();
        let geometry = output_layout(
            width,
            height,
            limits.max_texture_dimension_2d,
            limits.max_buffer_size,
        )?;
        if (self.width, self.height) == (width, height) {
            return Ok(());
        }
        let readback_bytes = usize::try_from(geometry.readback_bytes).map_err(|_| {
            NativeError::Geometry("readback buffer does not fit host memory".into())
        })?;
        // Build every replacement before publishing the new dimensions. A
        // rejected resize leaves the old renderer fully usable.
        let outputs = (0..OUTPUTS)
            .map(|index| Output::new(&self.device, width, height, index))
            .collect();
        let display = Output::named(&self.device, width, height, "display");
        let readback = Self::readback(&self.device, geometry);
        self.width = width;
        self.height = height;
        self.outputs = outputs;
        self.display = display;
        self.readback = readback;
        self.row_bytes = geometry.row_bytes;
        self.frame_bytes = geometry.frame_bytes;
        self.padded_row = geometry.padded_row;
        self.readback_bytes = readback_bytes;
        Ok(())
    }

    /// The adapter this is drawing on - hardware, or a software rasteriser.
    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// What this renderer holds through the GPU API right now, read off
    /// the textures and buffers themselves: a resize, a source upload and
    /// a cleared source all show in the next call. A handful of
    /// multiplications, so the host can ask every frame.
    pub fn footprint(&self) -> NativeFootprint {
        let outputs = || self.outputs.iter().chain(std::iter::once(&self.display));
        // The sources are written by every upload, the one-texel stand-in
        // included, and the uniforms by every draw.
        let always = self.sources.iter().map(texture_bytes).sum::<u64>()
            + self.globals.size()
            + self.signals.size();
        let textures = outputs()
            .flat_map(|output| output.textures.iter())
            .map(texture_bytes)
            .sum::<u64>()
            + always;
        let resident = outputs().map(Output::written_bytes).sum::<u64>() + always;
        let bytes = |count: u64| usize::try_from(count).unwrap_or(usize::MAX);
        NativeFootprint {
            width: self.width,
            height: self.height,
            textures: bytes(textures),
            resident: bytes(resident),
            readback: bytes(self.readback.size()),
            textures_in_process: self.textures_in_process,
        }
    }

    /// Draw a chain at `time` seconds and return RGBA from `o0`, top row
    /// first, which is the order the terminal draws rows in.
    pub fn render(&mut self, node: &HydraNode, time: f32) -> Result<Vec<u8>, NativeError> {
        self.draw(node, time)?;
        self.read(0)
    }

    /// Draw a chain into whatever output its `.out()` names, without reading
    /// anything back. A sketch with several chains draws each in turn.
    pub fn draw(&mut self, node: &HydraNode, time: f32) -> Result<(), NativeError> {
        let composed = compose_full(node, "highp")?;
        let shader = uplift_with_signal_slots(&composed.shader, &composed.signals);
        self.draw_shader(&shader, output_of(node), time)
    }

    /// Set what `a.fft` reads.
    pub fn set_audio(&mut self, bands: [f32; crate::HYDRA_AUDIO_BINS]) {
        self.audio = bands;
    }

    /// Set what `H(pattern)` reads this frame, one value per signal slot.
    pub fn set_signals(&self, values: &[f32]) {
        let mut packed = [0f32; MAX_SIGNALS];
        for (slot, value) in packed.iter_mut().zip(values) {
            *slot = *value;
        }
        let bytes: Vec<u8> = packed.iter().flat_map(|v| v.to_ne_bytes()).collect();
        self.queue.write_buffer(&self.signals, 0, &bytes);
    }

    /// The same, for a shader already composed, into a named output.
    pub fn draw_shader(
        &mut self,
        shader: &str,
        output: usize,
        time: f32,
    ) -> Result<(), NativeError> {
        let output = output.min(self.outputs.len() - 1);
        self.draw_shader_to(shader, DrawTarget::Output(output), time)
    }

    fn draw_shader_to(
        &mut self,
        shader: &str,
        target: DrawTarget,
        time: f32,
    ) -> Result<(), NativeError> {
        if !self.pipelines.contains_key(shader) {
            let pipeline = self.build(shader)?;
            // A dozen live layers is a big sketch; past the cap the oldest
            // compiled pipeline goes, because a session that generates
            // sketch after sketch must not keep them all on the GPU.
            const PIPELINE_CACHE_CAP: usize = 24;
            while self.pipeline_order.len() >= PIPELINE_CACHE_CAP {
                let oldest = self.pipeline_order.remove(0);
                self.pipelines.remove(&oldest);
            }
            self.pipelines.insert(shader.to_owned(), pipeline);
            self.pipeline_order.push(shader.to_owned());
            self.pipelines_built += 1;
        }

        let mut globals = [0u8; 16 + crate::HYDRA_AUDIO_BINS * 4];
        globals[0..4].copy_from_slice(&(self.width as f32).to_ne_bytes());
        globals[4..8].copy_from_slice(&(self.height as f32).to_ne_bytes());
        globals[8..12].copy_from_slice(&time.to_ne_bytes());
        for (band, value) in self.audio.iter().enumerate() {
            let at = 16 + band * 4;
            globals[at..at + 4].copy_from_slice(&value.to_ne_bytes());
        }
        self.queue.write_buffer(&self.globals, 0, &globals);

        // Every texture a sketch could sample, bound at once. The output being
        // drawn into contributes its front buffer here while the pass writes
        // its back buffer. This is why each output has two.
        // `prevBuffer` is upstream's previous buffer for the output currently
        // being rendered, not an alias for o0. The display compositor does
        // not read prevBuffer, so o0 is only a harmless layout filler there.
        let feedback = match target {
            DrawTarget::Output(output) => output,
            DrawTarget::Display => 0,
        };
        let bind = self.bind_group(feedback);
        let pipeline = &self.pipelines[shader];
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hydra"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hydra"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: match target {
                        DrawTarget::Output(output) => self.outputs[output].back(),
                        DrawTarget::Display => self.display.back(),
                    },
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        // Every output's front was bound, which has wgpu zero-fill it the
        // first time, and the target's back was drawn into.
        for output in &mut self.outputs {
            output.written[output.front] = true;
        }
        let drawn = match target {
            DrawTarget::Output(output) => &mut self.outputs[output],
            DrawTarget::Display => &mut self.display,
        };
        drawn.written[1 - drawn.front] = true;
        drawn.swap();
        Ok(())
    }

    /// Every binding a sketch can reach, in the order `uplift` assigns them.
    fn bind_group(&self, feedback_output: usize) -> wgpu::BindGroup {
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: crate::native::uplift::GLOBALS_BINDING,
            resource: self.globals.as_entire_binding(),
        }];
        // s0-s3, then o0-o3, then `prevBuffer` for the output this pass draws.
        let views = self
            .source_views
            .iter()
            .chain(self.outputs.iter().map(Output::front))
            .chain(std::iter::once(self.outputs[feedback_output].front()));
        for (slot, view) in views.enumerate() {
            let binding = crate::native::uplift::GLOBALS_BINDING + 1 + slot as u32 * 2;
            entries.push(wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(view),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: binding + 1,
                resource: wgpu::BindingResource::Sampler(&self.sampler),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: SIGNALS_BINDING,
            resource: self.signals.as_entire_binding(),
        });
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hydra"),
            layout: &self.layout,
            entries: &entries,
        })
    }

    /// Draw all four outputs into the canvas-only display target, in
    /// quadrants - hydra's `render()`. None of o0-o3 is written or swapped.
    pub fn render_all(&mut self) -> Result<(), NativeError> {
        let shader = render_all_shader();
        self.draw_shader_to(&shader, DrawTarget::Display, 0.0)
    }

    /// Clear every output buffer, matching hydra-synth's `hush()` without
    /// changing which output or composite the canvas is currently showing.
    pub fn hush(&mut self) {
        for index in 0..SOURCES {
            let _ = self.clear_source(index);
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hydra hush"),
            });
        for view in self
            .outputs
            .iter()
            .flat_map(|output| output.views.iter())
            .chain(self.display.views.iter())
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hydra hush"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }
        self.queue.submit([encoder.finish()]);
        for output in self
            .outputs
            .iter_mut()
            .chain(std::iter::once(&mut self.display))
        {
            output.written = [true; 2];
        }
    }

    /// How many shaders this device has ever compiled - the host's cue to
    /// retire the device before the driver's per-shader residue adds up.
    pub fn shaders_built(&self) -> usize {
        self.pipelines_built
    }

    /// Read one output back as RGBA, top row first.
    pub fn read(&self, output: usize) -> Result<Vec<u8>, NativeError> {
        let output = output.min(self.outputs.len() - 1);
        self.read_texture(self.outputs[output].readable())
    }

    /// Read the canvas-only result of bare `render()`.
    pub fn read_display(&self) -> Result<Vec<u8>, NativeError> {
        self.read_texture(self.display.readable())
    }

    fn read_texture(&self, texture: &wgpu::Texture) -> Result<Vec<u8>, NativeError> {
        // Reserve before submitting or mapping the GPU buffer. A fallible CPU
        // allocation must not leave a mapped buffer behind on its error path.
        let mut pixels = reserve_frame(self.width, self.height, self.frame_bytes)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hydra read"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([encoder.finish()]);

        let slice = self.readback.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::Wait);
        let mapped = receiver
            .recv()
            .map_err(|error| NativeError::Readback(error.to_string()))
            .and_then(|result| result.map_err(|error| NativeError::Readback(error.to_string())));
        if let Err(error) = mapped {
            // `unmap` is harmless when the backend rejected the mapping and
            // essential when delivery of a successful callback was lost.
            self.readback.unmap();
            return Err(error);
        }
        let result = (|| -> Result<(), NativeError> {
            let data = slice.get_mapped_range();
            if data.len() < self.readback_bytes {
                Err(NativeError::Readback(format!(
                    "mapped {} bytes for a {}-byte readback",
                    data.len(),
                    self.readback_bytes
                )))
            } else {
                // Rows in order, top first, which is what the terminal draws.
                //
                // No flip: `uplift` already puts hydra's `st` back on WebGL's
                // bottom-up axis, so the frame is the right way up in the
                // framebuffer. Flipping here as well was what made position look
                // correct while every rotation turned the wrong way.
                for line in 0..self.height {
                    let start = usize::try_from(line)
                        .ok()
                        .and_then(|line| line.checked_mul(self.padded_row as usize))
                        .ok_or_else(|| {
                            NativeError::Readback("mapped row offset overflowed host memory".into())
                        })?;
                    let end = start.checked_add(self.row_bytes).ok_or_else(|| {
                        NativeError::Readback("mapped row end overflowed host memory".into())
                    })?;
                    let row = data.get(start..end).ok_or_else(|| {
                        NativeError::Readback(format!(
                            "mapped readback ended before row {line} ({start}..{end} of {})",
                            data.len()
                        ))
                    })?;
                    pixels.extend_from_slice(row);
                }
                if pixels.len() != self.frame_bytes {
                    Err(NativeError::Readback(format!(
                        "assembled {} bytes for a {}-byte RGBA frame",
                        pixels.len(),
                        self.frame_bytes
                    )))
                } else {
                    Ok(())
                }
            }
        })();
        self.readback.unmap();
        result?;
        Ok(pixels)
    }

    /// Compile a shader into a pipeline.
    ///
    /// wgpu reports a bad shader through an error scope rather than a
    /// `Result`, and the default handler panics. A sketch that will not
    /// compile is an ordinary thing to be told about, so the scope is caught
    /// and turned into one.
    fn build(&self, shader: &str) -> Result<wgpu::RenderPipeline, NativeError> {
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = self.build_unchecked(shader);
        if let Some(error) = pollster::block_on(self.device.pop_error_scope()) {
            return Err(NativeError::Shader(error.to_string()));
        }
        Ok(pipeline)
    }

    fn build_unchecked(&self, shader: &str) -> wgpu::RenderPipeline {
        let vertex = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("hydra vertex"),
                source: wgpu::ShaderSource::Glsl {
                    shader: VERTEX.into(),
                    stage: wgpu::naga::ShaderStage::Vertex,
                    defines: Default::default(),
                },
            });
        let fragment = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("hydra fragment"),
                source: wgpu::ShaderSource::Glsl {
                    shader: shader.into(),
                    stage: wgpu::naga::ShaderStage::Fragment,
                    defines: Default::default(),
                },
            });
        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("hydra"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vertex,
                    entry_point: Some("main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fragment,
                    entry_point: Some("main"),
                    targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
    }
}

impl std::fmt::Debug for NativeRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeRenderer")
            .field("adapter", &self.adapter)
            .field("size", &(self.width, self.height))
            .field("pipelines", &self.pipelines.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refusal(index: usize, width: u32, height: u32, bytes: usize) -> String {
        source_layout(index, width, height, bytes, 4096)
            .expect_err("invalid source metadata")
            .to_string()
    }

    #[test]
    fn source_metadata_requires_a_real_slot_nonzero_dimensions_and_exact_rgba() {
        assert!(refusal(SOURCES, 1, 1, 4).contains("outside s0..s3"));
        assert!(refusal(0, 0, 1, 0).contains("non-zero dimensions"));
        assert!(refusal(0, 1, 0, 0).contains("non-zero dimensions"));
        assert!(refusal(0, 4097, 1, 4097 * 4).contains("dimension limit"));
        assert!(refusal(0, 2, 3, 23).contains("exactly 24 bytes"));
        assert!(refusal(0, 2, 3, 25).contains("exactly 24 bytes"));
        assert_eq!(
            source_layout(3, 2, 3, 24, 4096).expect("exact RGBA image"),
            SourceLayout {
                bytes: 24,
                bytes_per_row: 8
            }
        );
    }

    #[test]
    fn output_geometry_is_bounded_and_checked_before_allocation() {
        let boundary = output_layout(
            crate::HYDRA_MAX_OUTPUT_EDGE,
            1024,
            crate::HYDRA_MAX_OUTPUT_EDGE,
            u64::MAX,
        )
        .expect("4096x1024 is the exact four-megapixel boundary");
        assert_eq!(boundary.pixels, crate::HYDRA_MAX_OUTPUT_PIXELS);
        assert_eq!(
            boundary.aggregate_pixels,
            crate::HYDRA_MAX_OUTPUT_PIXELS * OUTPUT_TEXTURE_COPIES
        );
        assert_eq!(boundary.row_bytes, 16_384);
        assert_eq!(boundary.frame_bytes, 16 * 1024 * 1024);
        assert_eq!(boundary.readback_bytes, 16 * 1024 * 1024);

        for (width, height) in [(0, 1), (1, 0), (4097, 1), (4096, 1025)] {
            assert!(
                output_layout(width, height, u32::MAX, u64::MAX).is_err(),
                "{width}x{height} must be refused"
            );
        }

        let adapter = output_layout(2048, 1, 1024, u64::MAX)
            .expect_err("adapter dimensions are authoritative")
            .to_string();
        assert!(adapter.contains("adapter's 1024-pixel"), "{adapter}");
        let buffer = output_layout(1024, 1024, 4096, 100)
            .expect_err("device buffer limit is authoritative")
            .to_string();
        assert!(buffer.contains("device permits 100"), "{buffer}");

        let overflow = output_layout(u32::MAX, u32::MAX, u32::MAX, u64::MAX)
            .expect_err("ten output surfaces overflow u64")
            .to_string();
        assert!(overflow.contains("render textures overflows"), "{overflow}");
    }

    #[test]
    fn cpu_frame_reservation_is_fallible_before_mapping() {
        let error = reserve_frame(1, 1, usize::MAX)
            .expect_err("an impossible Vec capacity is fallible")
            .to_string();
        assert!(error.contains("could not reserve"), "{error}");
    }
}
