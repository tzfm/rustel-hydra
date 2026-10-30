//! The engine's handle on the renderer.
//!
//! One worker thread owns a [`NativeRenderer`]; everything else talks to it
//! through channels and atomics. The thread exists because a frame costs a few
//! milliseconds of GPU round-trip and the Session must not wait on it.
//!
//! Two pictures are drawn: the score's, behind the code, and the snippet
//! shelf's thumbnail. Separate renderers, so browsing the shelf does not take
//! the screen from a set that is playing.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak, mpsc};
use std::time::{Duration, Instant};

use crate::native::{NativeFootprint, NativeRenderer};
use crate::program::{HydraNode, HydraProgram, HydraStatement};
use crate::{HydraAudioFrame, HydraSignalFrame, HydraTuiFrame};

/// Something the renderer has to tell the engine.
#[derive(Clone, Debug, PartialEq)]
pub enum HydraEvent {
    /// The renderer is up. Carries what the adapter calls itself, which is how
    /// a reader tells hardware from a software rasteriser.
    Ready { renderer: String },
    /// A sketch could not be drawn. The message names why.
    Failed { message: String },
    /// What the renderer is managing, measured over two seconds.
    Frames { fps: u32 },
    /// An external texture source changed state. Acquisition lives above this
    /// crate, but shares the renderer's diagnostic stream so every product
    /// surface reports it consistently.
    Input {
        slot: u8,
        message: String,
        failed: bool,
    },
}

/// Why visuals could not be started.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HydraHostError {
    /// The renderer thread could not be started.
    Thread(String),
    /// No GPU and no software rasteriser.
    NoRenderer(String),
    /// The sketch is outside a documented ceiling.
    Program(String),
}

impl std::fmt::Display for HydraHostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Thread(why) => write!(f, "the renderer thread could not start: {why}"),
            Self::NoRenderer(why) => write!(f, "{why}"),
            Self::Program(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for HydraHostError {}

/// A frame waiting to be drawn, replaced rather than queued.
#[derive(Default)]
struct FrameSlot {
    frame: Mutex<Option<(u16, u16, Vec<u8>)>>,
}

#[derive(Default)]
struct TuiSlot {
    /// The terminal's own grid, waiting to become a texture.
    grid: Mutex<Option<HydraTuiFrame>>,
    sequence: AtomicU32,
}

struct SignalSchedule {
    received: Instant,
    step_ms: f64,
    slots: Vec<Vec<f32>>,
}

impl SignalSchedule {
    fn from_frame(frame: &HydraSignalFrame, received: Instant) -> Self {
        let step_ms = if frame.step.is_finite() && frame.step > 0.0 {
            frame.step
        } else {
            crate::HYDRA_SIGNAL_STEP_MS
        };
        let slots = frame
            .slots
            .iter()
            .take(crate::MAX_HYDRA_SIGNALS)
            .map(|samples| {
                samples
                    .iter()
                    .take(crate::HYDRA_SIGNAL_SAMPLES)
                    .map(|sample| sample.as_f64().unwrap_or(0.0) as f32)
                    .collect()
            })
            .collect();
        Self {
            received,
            step_ms,
            slots,
        }
    }

    fn values_at(&self, now: Instant) -> Vec<f32> {
        let elapsed_ms = now.saturating_duration_since(self.received).as_secs_f64() * 1000.0;
        let index = (elapsed_ms / self.step_ms).floor() as usize;
        self.slots
            .iter()
            .map(|samples| {
                samples
                    .get(index.min(samples.len().saturating_sub(1)))
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect()
    }
}

/// One decoded external frame. The bytes are shared with the renderer worker:
/// a camera replaces this at frame rate, and copying every publication would
/// turn the latest-only slot back into a queue's worth of memory traffic.
#[derive(Clone)]
struct InputFrame {
    width: u32,
    height: u32,
    rgba: Arc<[u8]>,
}

/// One of Hydra's `s0`-`s3` external inputs.
///
/// `generation` belongs to the source request, while `revision` belongs to
/// the texture contents. Rebinding a slot advances both and empties it, so a
/// slow image request or a camera thread stopped by policy can never publish
/// over the source that replaced it.
#[derive(Default)]
struct InputSlot {
    generation: AtomicU64,
    revision: AtomicU64,
    frame: Mutex<Option<InputFrame>>,
    /// The held frame's size in bytes, written under the frame's lock so a
    /// memory reading can have it without taking that lock from a camera
    /// publishing at frame rate.
    bytes: AtomicUsize,
}

impl InputSlot {
    fn bind(&self) -> u64 {
        let generation = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
        if let Ok(mut frame) = self.frame.lock() {
            frame.take();
            self.bytes.store(0, Ordering::Relaxed);
            self.revision.fetch_add(1, Ordering::Release);
        }
        generation
    }

    fn clear(&self) {
        self.bind();
    }

    /// Clear only if `generation` still owns this slot. The frame lock makes
    /// the comparison and generation advance atomic with respect to publish;
    /// a successor bind that wins the race can never be cleared by its old
    /// camera worker.
    fn clear_generation(&self, generation: u64) -> bool {
        let Ok(mut frame) = self.frame.lock() else {
            return false;
        };
        if self
            .generation
            .compare_exchange(
                generation,
                generation.wrapping_add(1),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return false;
        }
        frame.take();
        self.bytes.store(0, Ordering::Relaxed);
        self.revision.fetch_add(1, Ordering::Release);
        true
    }

    fn snapshot(&self) -> (u64, Option<InputFrame>) {
        let Ok(frame) = self.frame.lock() else {
            return (self.revision.load(Ordering::Acquire), None);
        };
        (self.revision.load(Ordering::Acquire), frame.clone())
    }
}

/// What one of the worker's renderers holds, as the worker last saw it.
///
/// The worker is the only thread that can touch a renderer, and the one
/// asking is the terminal's, which must never wait on a GPU round-trip. So
/// the worker writes what it holds here once a frame - after it has opened,
/// resized or retired anything - and a reader takes it with a few loads.
/// The fields are read one by one and may be a frame apart; they are an
/// estimate either way.
#[derive(Default)]
struct RendererGauge {
    /// `width << 32 | height`; zero while no renderer is open.
    size: AtomicU64,
    textures: AtomicUsize,
    resident: AtomicUsize,
    readback: AtomicUsize,
    textures_in_process: AtomicBool,
}

impl RendererGauge {
    fn record(&self, footprint: Option<NativeFootprint>) {
        let footprint = footprint.unwrap_or_default();
        self.textures.store(footprint.textures, Ordering::Relaxed);
        self.resident.store(footprint.resident, Ordering::Relaxed);
        self.readback.store(footprint.readback, Ordering::Relaxed);
        self.textures_in_process
            .store(footprint.textures_in_process, Ordering::Relaxed);
        // The size last: a reader that sees a renderer sees its bytes.
        self.size.store(
            (u64::from(footprint.width) << 32) | u64::from(footprint.height),
            Ordering::Release,
        );
    }

    fn read(&self) -> Option<NativeFootprint> {
        let size = self.size.load(Ordering::Acquire);
        (size != 0).then(|| NativeFootprint {
            width: (size >> 32) as u32,
            height: size as u32,
            textures: self.textures.load(Ordering::Relaxed),
            resident: self.resident.load(Ordering::Relaxed),
            readback: self.readback.load(Ordering::Relaxed),
            textures_in_process: self.textures_in_process.load(Ordering::Relaxed),
        })
    }
}

/// What Hydra holds in memory, estimated from what it asked for.
///
/// Up to three renderers, one per picture, each open only while its picture
/// is drawn and for a few seconds after; and the camera and image frames
/// decoded for `s0`-`s3`. Each renderer's share is a [`NativeFootprint`],
/// with what that can and cannot count.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct HydraMemory {
    /// The score's picture, behind the code.
    pub score: Option<NativeFootprint>,
    /// The snippet shelf's preview, drawn as the studio's background
    /// while a sketch is browsed.
    pub preview: Option<NativeFootprint>,
    /// The theme's own sketch.
    pub theme: Option<NativeFootprint>,
    /// Camera and image frames decoded and waiting for the GPU: process
    /// memory, whatever the adapter.
    pub inputs: usize,
}

impl HydraMemory {
    /// The open renderers, named for the picture each draws.
    pub fn renderers(&self) -> impl Iterator<Item = (&'static str, NativeFootprint)> {
        [
            ("score", self.score),
            ("theme", self.theme),
            ("shelf preview", self.preview),
        ]
        .into_iter()
        .filter_map(|(name, footprint)| Some((name, footprint?)))
    }

    /// What counts toward the process: the renderers' shares of it and the
    /// inputs.
    pub fn process_bytes(&self) -> usize {
        self.renderers()
            .map(|(_, footprint)| footprint.process_bytes())
            .sum::<usize>()
            + self.inputs
    }

    /// Textures the GPU's driver holds outside the process's figure: any
    /// GPU's but Apple silicon's, integrated ones included.
    pub fn gpu_bytes(&self) -> usize {
        self.renderers()
            .map(|(_, footprint)| footprint.gpu_bytes())
            .sum()
    }

    /// Nothing is open and nothing is held.
    pub fn is_empty(&self) -> bool {
        self.renderers().next().is_none() && self.inputs == 0
    }
}

#[derive(Default)]
struct Shared {
    /// A sketch is installed and the renderer is up.
    open: AtomicBool,
    /// A snippet is being previewed in the shelf.
    previewing: AtomicBool,
    /// Bumped on every preview start or stop, so a frame drawn for a
    /// snippet that has since been closed or replaced cannot land in the
    /// slot behind the clear - the same guard the theme has.
    preview_epoch: AtomicU64,
    /// The preview epoch whose frame last landed.
    preview_drawn_epoch: AtomicU64,
    /// A theme carries a sketch of its own. It shows whenever the score's
    /// picture does not.
    theme_installed: AtomicBool,
    /// Bumped on every theme install or removal, so a frame drawn for a
    /// sketch that has since been swapped or removed cannot land in the
    /// slot behind the change.
    theme_epoch: AtomicU64,
    /// The exact current theme epoch whose shader drew and whose selected
    /// output was read successfully. Camera acquisition waits for this
    /// acknowledgement instead of trusting syntax alone.
    theme_drawn_epoch: AtomicU64,
    /// The set is playing, so the picture should be on screen.
    drawing: AtomicBool,
    wants_tui: AtomicBool,
    wants_audio: AtomicBool,
    /// Delivery sizes, packed `width << 16 | height`.
    deliver_score: AtomicU32,
    deliver_preview: AtomicU32,
    smoothing: AtomicBool,
    tui: TuiSlot,
    frames: FrameSlot,
    preview: FrameSlot,
    theme: FrameSlot,
    /// Camera and image frames waiting to become Hydra source textures.
    inputs: [InputSlot; crate::HYDRA_SOURCE_SLOTS],
    /// Latest lookahead schedule behind `H(pattern)`.
    signals: Mutex<Option<SignalSchedule>>,
    /// What the engine is playing, for `detectAudio`.
    audio: Mutex<[f32; crate::HYDRA_AUDIO_BINS]>,
    /// Frames delivered, for the rate the log reports.
    delivered: AtomicU64,
    /// What the score's, the shelf's and the theme's renderers hold, in
    /// that order.
    renderers: [RendererGauge; 3],
}

impl Shared {
    fn memory(&self) -> HydraMemory {
        let [score, preview, theme] = &self.renderers;
        HydraMemory {
            score: score.read(),
            preview: preview.read(),
            theme: theme.read(),
            inputs: self
                .inputs
                .iter()
                .map(|input| input.bytes.load(Ordering::Relaxed))
                .sum(),
        }
    }
}

/// Clears the renderers' gauges when the worker ends, however it ends: its
/// renderers go with it.
struct GaugesClearedOnExit<'a>(&'a Shared);

impl Drop for GaugesClearedOnExit<'_> {
    fn drop(&mut self) {
        for gauge in &self.0.renderers {
            gauge.record(None);
        }
    }
}

/// The producer end of Hydra's external source textures.
///
/// A producer first calls [`Self::bind`] and gives the returned lease to the
/// image/camera worker. The lease is generation checked at publication time;
/// replacing or clearing the slot makes every older lease inert. Slots keep
/// only their newest complete RGBA frame.
#[derive(Clone)]
pub struct HydraInputSink {
    shared: Weak<Shared>,
}

impl HydraInputSink {
    /// Claim and clear `s{slot}`, returning a lease for this exact source
    /// request. Invalid slots are refused rather than aliased to `s3`.
    pub fn bind(&self, slot: u8) -> Option<HydraInputLease> {
        let shared = self.shared.upgrade()?;
        let input = shared.inputs.get(usize::from(slot))?;
        let generation = input.bind();
        Some(HydraInputLease {
            shared: Arc::downgrade(&shared),
            slot,
            generation,
        })
    }

    /// Invalidate the producer currently attached to `s{slot}` and make the
    /// GPU see an empty source on its next draw.
    pub fn clear(&self, slot: u8) {
        let Some(shared) = self.shared.upgrade() else {
            return;
        };
        if let Some(input) = shared.inputs.get(usize::from(slot)) {
            input.clear();
        }
    }
}

/// A generation-scoped right to publish frames to one Hydra source slot.
#[derive(Clone)]
pub struct HydraInputLease {
    shared: Weak<Shared>,
    slot: u8,
    generation: u64,
}

impl HydraInputLease {
    /// Invalidate and clear this source only if no newer producer has rebound
    /// its slot. This is the safe teardown primitive for asynchronous camera
    /// work: cancellation may arrive after an image has taken over the same
    /// `sN`, and must not erase that replacement.
    pub fn clear_if_current(&self) -> bool {
        let Some(shared) = self.shared.upgrade() else {
            return false;
        };
        let Some(input) = shared.inputs.get(usize::from(self.slot)) else {
            return false;
        };
        input.clear_generation(self.generation)
    }

    /// Replace the pending source frame. Returns `false` when this request was
    /// superseded (or the host closed); stale work is an ordinary race, not a
    /// renderer error.
    pub fn publish(&self, width: u32, height: u32, rgba: Vec<u8>) -> Result<bool, String> {
        // Match the decoder/acquisition boundary: four 2048² RGBA slots are
        // at most 64 MiB retained, even for a direct public-sink publisher.
        const MAX_EDGE: u32 = 2048;
        if width == 0 || height == 0 || width > MAX_EDGE || height > MAX_EDGE {
            return Err(format!(
                "Hydra source dimensions {width}x{height} are outside 1..={MAX_EDGE}"
            ));
        }
        let expected = usize::try_from(width)
            .ok()
            .and_then(|width| {
                usize::try_from(height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| "Hydra source dimensions overflow host memory".to_owned())?;
        if rgba.len() != expected {
            return Err(format!(
                "Hydra source {width}x{height} wants {expected} RGBA bytes, got {}",
                rgba.len()
            ));
        }
        let Some(shared) = self.shared.upgrade() else {
            return Ok(false);
        };
        let Some(input) = shared.inputs.get(usize::from(self.slot)) else {
            return Ok(false);
        };
        let Ok(mut held) = input.frame.lock() else {
            return Ok(false);
        };
        // Checked while holding the same lock a rebind clears under. An old
        // producer that passed a speculative check before the rebind cannot
        // sneak its frame in afterwards.
        if input.generation.load(Ordering::Acquire) != self.generation {
            return Ok(false);
        }
        input.bytes.store(rgba.len(), Ordering::Relaxed);
        *held = Some(InputFrame {
            width,
            height,
            rgba: Arc::from(rgba),
        });
        input.revision.fetch_add(1, Ordering::Release);
        Ok(true)
    }
}

/// Which of the host's pictures a [`HydraFrames`] handle reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Stream {
    Score,
    Preview,
    Theme,
}

/// The terminal's end of the picture: the newest frame the renderer drew.
#[derive(Clone)]
pub struct HydraFrames {
    shared: Arc<Shared>,
    stream: Stream,
}

impl HydraFrames {
    /// Whether there is a picture to draw right now.
    ///
    /// The score's picture needs an installed sketch and a playing set: a
    /// stopped set shows nothing. The shelf's picture needs only a previewed
    /// snippet, because browsing happens with the transport stopped. The
    /// theme's picture shows exactly when the score's picture does not,
    /// whether or not the transport runs.
    pub fn wanted(&self) -> bool {
        match self.stream {
            Stream::Preview => self.shared.previewing.load(Ordering::Relaxed),
            Stream::Score => {
                self.shared.open.load(Ordering::Relaxed)
                    && self.shared.drawing.load(Ordering::Relaxed)
            }
            Stream::Theme => {
                self.shared.theme_installed.load(Ordering::Relaxed)
                    && !(self.shared.open.load(Ordering::Relaxed)
                        && self.shared.drawing.load(Ordering::Relaxed))
            }
        }
    }

    /// The newest frame, if one arrived since it was last asked.
    pub fn take(&self) -> Option<(u16, u16, Vec<u8>)> {
        let slot = match self.stream {
            Stream::Preview => &self.shared.preview,
            Stream::Score => &self.shared.frames,
            Stream::Theme => &self.shared.theme,
        };
        slot.frame.lock().ok()?.take()
    }

    /// Number of score frames the renderer has completed successfully.
    ///
    /// This is cumulative for the host lifetime and deliberately excludes the
    /// independent snippet preview. Consumers can sample it off the renderer
    /// thread to prove that a live sketch kept advancing without retaining a
    /// frame backlog.
    pub fn delivered(&self) -> u64 {
        self.shared.delivered.load(Ordering::Relaxed)
    }

    /// The identity of the latest preview or theme install or removal.
    /// Score frames do not use sketch epochs.
    pub fn sketch_epoch(&self) -> Option<u64> {
        match self.stream {
            Stream::Preview => Some(self.shared.preview_epoch.load(Ordering::Acquire)),
            Stream::Theme => Some(self.shared.theme_epoch.load(Ordering::Acquire)),
            Stream::Score => None,
        }
    }

    /// Whether this exact installed sketch has completed a draw and readback.
    /// A replaced or removed sketch cannot acknowledge its replacement.
    pub fn sketch_drawn(&self, epoch: u64) -> bool {
        let (current, drawn, installed) = match self.stream {
            Stream::Preview => (
                &self.shared.preview_epoch,
                &self.shared.preview_drawn_epoch,
                &self.shared.previewing,
            ),
            Stream::Theme => (
                &self.shared.theme_epoch,
                &self.shared.theme_drawn_epoch,
                &self.shared.theme_installed,
            ),
            Stream::Score => return false,
        };
        // Drawn is read before current: a replacement that lands between
        // the two loads moves the current epoch, so the comparison fails
        // instead of certifying a sketch the replacement already ended.
        // Reading current first could pair this epoch with a drawn value
        // the replacement is one instruction away from clearing.
        epoch != 0
            && drawn.load(Ordering::Acquire) == epoch
            && current.load(Ordering::Acquire) == epoch
            && installed.load(Ordering::Acquire)
    }

    /// What the whole host holds - every picture's renderer, not only this
    /// stream's - without waiting on the renderer thread.
    pub fn memory(&self) -> HydraMemory {
        self.shared.memory()
    }
}

/// The half of the host that can be handed to the terminal thread.
#[derive(Clone)]
pub struct HydraTuiSink {
    shared: Arc<Shared>,
}

impl HydraTuiSink {
    /// Whether the sketch asked for `feedStrudel`.
    pub fn wanted(&self) -> bool {
        self.shared.wants_tui.load(Ordering::Relaxed)
    }

    /// Publish a frame, overwriting whatever has not been drawn yet: a texture
    /// wants the newest grid, never a backlog.
    pub fn publish(&self, frame: &HydraTuiFrame) -> Result<bool, String> {
        // Validate at the public boundary, before cloning an attacker-sized
        // cell vector into the latest-frame slot. The worker validates again
        // before its own fallible pixel allocation.
        crate::native::validate_tui_frame(frame, TUI_CELL_PIXELS)?;
        if !self.wanted() {
            return Ok(false);
        }
        let mut slot = self
            .shared
            .tui
            .grid
            .lock()
            .map_err(|_| "Hydra terminal frame slot is unavailable".to_owned())?;
        *slot = Some(frame.clone());
        self.shared.tui.sequence.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }
}

/// What the worker is told to do.
enum Command {
    /// Draw this, or nothing.
    Apply(Box<Option<HydraProgram>>),
    /// Show a snippet in the shelf, or stop.
    Preview {
        code: Option<String>,
        epoch: u64,
    },
    /// The theme's own sketch, or none. Furniture: drawn whenever the
    /// score's picture is not.
    Theme {
        code: Option<String>,
        epoch: u64,
    },
    Exit,
}

struct Running {
    commands: mpsc::Sender<Command>,
    handle: Option<std::thread::JoinHandle<()>>,
}

/// A handle to the renderer, or to the absence of one.
///
/// Creating a host costs nothing and starts nothing. The first sketch starts
/// the thread.
pub struct HydraHost {
    running: Option<Running>,
    shared: Arc<Shared>,
    events: mpsc::Receiver<HydraEvent>,
    sender: mpsc::Sender<HydraEvent>,
}

impl Default for HydraHost {
    fn default() -> Self {
        Self::new()
    }
}

impl HydraHost {
    pub fn new() -> Self {
        let (sender, events) = mpsc::channel();
        Self {
            running: None,
            shared: Arc::new(Shared::default()),
            events,
            sender,
        }
    }

    pub fn is_open(&self) -> bool {
        self.shared.open.load(Ordering::Relaxed)
    }

    pub fn wants_tui(&self) -> bool {
        self.shared.wants_tui.load(Ordering::Relaxed)
    }

    pub fn wants_audio(&self) -> bool {
        // The shelf's thumbnail always wants it: an audio-reactive snippet
        // that previews flat is indistinguishable from a broken one. The
        // theme's sketch is on screen between scores, so it breathes too.
        self.shared.wants_audio.load(Ordering::Relaxed)
            || self.shared.previewing.load(Ordering::Relaxed)
            || self.shared.theme_installed.load(Ordering::Relaxed)
    }

    /// Tell the renderer whether the set is playing. A stopped set wipes the
    /// picture rather than leaving the last frame on screen.
    pub fn set_drawing(&self, drawing: bool) {
        self.shared.drawing.store(drawing, Ordering::Relaxed);
        if !drawing && let Ok(mut slot) = self.shared.frames.frame.lock() {
            slot.take();
        }
    }

    /// Install what a score asked for. An empty program stops drawing.
    pub fn apply(&mut self, program: HydraProgram) -> Result<(), HydraHostError> {
        if program.is_empty() {
            self.shared.open.store(false, Ordering::Relaxed);
            self.shared.wants_tui.store(false, Ordering::Relaxed);
            self.shared.wants_audio.store(false, Ordering::Relaxed);
            if let Some(running) = &self.running {
                let _ = running.commands.send(Command::Apply(Box::new(None)));
            }
            if let Ok(mut slot) = self.shared.frames.frame.lock() {
                slot.take();
            }
            return Ok(());
        }
        program
            .validate()
            .map_err(|error| HydraHostError::Program(error.to_string()))?;
        self.shared
            .wants_tui
            .store(program.options.feed_strudel, Ordering::Relaxed);
        self.shared
            .wants_audio
            .store(program.options.detect_audio, Ordering::Relaxed);
        self.start()?;
        self.shared.open.store(true, Ordering::Relaxed);
        if let Some(running) = &self.running {
            let _ = running
                .commands
                .send(Command::Apply(Box::new(Some(program))));
        }
        Ok(())
    }

    /// Refresh the values behind `H(pattern)`.
    pub fn signals(&self, frame: &HydraSignalFrame) {
        if let Ok(mut slot) = self.shared.signals.lock() {
            *slot = Some(SignalSchedule::from_frame(frame, Instant::now()));
        }
    }

    /// Publish what the engine is playing, for `detectAudio`.
    pub fn audio(&self, frame: &HydraAudioFrame) {
        if let Ok(mut slot) = self.shared.audio.lock() {
            // The browser analyser smooths between frames
            // (smoothingTimeConstant), and sketches are written against
            // that easing; raw per-frame bins flicker.
            for (held, next) in slot.iter_mut().zip(frame.bins) {
                // hydra-synth defaults `smooth` to 0.4 and weights the
                // previous frame by that value, not its complement.
                *held = *held * 0.4 + next * 0.6;
            }
        }
    }

    pub fn frames(&self) -> HydraFrames {
        HydraFrames {
            shared: self.shared.clone(),
            stream: Stream::Score,
        }
    }

    pub fn preview_frames(&self) -> HydraFrames {
        HydraFrames {
            shared: self.shared.clone(),
            stream: Stream::Preview,
        }
    }

    /// A handle for the theme's own picture. It shows behind the code
    /// whenever the score's picture does not.
    pub fn theme_frames(&self) -> HydraFrames {
        HydraFrames {
            shared: self.shared.clone(),
            stream: Stream::Theme,
        }
    }

    /// The identity assigned to the most recent theme install/removal.
    pub fn theme_epoch(&self) -> u64 {
        self.shared.theme_epoch.load(Ordering::Acquire)
    }

    /// Whether this exact installed theme has completed a visible GPU draw
    /// and selected-output read. A stale renderer can never acknowledge a
    /// later theme because epochs are compared exactly.
    pub fn theme_drawn(&self, epoch: u64) -> bool {
        // Drawn before current, as `sketch_drawn` reads it: a replacement
        // that lands between the loads moves the current epoch and the
        // comparison fails instead of certifying the old theme.
        epoch != 0
            && self.shared.theme_drawn_epoch.load(Ordering::Acquire) == epoch
            && self.shared.theme_epoch.load(Ordering::Acquire) == epoch
            && self.shared.theme_installed.load(Ordering::Acquire)
    }

    pub fn set_frame_size(&self, width: u16, height: u16) {
        self.shared.deliver_score.store(
            (u32::from(width) << 16) | u32::from(height),
            Ordering::Relaxed,
        );
    }

    pub fn set_preview_frame_size(&self, width: u16, height: u16) {
        self.shared.deliver_preview.store(
            (u32::from(width) << 16) | u32::from(height),
            Ordering::Relaxed,
        );
    }

    /// Whether scaling to the delivery size averages the source pixels or
    /// picks one. Neither is faster on the GPU; it is how it should look.
    pub fn set_smoothing(&self, smoothing: bool) {
        self.shared.smoothing.store(smoothing, Ordering::Relaxed);
    }

    /// Show a snippet in the shelf's own picture, or `None` to stop.
    pub fn preview(&mut self, code: Option<&str>) -> Result<(), HydraHostError> {
        let wanted = code.is_some();
        if wanted {
            self.start()?;
        }
        self.shared.previewing.store(wanted, Ordering::Relaxed);
        let epoch = self.shared.preview_epoch.fetch_add(1, Ordering::AcqRel) + 1;
        self.shared.preview_drawn_epoch.store(0, Ordering::Release);
        // Synchronize with the renderer's final current-epoch check: a frame
        // from the snippet just closed cannot reappear after this clear.
        if let Ok(mut slot) = self.shared.preview.frame.lock() {
            slot.take();
        }
        if let Some(running) = &self.running {
            let _ = running.commands.send(Command::Preview {
                code: code.map(str::to_owned),
                epoch,
            });
        }
        Ok(())
    }

    /// Install, replace or clear the theme's own sketch.
    ///
    /// One chain, the shelf-snippet grammar, no `initHydra` anywhere - the
    /// theme is not a score. Parsed on the render thread; a sketch that does
    /// not parse reports through the same events a bad snippet does.
    pub fn theme(&mut self, code: Option<&str>) -> Result<(), HydraHostError> {
        let wanted = code.is_some();
        if wanted {
            self.start()?;
        }
        self.shared.theme_installed.store(wanted, Ordering::Release);
        let epoch = self.shared.theme_epoch.fetch_add(1, Ordering::AcqRel) + 1;
        self.shared.theme_drawn_epoch.store(0, Ordering::Release);
        // Synchronize with the renderer's final current-epoch check. A frame
        // from the old theme cannot reappear after this clear.
        if let Ok(mut slot) = self.shared.theme.frame.lock() {
            slot.take();
        }
        if let Some(running) = &self.running {
            let _ = running.commands.send(Command::Theme {
                code: code.map(str::to_owned),
                epoch,
            });
        }
        Ok(())
    }

    pub fn tui_sink(&self) -> HydraTuiSink {
        HydraTuiSink {
            shared: self.shared.clone(),
        }
    }

    /// A latest-frame producer for camera and web-image source textures.
    /// Taking the handle does not start the renderer or any input device.
    pub fn input_sink(&self) -> HydraInputSink {
        HydraInputSink {
            shared: Arc::downgrade(&self.shared),
        }
    }

    pub fn take_events(&self) -> Vec<HydraEvent> {
        self.events.try_iter().collect()
    }

    /// What the renderers and the inputs hold, without waiting on the
    /// renderer thread.
    pub fn memory(&self) -> HydraMemory {
        self.shared.memory()
    }

    pub fn close(&mut self) {
        self.shared.open.store(false, Ordering::Relaxed);
        self.shared.previewing.store(false, Ordering::Relaxed);
        for input in &self.shared.inputs {
            input.clear();
        }
        if let Some(mut running) = self.running.take() {
            let _ = running.commands.send(Command::Exit);
            if let Some(handle) = running.handle.take() {
                let _ = handle.join();
            }
        }
    }

    /// Start the worker, if it is not already up.
    fn start(&mut self) -> Result<(), HydraHostError> {
        if self.running.is_some() {
            return Ok(());
        }
        let (commands, orders) = mpsc::channel();
        let shared = self.shared.clone();
        let events = self.sender.clone();
        let handle = std::thread::Builder::new()
            .name("rustel-hydra".into())
            .spawn(move || worker(&shared, &events, &orders))
            .map_err(|error| HydraHostError::Thread(error.to_string()))?;
        self.running = Some(Running {
            commands,
            handle: Some(handle),
        });
        Ok(())
    }
}

impl Drop for HydraHost {
    fn drop(&mut self) {
        self.close();
    }
}

/// How many pixels a character cell becomes when the terminal is fed to a
/// sketch.
///
/// Four is enough for a glyph to read as a shape rather than a blob, and keeps
/// a 200x50 grid at 800x200 - smaller than the render it is sampled by.
const TUI_CELL_PIXELS: usize = 4;

/// The render budget, in fragments, before the delivery size is taken into
/// account.
///
/// The GPU draws straight at the size the terminal reads, so smoothing means
/// supersampling: render larger, average down, and lose the crawl that point
/// sampling gives moving detail. A coarse delivery gets a large factor from
/// this budget, and a near-native one gains nothing.
const TARGET_RENDER_PIXELS: u64 = 1_000_000;

/// How many render fragments to spend per delivered pixel.
///
/// Smoothing off still renders above the delivery: the factor exists to give
/// `osc`, `noise`, `voronoi` and the feedback chains enough fragments to be
/// themselves, and averaging that down is what the setting chooses about.
fn render_factor(width: u32, height: u32, shared: &Shared) -> u32 {
    let base = u64::from(width.max(1)) * u64::from(height.max(1));
    let ceiling = if shared.smoothing.load(Ordering::Relaxed) {
        8
    } else {
        4
    };
    (TARGET_RENDER_PIXELS / base.max(1))
        .isqrt()
        .clamp(1, ceiling) as u32
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FrameGeometry {
    width: u32,
    height: u32,
    render_width: u32,
    render_height: u32,
    factor: u32,
}

/// Bound both the delivered frame and its optional supersampled render before
/// either reaches a GPU allocation. Supersampling is a quality hint: when the
/// 3x surface would exceed the protocol budget, rendering at 1x is the safe
/// and useful fallback.
///
/// `factor` is what the caller would LIKE to render at, per delivered pixel.
/// It steps down until the render fits the edge and area limits rather than
/// falling straight back to 1, because the whole point is to render the sketch
/// at a sane resolution even when the delivery is a handful of terminal cells:
/// a 200x50 cell backdrop asking for 1 would evaluate `osc`, `noise` and every
/// feedback chain at ten thousand fragments for the entire picture, which is
/// what made the source signal itself look broken rather than merely
/// downscaled.
fn frame_geometry(width: u32, height: u32, factor: u32) -> Result<FrameGeometry, String> {
    let (width, height) = (width.max(1), height.max(1));
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| format!("Hydra frame geometry {width}x{height} overflows"))?;
    if width > crate::HYDRA_MAX_OUTPUT_EDGE
        || height > crate::HYDRA_MAX_OUTPUT_EDGE
        || pixels > crate::HYDRA_MAX_OUTPUT_PIXELS
    {
        return Err(format!(
            "Hydra frame {width}x{height} exceeds the {}-pixel edge / {}-pixel area limit",
            crate::HYDRA_MAX_OUTPUT_EDGE,
            crate::HYDRA_MAX_OUTPUT_PIXELS
        ));
    }

    // Step down rather than give up: a factor that does not fit is not a
    // reason to render at one fragment per delivered pixel.
    let (render_width, render_height, factor) = (1..=factor.max(1))
        .rev()
        .find_map(|factor| {
            let render_width = width.checked_mul(factor)?;
            let render_height = height.checked_mul(factor)?;
            let render_pixels = u64::from(render_width).checked_mul(u64::from(render_height))?;
            (render_width <= crate::HYDRA_MAX_OUTPUT_EDGE
                && render_height <= crate::HYDRA_MAX_OUTPUT_EDGE
                && render_pixels <= crate::HYDRA_MAX_OUTPUT_PIXELS)
                .then_some((render_width, render_height, factor))
        })
        .unwrap_or((width, height, 1));
    Ok(FrameGeometry {
        width,
        height,
        render_width,
        render_height,
        factor,
    })
}

/// Reduce an RGBA frame by an integer factor: take one pixel out of each
/// box instead of averaging the box.
///
/// The render is larger than the delivery either way. That gives `osc`,
/// `noise` and the feedback chains enough fragments to resolve their detail.
/// This is the reduction for smoothing off, and it is not only the cheaper
/// one: `downsample` averages the colour and writes an opaque alpha, so the
/// shape that a `luma` mask cut is lost. Point sampling keeps every
/// delivered pixel a value the sketch produced, alpha included.
fn point_sample(pixels: &[u8], width: u32, height: u32, factor: u32) -> Result<Vec<u8>, String> {
    if factor == 0
        || width == 0
        || height == 0
        || !width.is_multiple_of(factor)
        || !height.is_multiple_of(factor)
    {
        return Err(format!(
            "cannot point-sample {width}x{height} RGBA by factor {factor}"
        ));
    }
    let expected = usize::try_from(u64::from(width) * u64::from(height) * 4)
        .map_err(|_| format!("{width}x{height} RGBA byte count overflows host memory"))?;
    if pixels.len() != expected {
        return Err(format!(
            "cannot point-sample {width}x{height}: expected {expected} RGBA bytes, got {}",
            pixels.len()
        ));
    }
    let (out_w, out_h) = (width / factor, height / factor);
    let mut out = Vec::with_capacity(out_w as usize * out_h as usize * 4);
    // The box's centre, so the sample is not biased to its top-left corner.
    let centre = factor / 2;
    for row in 0..out_h {
        for column in 0..out_w {
            let y = (row * factor + centre).min(height - 1);
            let x = (column * factor + centre).min(width - 1);
            let at = (y as usize * width as usize + x as usize) * 4;
            out.extend_from_slice(&pixels[at..at + 4]);
        }
    }
    Ok(out)
}

/// Average an RGBA frame down by an integer factor, box by box. The colour
/// channels are averaged and the result is opaque.
fn downsample(pixels: &[u8], width: u32, height: u32, factor: u32) -> Result<Vec<u8>, String> {
    if factor == 0
        || width == 0
        || height == 0
        || !width.is_multiple_of(factor)
        || !height.is_multiple_of(factor)
    {
        return Err(format!(
            "cannot downsample {width}x{height} RGBA by factor {factor}"
        ));
    }
    let expected = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(4))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| format!("{width}x{height} RGBA byte count overflows host memory"))?;
    if pixels.len() != expected {
        return Err(format!(
            "cannot downsample {width}x{height}: expected {expected} RGBA bytes, got {}",
            pixels.len()
        ));
    }
    let (out_w, out_h) = (width / factor, height / factor);
    let output_bytes = u64::from(out_w)
        .checked_mul(u64::from(out_h))
        .and_then(|pixels| pixels.checked_mul(4))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| format!("{out_w}x{out_h} RGBA byte count overflows host memory"))?;
    let mut out = Vec::new();
    out.try_reserve_exact(output_bytes)
        .map_err(|error| format!("could not reserve {output_bytes} downsample bytes: {error}"))?;
    let width = usize::try_from(width).map_err(|_| "frame width does not fit host memory")?;
    let factor = usize::try_from(factor).map_err(|_| "sample factor does not fit host memory")?;
    for row in 0..out_h {
        for column in 0..out_w {
            let mut sums = [0u32; 3];
            for y in 0..factor {
                let line = (row as usize * factor + y) * width;
                for x in 0..factor {
                    let at = (line + column as usize * factor + x) * 4;
                    for (sum, channel) in sums.iter_mut().zip(&pixels[at..at + 3]) {
                        *sum += u32::from(*channel);
                    }
                }
            }
            let count = u32::try_from(factor * factor)
                .map_err(|_| "sample factor squared does not fit u32")?;
            out.extend(sums.iter().map(|sum| (sum / count) as u8));
            out.push(255);
        }
    }
    debug_assert_eq!(out.len(), output_bytes);
    Ok(out)
}

/// How large the shelf's thumbnail is when the terminal has not said.
/// Square, because the box it lands in is.
const PREVIEW_EDGE: u32 = 320;

/// How often the worker draws when there is something to draw.
const FRAME_INTERVAL: Duration =
    Duration::from_nanos(1_000_000_000 / crate::HYDRA_FRAMES_PER_SECOND as u64);

/// Step every array argument at `bpm` instead of Hydra's default 30. An
/// array's index runs at `time * speed * bpm / 60`, and the composer writes
/// it for 30 bpm, so a list's own speed carries the ratio.
fn step_arrays_at(program: &mut HydraProgram, bpm: f64) {
    fn walk(node: &mut HydraNode, ratio: f64) {
        match node {
            HydraNode::List { v, speed, .. } => {
                *speed *= ratio;
                v.iter_mut().for_each(|entry| walk(entry, ratio));
            }
            HydraNode::Chain { args, calls, .. } => {
                args.iter_mut().for_each(|arg| walk(arg, ratio));
                calls
                    .iter_mut()
                    .flat_map(|call| call.args.iter_mut())
                    .for_each(|arg| walk(arg, ratio));
            }
            _ => {}
        }
    }
    if bpm == 30.0 {
        return;
    }
    for statement in &mut program.statements {
        if let HydraStatement::Evaluate { node } = statement {
            walk(node, bpm / 30.0);
        }
    }
}

/// When a sketch with `fps` draws after a draw at `now`: one period after the
/// last planned draw, so the rate holds on average. A draw more than a
/// period late, after a pause, plans from `now` instead of catching up.
fn next_fps_draw(planned: Option<Instant>, now: Instant, fps: f64) -> Instant {
    let period = Duration::from_secs_f64(1.0 / fps);
    let next = planned.unwrap_or(now) + period;
    if next < now { now + period } else { next }
}

/// The pace this frame runs at: the score program's own fps when one is
/// installed, and the default interval when none is, so a preview or a
/// theme alone keeps the usual cadence.
fn frame_interval(score_fps: Option<u32>) -> Duration {
    match score_fps {
        Some(fps) if fps > 0 => Duration::from_nanos(1_000_000_000 / u64::from(fps)),
        _ => FRAME_INTERVAL,
    }
}

/// Sends each distinct failure message once. The sent messages are forgotten
/// after a frame in which nothing failed, and on `rearm`.
struct SaidOnce<'a> {
    events: &'a mpsc::Sender<HydraEvent>,
    /// Messages sent since they were last forgotten.
    said: Vec<String>,
    /// Whether `failed` was called since the last `end_frame`.
    failed_this_frame: bool,
}

#[derive(Clone, Copy)]
struct SourceUploadState {
    revision: u64,
    owned: bool,
}

impl Default for SourceUploadState {
    fn default() -> Self {
        Self {
            // Every fresh renderer must receive either the held frame or an
            // explicit clear before its first shader samples this slot.
            revision: u64::MAX,
            owned: false,
        }
    }
}

impl<'a> SaidOnce<'a> {
    fn new(events: &'a mpsc::Sender<HydraEvent>) -> Self {
        Self {
            events,
            said: Vec::new(),
            failed_this_frame: false,
        }
    }

    fn failed(&mut self, message: String) {
        self.failed_this_frame = true;
        if self.said.contains(&message) {
            return;
        }
        let _ = self.events.send(HydraEvent::Failed {
            message: message.clone(),
        });
        self.said.push(message);
    }

    /// Ends a frame, forgetting the sent messages if nothing failed in it.
    fn end_frame(&mut self) {
        if !std::mem::take(&mut self.failed_this_frame) {
            self.said.clear();
        }
    }

    /// Forgets the sent messages, so a new program reports its failures.
    fn rearm(&mut self) {
        self.said.clear();
    }
}

/// The renderer thread.
fn worker(
    shared: &Arc<Shared>,
    events: &mpsc::Sender<HydraEvent>,
    orders: &mpsc::Receiver<Command>,
) {
    // Before the renderers, so it is dropped after them.
    let _gauges = GaugesClearedOnExit(shared);
    let mut score: Option<HydraProgram> = None;
    // The score's own `time`: it runs at the sketch's `speed` and restarts
    // where the sketch sets `time`.
    let mut clock = crate::HydraClock::default();
    let mut score_time = 0.0f64;
    let mut ticked = Instant::now();
    // When `fps` draws the score next. Each draw moves it one period on, so
    // the worker's own ticks round single frames but not the rate.
    let mut next_draw: Option<Instant> = None;
    let mut preview: Option<(HydraNode, u64)> = None;
    // Keep the command's epoch attached to its parsed node. Reading the
    // shared current epoch beside an older local node would let that old
    // shader acknowledge a replacement command that has not been consumed.
    let mut theme: Option<(HydraNode, u64)> = None;
    let mut renderer: Option<NativeRenderer> = None;
    let mut previewer: Option<NativeRenderer> = None;
    let mut themer: Option<NativeRenderer> = None;
    let mut source_uploads = [SourceUploadState::default(); crate::HYDRA_SOURCE_SLOTS];
    let mut theme_source_uploads = [SourceUploadState::default(); crate::HYDRA_SOURCE_SLOTS];
    let mut score_changed = false;
    // When a stream goes quiet, its renderer (device, textures, pipeline
    // cache) is dropped after a short grace and rebuilt in one frame if the
    // stream comes back. This frees the memory that an idle renderer holds.
    const RENDERER_LINGER: Duration = Duration::from_secs(5);
    let mut renderer_used = Instant::now();
    let mut previewer_used = Instant::now();
    let mut themer_used = Instant::now();
    let started = Instant::now();
    let mut counted = Instant::now();
    let mut frames = 0u32;
    let mut once = SaidOnce::new(events);
    let mut preview_once = SaidOnce::new(events);
    let mut theme_once = SaidOnce::new(events);

    loop {
        // Orders first, so a new sketch is never a frame late.
        loop {
            match orders.try_recv() {
                Ok(Command::Exit) | Err(mpsc::TryRecvError::Disconnected) => return,
                Ok(Command::Apply(program)) => {
                    score = *program;
                    clock = score.as_ref().map(HydraProgram::clock).unwrap_or_default();
                    if let Some(time) = clock.time {
                        // From now: the time this turn already spent does
                        // not count against the restarted clock.
                        score_time = time;
                        ticked = Instant::now();
                    }
                    if let Some(program) = score.as_mut() {
                        step_arrays_at(program, clock.bpm);
                    }
                    next_draw = None;
                    score_changed = true;
                    once.rearm();
                }
                Ok(Command::Preview { code, epoch }) => {
                    preview_once.rearm();
                    preview = match code {
                        // A snippet is text, not a recording: nothing evaluated
                        // it. It is read here into the same nodes a score
                        // would have produced.
                        Some(code) => match crate::glsl::parse_chain(&code) {
                            Ok(node) => Some((node, epoch)),
                            Err(error) => {
                                let _ = events.send(HydraEvent::Failed {
                                    message: format!("snippet: {error}"),
                                });
                                None
                            }
                        },
                        None => None,
                    };
                }
                Ok(Command::Theme { code, epoch }) => {
                    theme_once.rearm();
                    theme = match code {
                        Some(code) => match crate::glsl::parse_chain(&code) {
                            Ok(node) => Some((node, epoch)),
                            Err(error) => {
                                let _ = events.send(HydraEvent::Failed {
                                    message: format!("theme: {error}"),
                                });
                                None
                            }
                        },
                        None => None,
                    };
                    // The worker is the last writer in the drain, so it must
                    // state both outcomes: a stale None followed by a good
                    // sketch would otherwise leave the flag false and the
                    // held sketch dark for good.
                    shared
                        .theme_installed
                        .store(theme.is_some(), Ordering::Relaxed);
                    if theme.is_none()
                        && let Ok(mut slot) = shared.theme.frame.lock()
                    {
                        slot.take();
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
            }
        }

        let time = started.elapsed().as_secs_f32();
        let now = Instant::now();
        score_time += now.saturating_duration_since(ticked).as_secs_f64() * clock.speed;
        ticked = now;
        // `fps` draws less often than the host asks. The last frame stays.
        let due = next_draw.is_none_or(|next| now >= next);
        let drawing = shared.open.load(Ordering::Relaxed) && shared.drawing.load(Ordering::Relaxed);
        if drawing && score.is_some() {
            renderer_used = Instant::now();
        }
        if drawing
            && due
            && let Some(program) = &score
        {
            if let Some(fps) = clock.fps {
                next_draw = Some(next_fps_draw(next_draw, now, fps));
            }
            draw(
                &mut renderer,
                program,
                shared,
                events,
                &mut once,
                score_time as f32,
                &shared.deliver_score,
                &shared.frames,
                true,
                &mut source_uploads,
                score_changed,
            );
            once.end_frame();
            score_changed = false;
            frames += 1;
        }
        if shared.previewing.load(Ordering::Relaxed)
            && let Some((node, epoch)) = &preview
        {
            previewer_used = Instant::now();
            draw_one(
                &mut previewer,
                node,
                shared,
                &mut preview_once,
                time,
                &shared.deliver_preview,
                &shared.preview,
                Some((&shared.preview_epoch, &shared.preview_drawn_epoch, *epoch)),
                None,
            );
            preview_once.end_frame();
        }
        // The theme's picture is furniture: it fills the screen whenever the
        // score's picture is not on it, transport or no transport, at the
        // same delivery size the backdrop reads.
        if !drawing
            && shared.theme_installed.load(Ordering::Relaxed)
            && let Some((node, epoch)) = &theme
        {
            themer_used = Instant::now();
            draw_one(
                &mut themer,
                node,
                shared,
                &mut theme_once,
                time,
                &shared.deliver_score,
                &shared.theme,
                Some((&shared.theme_epoch, &shared.theme_drawn_epoch, *epoch)),
                Some(&mut theme_source_uploads),
            );
            theme_once.end_frame();
        }

        if renderer.is_some() && renderer_used.elapsed() >= RENDERER_LINGER {
            renderer = None;
        }
        if previewer.is_some() && previewer_used.elapsed() >= RENDERER_LINGER {
            previewer = None;
        }
        if themer.is_some() && themer_used.elapsed() >= RENDERER_LINGER {
            themer = None;
        }
        // The driver keeps residue per compiled shader that only device
        // teardown returns; past a threshold the renderer is retired and
        // rebuilt on its next frame. Generating snippet after snippet used
        // to grow the process forever.
        const SHADERS_PER_DEVICE: usize = 48;
        for slot in [&mut renderer, &mut previewer, &mut themer] {
            if slot
                .as_ref()
                .is_some_and(|open| open.shaders_built() >= SHADERS_PER_DEVICE)
            {
                *slot = None;
            }
        }
        // After everything that opens, resizes or retires a renderer this
        // frame, so what the breakdown reads is what is held.
        for (gauge, slot) in shared
            .renderers
            .iter()
            .zip([&renderer, &previewer, &themer])
        {
            gauge.record(slot.as_ref().map(NativeRenderer::footprint));
        }
        if counted.elapsed() >= Duration::from_secs(2) {
            let rate = frames / counted.elapsed().as_secs().max(1) as u32;
            let _ = events.send(HydraEvent::Frames { fps: rate });
            frames = 0;
            counted = Instant::now();
        }
        std::thread::sleep(frame_interval(score.as_ref().map(|p| p.options.fps)));
    }
}

/// Draw one program into one slot, opening a renderer for it if needed.
#[allow(clippy::too_many_arguments)]
fn draw(
    renderer: &mut Option<NativeRenderer>,
    program: &HydraProgram,
    shared: &Arc<Shared>,
    events: &mpsc::Sender<HydraEvent>,
    once: &mut SaidOnce<'_>,
    time: f32,
    size: &AtomicU32,
    slot: &FrameSlot,
    announce: bool,
    source_uploads: &mut [SourceUploadState; crate::HYDRA_SOURCE_SLOTS],
    score_changed: bool,
) {
    let packed = size.load(Ordering::Relaxed);
    let (width, height) = if packed == 0 {
        (program.options.width, program.options.height)
    } else {
        (packed >> 16, packed & 0xffff)
    };
    let geometry = match frame_geometry(width, height, render_factor(width, height, shared)) {
        Ok(geometry) => geometry,
        Err(message) => {
            once.failed(message);
            return;
        }
    };

    let opening = renderer.is_none();
    if opening {
        match NativeRenderer::new(geometry.render_width, geometry.render_height) {
            Ok(open) => {
                if announce {
                    let _ = events.send(HydraEvent::Ready {
                        renderer: open.adapter().to_owned(),
                    });
                }
                *renderer = Some(open);
            }
            Err(error) => {
                once.failed(error.to_string());
                return;
            }
        }
    }
    let Some(open) = renderer.as_mut() else {
        return;
    };
    if opening {
        source_uploads.fill(SourceUploadState::default());
    }
    // A size change rebuilds textures, not the device.
    if let Err(error) = open.resize(geometry.render_width, geometry.render_height) {
        once.failed(error.to_string());
        return;
    }
    if score_changed && program_has_hush(program) {
        // `hush()` is an evaluation-time reset, not a per-frame transform.
        // Clear once when the program arrives, then let any later `.out()`
        // statements establish their normal feedback across future frames.
        open.hush();
    }

    if let Ok(schedule) = shared.signals.lock() {
        let values = schedule
            .as_ref()
            .map(|schedule| schedule.values_at(Instant::now()))
            .unwrap_or_default();
        open.set_signals(&values);
    }
    if let Ok(bands) = shared.audio.lock() {
        open.set_audio(score_audio(program, *bands));
    }

    let external = configured_source_slots(program);
    sync_input_sources(open, shared, &external, source_uploads, once);

    // `feedStrudel`: the terminal becomes the texture `s0` samples.
    // An explicit `s0.initCam()` / `s0.initImage()` owns s0, just as it does
    // in hydra-synth; feedStrudel may not overwrite it behind the score's back.
    if feed_strudel_owns_s0(program, &external)
        && let Ok(grid) = shared.tui.grid.lock()
        && let Some(frame) = grid.as_ref()
    {
        match crate::native::rasterize(frame, TUI_CELL_PIXELS) {
            Ok((w, h, rgba)) => {
                if let Err(error) = open.set_source(0, w, h, &rgba) {
                    once.failed(error.to_string());
                }
            }
            Err(message) => once.failed(message),
        }
    }

    // A statement that will not compile costs its own layer, not the screen:
    // the frame is still read back, so the rest of the sketch is drawn and the
    // terminal is never left waiting on a frame that is not coming.
    let chains = match chains_of(program) {
        Ok(chains) => chains,
        Err(message) => {
            once.failed(message);
            Vec::new()
        }
    };
    let readback = if program.statements.is_empty() {
        chains.first().map_or(CanvasSelection::Output(0), |node| {
            CanvasSelection::Output(single_chain_output(node))
        })
    } else {
        selected_output(program)
    };
    for node in chains {
        if let Err(error) = open.draw(&node, time) {
            once.failed(error.to_string());
        }
    }
    run_commands(open, program);

    let pixels = match readback {
        CanvasSelection::Output(output) => open.read(output),
        CanvasSelection::Display => open.read_display(),
    };
    match pixels {
        Ok(pixels) => deliver(pixels, geometry, shared, slot, once),
        Err(error) => once.failed(error.to_string()),
    }
}

/// Reduces a supersampled render to its delivery size. A factor of 1 passes
/// the render through.
fn reduce_to_delivery(
    pixels: Vec<u8>,
    geometry: FrameGeometry,
    smoothing: bool,
) -> Result<Vec<u8>, String> {
    if geometry.factor <= 1 {
        return Ok(pixels);
    }
    // Averaging steadies shimmer; point sampling keeps the sketch's own
    // values, alpha included.
    let reduce = if smoothing { downsample } else { point_sample };
    reduce(
        &pixels,
        geometry.render_width,
        geometry.render_height,
        geometry.factor,
    )
}

/// Reduces a score frame the renderer read back, counts it as delivered and
/// holds it. A frame that will not reduce is reported through `once`, and is
/// neither counted nor held.
fn deliver(
    pixels: Vec<u8>,
    geometry: FrameGeometry,
    shared: &Shared,
    slot: &FrameSlot,
    once: &mut SaidOnce<'_>,
) {
    let smoothing = shared.smoothing.load(Ordering::Relaxed);
    let pixels = match reduce_to_delivery(pixels, geometry, smoothing) {
        Ok(pixels) => pixels,
        Err(message) => {
            once.failed(message);
            return;
        }
    };
    shared.delivered.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut held) = slot.frame.lock() {
        // `set_drawing(false)` clears this slot while a GPU readback
        // may already be in flight. Recheck visibility under the same
        // lock used by that clear: either this frame is rejected, or
        // the stop waits for this write and clears it afterwards.
        if shared.open.load(Ordering::Relaxed) && shared.drawing.load(Ordering::Relaxed) {
            *held = Some((geometry.width as u16, geometry.height as u16, pixels));
        }
    }
}

fn score_audio(
    program: &HydraProgram,
    bands: [f32; crate::HYDRA_AUDIO_BINS],
) -> [f32; crate::HYDRA_AUDIO_BINS] {
    if program.options.detect_audio {
        bands
    } else {
        [0.0; crate::HYDRA_AUDIO_BINS]
    }
}

/// Source slots explicitly configured by the recording. There is no implicit
/// inheritance between evaluations: deleting `s0.initCam()` gives s0 back to
/// feedStrudel (or to the renderer's empty texture) on the next program.
fn configured_source_slots(program: &HydraProgram) -> [bool; crate::HYDRA_SOURCE_SLOTS] {
    let mut configured = [false; crate::HYDRA_SOURCE_SLOTS];
    for statement in &program.statements {
        match statement {
            HydraStatement::ConfigureSource { slot, .. } => {
                if let Some(owned) = configured.get_mut(usize::from(*slot)) {
                    *owned = true;
                }
            }
            HydraStatement::ClearSource { slot } => {
                if let Some(owned) = configured.get_mut(usize::from(*slot)) {
                    *owned = false;
                }
            }
            HydraStatement::Evaluate { node } if matches!(command_of(node), Some(("hush", []))) => {
                configured.fill(false);
            }
            _ => {}
        }
    }
    configured
}

fn feed_strudel_owns_s0(
    program: &HydraProgram,
    configured: &[bool; crate::HYDRA_SOURCE_SLOTS],
) -> bool {
    program.options.feed_strudel && !configured[0]
}

/// Uploads each configured slot's newest frame and clears each slot the
/// program released. A slot whose upload fails is retried every frame, so its
/// failure lasts until it uploads or its source changes.
fn sync_input_sources(
    open: &mut NativeRenderer,
    shared: &Shared,
    configured: &[bool; crate::HYDRA_SOURCE_SLOTS],
    uploads: &mut [SourceUploadState; crate::HYDRA_SOURCE_SLOTS],
    once: &mut SaidOnce<'_>,
) {
    for (index, ((input, configured), uploaded)) in shared
        .inputs
        .iter()
        .zip(configured)
        .zip(uploads.iter_mut())
        .enumerate()
    {
        let (revision, frame) = input.snapshot();
        if !*configured {
            // A revision change can also retire feedStrudel's s0: that path
            // uploads directly and therefore never marks `owned`.
            if source_needs_clear(*configured, revision, *uploaded)
                && let Err(error) = open.clear_source(index)
            {
                once.failed(error.to_string());
                continue;
            }
            *uploaded = SourceUploadState {
                revision,
                owned: false,
            };
            continue;
        }
        if uploaded.owned && uploaded.revision == revision {
            continue;
        }
        let result = match frame {
            Some(frame) => open.set_source(index, frame.width, frame.height, &frame.rgba),
            None => open.clear_source(index),
        };
        match result {
            Ok(()) => {
                *uploaded = SourceUploadState {
                    revision,
                    owned: true,
                }
            }
            Err(error) => once.failed(error.to_string()),
        }
    }
}

fn source_needs_clear(configured: bool, revision: u64, uploaded: SourceUploadState) -> bool {
    !configured && (uploaded.owned || uploaded.revision != revision)
}

/// What a program actually draws.
///
/// A recording carries statements, because the score realm watched a score
/// being evaluated. A program built from text carries no statements, because
/// nothing evaluated it, so its text is read into the same nodes. Both reach
/// the renderer as chains.
fn chains_of(program: &HydraProgram) -> Result<Vec<HydraNode>, String> {
    if !program.statements.is_empty() {
        // hydra-synth retains one source chain per output. Re-evaluating a
        // later `.out(oN)` replaces the earlier chain, and its render loop
        // always ticks the surviving outputs in o0..o3 order. `hush()` resets
        // every earlier source while allowing later chains in the same score.
        let mut outputs: [Option<HydraNode>; crate::native::OUTPUTS] =
            std::array::from_fn(|_| None);
        for statement in &program.statements {
            let HydraStatement::Evaluate { node } = statement else {
                continue;
            };
            if matches!(command_of(node), Some(("hush", []))) {
                outputs.fill(None);
            } else if !is_command(node) {
                outputs[crate::glsl::output_of(node)] = Some(node.clone());
            }
        }
        return Ok(outputs.into_iter().flatten().collect());
    }
    let Some(raw) = &program.raw else {
        return Ok(Vec::new());
    };
    crate::glsl::parse_chain(raw)
        .map(|node| vec![node])
        .map_err(|error| format!("sketch: {error}"))
}

/// Whether a recorded statement is an instruction rather than a sketch.
fn command_of(node: &HydraNode) -> Option<(&str, &[HydraNode])> {
    let HydraNode::Chain { head, args, calls } = node else {
        return None;
    };
    if !calls.is_empty() {
        return None;
    }
    matches!(
        head.as_str(),
        "render" | "hush" | "setResolution" | "update" | "setFunction" | "speed" | "bpm"
    )
    .then(|| (head.as_str(), args.as_slice()))
}

fn is_command(node: &HydraNode) -> bool {
    command_of(node).is_some()
}

fn program_has_hush(program: &HydraProgram) -> bool {
    program.statements.iter().any(|statement| {
        matches!(
            statement,
            HydraStatement::Evaluate { node }
                if matches!(command_of(node), Some(("hush", [])))
        )
    })
}

/// Carry out the canvas-only composite, at most once for the final selection.
/// Earlier `render()` calls are overwritten by later canvas commands just as
/// they are upstream; replaying every recorded call would turn 64 statements
/// into 64 unnecessary full-screen passes on every frame.
fn run_commands(open: &mut NativeRenderer, program: &HydraProgram) {
    if wants_display_composite(program) {
        // `render()` shows all four outputs on a canvas-only target. It must
        // never write or swap o0: that changes feedback.
        let _ = open.render_all();
    }
}

fn wants_display_composite(program: &HydraProgram) -> bool {
    selected_output(program) == CanvasSelection::Display
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CanvasSelection {
    Output(usize),
    Display,
}

/// Which surface a recording puts on the canvas: o0 unless a `render()`
/// selects another, and the four-up display for a bare `render()`. A raw
/// program has no statements; `draw` reads its chain's output instead.
fn selected_output(program: &HydraProgram) -> CanvasSelection {
    program
        .statements
        .iter()
        .rev()
        .find_map(|statement| {
            let HydraStatement::Evaluate { node } = statement else {
                return None;
            };
            if matches!(command_of(node), Some(("hush", []))) {
                // Upstream's hush ends with render(o0). A later render command
                // can override it; an earlier one cannot survive it.
                return Some(CanvasSelection::Output(0));
            }
            let Some(("render", args)) = command_of(node) else {
                return None;
            };
            let [output] = args else {
                // Bare `render()` selects the composite. It is authoritative when
                // it comes after an earlier `render(oN)`; other invalid arities
                // are likewise never a reason to resurrect an older choice.
                return Some(if args.is_empty() {
                    CanvasSelection::Display
                } else {
                    CanvasSelection::Output(0)
                });
            };
            // The recorder preserves pristine globals as zero-call chains,
            // while hand-built protocol fixtures may use the explicit Global
            // node. They are the same Hydra value (`render(o3)`).
            let name = match output {
                HydraNode::Global { name } => name.as_str(),
                HydraNode::Chain { head, args, calls } if args.is_empty() && calls.is_empty() => {
                    head.as_str()
                }
                _ => return Some(CanvasSelection::Output(0)),
            };
            Some(CanvasSelection::Output(
                name.strip_prefix('o')
                    .and_then(|index| index.parse::<usize>().ok())
                    .filter(|index| *index < crate::native::OUTPUTS)
                    .unwrap_or(0),
            ))
        })
        .unwrap_or(CanvasSelection::Output(0))
}

/// Draw a single chain into its own slot: the shelf's preview or the theme's
/// sketch. Neither has a program around it.
#[allow(clippy::too_many_arguments)]
fn draw_one(
    renderer: &mut Option<NativeRenderer>,
    node: &HydraNode,
    shared: &Arc<Shared>,
    once: &mut SaidOnce<'_>,
    time: f32,
    size: &AtomicU32,
    slot: &FrameSlot,
    still: Option<(&AtomicU64, &AtomicU64, u64)>,
    source_uploads: Option<&mut [SourceUploadState; crate::HYDRA_SOURCE_SLOTS]>,
) {
    let packed = size.load(Ordering::Relaxed);
    let (width, height) = if packed == 0 {
        (PREVIEW_EDGE, PREVIEW_EDGE)
    } else {
        (packed >> 16, packed & 0xffff)
    };
    // Smoothing is a property of the backdrop, not of which sketch fills it,
    // so the theme's picture renders at the same factor the score's does. The
    // shelf especially: at one fragment per delivered pixel it was drawing a
    // few hundred in total.
    let geometry = match frame_geometry(width, height, render_factor(width, height, shared)) {
        Ok(geometry) => geometry,
        Err(message) => {
            once.failed(message);
            return;
        }
    };
    let opening = renderer.is_none();
    if opening {
        match NativeRenderer::new(geometry.render_width, geometry.render_height) {
            Ok(open) => *renderer = Some(open),
            Err(error) => {
                once.failed(error.to_string());
                return;
            }
        }
    }
    let Some(open) = renderer.as_mut() else {
        return;
    };
    if let Err(error) = open.resize(geometry.render_width, geometry.render_height) {
        once.failed(error.to_string());
        return;
    }
    if let Some(uploads) = source_uploads {
        if opening {
            uploads.fill(SourceUploadState::default());
        }
        // Camera-backed themes deliberately own only s0. Score acquisition
        // and the theme acquisition share a generation-safe input slot, but
        // runtime lifecycle policy ensures only the visible owner publishes.
        sync_input_sources(open, shared, &[true, false, false, false], uploads, once);
    }
    // The engine's audio reaches this picture too, so an audio-reactive
    // snippet previews reacting and an audio-reactive theme breathes with
    // the set.
    if let Ok(bins) = shared.audio.lock() {
        open.set_audio(*bins);
    }
    if let Err(error) = open.draw(node, time) {
        once.failed(error.to_string());
        return;
    }
    // `NativeRenderer::draw` writes to the first valid `.out(oN)` selected by
    // `output_of`; read that same target. Reading o0 unconditionally made a
    // valid camera theme such as `src(s0).out(o1)` capture invisibly.
    match open.read(single_chain_output(node)) {
        Ok(pixels) => {
            // A frame drawn for a sketch that was swapped or removed while the
            // GPU worked must not land: the slot was cleared for a reason.
            if !epoch_is_current(still) {
                return;
            }
            let pixels = match reduce_to_delivery(
                pixels,
                geometry,
                shared.smoothing.load(Ordering::Relaxed),
            ) {
                Ok(pixels) => pixels,
                Err(message) => {
                    once.failed(message);
                    return;
                }
            };
            if let Ok(mut held) = slot.frame.lock() {
                if !epoch_is_current(still) {
                    return;
                }
                *held = Some((geometry.width as u16, geometry.height as u16, pixels));
                acknowledge_epoch(still);
            }
        }
        Err(error) => once.failed(error.to_string()),
    }
}

/// Whether the epoch a draw started under is still the one in force -
/// for the theme and the preview alike.
fn epoch_is_current(still: Option<(&AtomicU64, &AtomicU64, u64)>) -> bool {
    still.is_none_or(|(epoch, _, expected)| epoch.load(Ordering::Acquire) == expected)
}

fn acknowledge_epoch(still: Option<(&AtomicU64, &AtomicU64, u64)>) {
    if let Some((epoch, drawn_epoch, expected)) = still
        && epoch.load(Ordering::Acquire) == expected
    {
        drawn_epoch.store(expected, Ordering::Release);
    }
}

fn single_chain_output(node: &HydraNode) -> usize {
    crate::glsl::output_of(node)
}

#[cfg(test)]
mod frame_acknowledgement {
    use super::*;

    #[test]
    fn frame_handles_acknowledge_only_the_current_installed_sketch() {
        let host = HydraHost::new();
        let streams = [
            (
                host.preview_frames(),
                &host.shared.preview_epoch,
                &host.shared.preview_drawn_epoch,
                &host.shared.previewing,
            ),
            (
                host.theme_frames(),
                &host.shared.theme_epoch,
                &host.shared.theme_drawn_epoch,
                &host.shared.theme_installed,
            ),
        ];

        for (frames, current, drawn, installed) in streams {
            assert_eq!(frames.sketch_epoch(), Some(0));
            assert!(!frames.sketch_drawn(0));

            installed.store(true, Ordering::Release);
            current.store(7, Ordering::Release);
            acknowledge_epoch(Some((current, drawn, 7)));
            assert_eq!(frames.sketch_epoch(), Some(7));
            assert!(frames.sketch_drawn(7));

            current.store(8, Ordering::Release);
            assert!(!frames.sketch_drawn(7), "the previous sketch is stale");
            assert!(!frames.sketch_drawn(8), "the replacement has not drawn");
            acknowledge_epoch(Some((current, drawn, 7)));
            assert!(
                !frames.sketch_drawn(8),
                "a stale draw cannot acknowledge it"
            );
            acknowledge_epoch(Some((current, drawn, 8)));
            assert!(frames.sketch_drawn(8));

            installed.store(false, Ordering::Release);
            assert!(!frames.sketch_drawn(8), "a removed sketch is not current");
        }

        let score = host.frames();
        assert_eq!(score.sketch_epoch(), None);
        assert!(
            !score.sketch_drawn(8),
            "score frames do not use sketch epochs"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{HydraCall, HydraOptions, HydraSource};

    /// `fps = 12` on the worker's 30 ticks a second draws 12 frames a
    /// second, not the 10 that comparing with the last draw gives.
    #[test]
    fn fps_holds_its_rate_on_the_worker_ticks() {
        let start = Instant::now();
        let tick = Duration::from_nanos(1_000_000_000 / 30);
        let mut planned = None;
        let mut draws = 0;
        for index in 0..300 {
            let now = start + tick * index;
            if planned.is_none_or(|next| now >= next) {
                draws += 1;
                planned = Some(next_fps_draw(planned, now, 12.0));
            }
        }
        assert!((119..=121).contains(&draws), "{draws} draws in ten seconds");
    }

    #[test]
    fn bpm_steps_every_array_argument_at_its_rate() {
        let list = |speed| HydraNode::List {
            v: vec![HydraNode::Number { v: 3.0 }, HydraNode::Number { v: 4.0 }],
            speed,
            smooth: 0.0,
            ease: "linear".into(),
            offset: 0.0,
        };
        let mut program = HydraProgram {
            statements: vec![HydraStatement::Evaluate {
                node: HydraNode::Chain {
                    head: "shape".into(),
                    args: vec![list(1.0)],
                    calls: vec![HydraCall {
                        method: "rotate".into(),
                        args: vec![list(2.0)],
                    }],
                },
            }],
            ..HydraProgram::default()
        };
        step_arrays_at(&mut program, 120.0);
        let HydraStatement::Evaluate {
            node: HydraNode::Chain { args, calls, .. },
        } = &program.statements[0]
        else {
            panic!("the chain stays a chain");
        };
        assert_eq!(
            args[0],
            list(4.0),
            "120 bpm steps four times as often as 30"
        );
        assert_eq!(calls[0].args[0], list(8.0));
    }

    fn output(head: &str, slot: usize) -> HydraStatement {
        HydraStatement::Evaluate {
            node: HydraNode::Chain {
                head: head.into(),
                args: Vec::new(),
                calls: vec![HydraCall {
                    method: "out".into(),
                    args: vec![HydraNode::Global {
                        name: format!("o{slot}"),
                    }],
                }],
            },
        }
    }

    fn command(head: &str) -> HydraStatement {
        HydraStatement::Evaluate {
            node: HydraNode::Chain {
                head: head.into(),
                args: Vec::new(),
                calls: Vec::new(),
            },
        }
    }

    fn render_selection(slot: usize) -> HydraStatement {
        HydraStatement::Evaluate {
            node: HydraNode::Chain {
                head: "render".into(),
                args: vec![HydraNode::Global {
                    name: format!("o{slot}"),
                }],
                calls: Vec::new(),
            },
        }
    }

    #[test]
    fn frame_geometry_bounds_delivery_and_steps_the_factor_down_to_fit() {
        assert_eq!(
            frame_geometry(640, 360, 3).unwrap(),
            FrameGeometry {
                width: 640,
                height: 360,
                render_width: 1920,
                render_height: 1080,
                factor: 3,
            }
        );
        // A factor that will not fit steps DOWN to the largest that does,
        // rather than collapsing to one fragment per delivered pixel.
        assert_eq!(
            frame_geometry(4096, 1, 8).unwrap(),
            FrameGeometry {
                width: 4096,
                height: 1,
                render_width: 4096,
                render_height: 1,
                factor: 1,
            },
            "an oversized factor must not turn a valid delivery into an invalid allocation"
        );
        let stepped = frame_geometry(1000, 500, 8).unwrap();
        assert!(
            stepped.factor > 1 && stepped.factor < 8,
            "a factor that half-fits should land between the two, not at either end: {stepped:?}"
        );
        assert_eq!(frame_geometry(0, 0, 1).unwrap().width, 1);
        assert!(frame_geometry(4096, 1024, 1).is_ok());
        assert!(frame_geometry(4096, 1025, 1).is_err());
        assert!(frame_geometry(u32::MAX, u32::MAX, 8).is_err());
    }

    /// A terminal-sized delivery must not render at terminal size.
    ///
    /// A 200x50 cell backdrop asking for one fragment per cell evaluated the
    /// whole sketch at ten thousand fragments, which is why `noise`, `voronoi`
    /// and every feedback chain looked broken rather than merely downscaled.
    #[test]
    fn a_coarse_delivery_buys_a_large_render_factor() {
        for (width, height) in [(200u32, 50u32), (80, 24), (120, 40)] {
            let base = u64::from(width) * u64::from(height);
            let factor = (TARGET_RENDER_PIXELS / base).isqrt().clamp(1, 8) as u32;
            let geometry = frame_geometry(width, height, factor).unwrap();
            let rendered = u64::from(geometry.render_width) * u64::from(geometry.render_height);
            assert!(
                rendered >= base * 16,
                "{width}x{height} rendered only {rendered} fragments for {base} cells"
            );
            assert_eq!(geometry.render_width, width * geometry.factor);
            assert_eq!(geometry.render_height, height * geometry.factor);
        }
    }

    #[test]
    fn downsample_requires_exact_checked_rgba_geometry() {
        let pixels = [9, 18, 27, 99].repeat(9);
        assert_eq!(downsample(&pixels, 3, 3, 3).unwrap(), [9, 18, 27, 255]);
        assert!(downsample(&pixels[..35], 3, 3, 3).is_err());
        assert!(downsample(&pixels, 3, 3, 0).is_err());
        assert!(downsample(&pixels, 3, 2, 3).is_err());
    }

    /// Point-sampling keeps values the sketch produced; averaging blends them.
    ///
    /// The distinction is what alpha needs: a mask that leaves half a frame
    /// thin must not average into a uniformly middling opacity, or the shape
    /// it cut is gone.
    #[test]
    fn point_sample_keeps_a_real_pixel_rather_than_a_blend() {
        // A 2x2 box: three transparent black pixels and one opaque white.
        let pixels = [
            0, 0, 0, 0, 255, 255, 255, 255, //
            0, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(
            downsample(&pixels, 2, 2, 2).unwrap(),
            [63, 63, 63, 255],
            "averaging blends the colour and forces the result opaque"
        );
        assert_eq!(
            point_sample(&pixels, 2, 2, 2).unwrap(),
            [0, 0, 0, 0],
            "point-sampling returns one of the four verbatim"
        );
        assert!(point_sample(&pixels, 2, 2, 0).is_err());
        assert!(point_sample(&pixels, 3, 2, 2).is_err());
    }

    #[test]
    fn input_leases_are_latest_only_and_generation_safe() {
        let host = HydraHost::new();
        let sink = host.input_sink();
        let old = sink.bind(0).expect("s0 lease");
        assert!(old.publish(1, 1, vec![1, 2, 3, 4]).unwrap());

        let current = sink.bind(0).expect("replacement s0 lease");
        assert!(!old.publish(1, 1, vec![5, 6, 7, 8]).unwrap());
        assert!(current.publish(1, 1, vec![9, 10, 11, 12]).unwrap());
        assert!(current.publish(1, 1, vec![13, 14, 15, 16]).unwrap());

        let (_, frame) = host.shared.inputs[0].snapshot();
        assert_eq!(frame.expect("latest frame").rgba.as_ref(), [13, 14, 15, 16]);
        assert!(sink.bind(crate::HYDRA_SOURCE_SLOTS as u8).is_none());
    }

    #[test]
    fn single_chain_delivery_reads_the_same_output_the_renderer_draws() {
        let parsed = |code| crate::glsl::parse_chain(code).expect("chain parses");
        assert_eq!(single_chain_output(&parsed("src(s0).out(o1)")), 1);
        assert_eq!(single_chain_output(&parsed("src(s0).out()")), 0);
        assert_eq!(
            single_chain_output(&parsed("src(s0).out(o1).out(o3)")),
            1,
            "the renderer and reader both use Hydra's first out call for an odd repeated-out chain"
        );
    }

    #[test]
    fn theme_draw_acknowledgement_is_exact_and_invalidated_by_replacement() {
        let host = HydraHost::new();
        host.shared.theme_installed.store(true, Ordering::Release);
        host.shared.theme_epoch.store(7, Ordering::Release);
        host.shared.theme_drawn_epoch.store(7, Ordering::Release);
        assert!(host.theme_drawn(7));
        assert!(!host.theme_drawn(6));

        host.shared.theme_epoch.store(8, Ordering::Release);
        assert!(
            !host.theme_drawn(7),
            "an old successful renderer cannot authorize the replacement"
        );
        host.shared.theme_drawn_epoch.store(8, Ordering::Release);
        assert!(host.theme_drawn(8));
        host.shared.theme_installed.store(false, Ordering::Release);
        assert!(!host.theme_drawn(8));

        let current = AtomicU64::new(12);
        let drawn = AtomicU64::new(0);
        acknowledge_epoch(Some((&current, &drawn, 11)));
        assert_eq!(
            drawn.load(Ordering::Acquire),
            0,
            "an old local theme node cannot stamp a queued replacement's epoch"
        );
        acknowledge_epoch(Some((&current, &drawn, 12)));
        assert_eq!(drawn.load(Ordering::Acquire), 12);
    }

    #[test]
    fn stale_lease_clear_cannot_erase_its_replacement() {
        let host = HydraHost::new();
        let sink = host.input_sink();
        let old_camera = sink.bind(0).expect("camera lease");
        assert!(old_camera.publish(1, 1, vec![1, 2, 3, 255]).unwrap());
        let image = sink.bind(0).expect("replacement image lease");
        assert!(image.publish(1, 1, vec![4, 5, 6, 255]).unwrap());

        assert!(!old_camera.clear_if_current());
        let (_, frame) = host.shared.inputs[0].snapshot();
        assert_eq!(
            frame.expect("image survives stale clear").rgba.as_ref(),
            [4, 5, 6, 255]
        );
        assert!(image.clear_if_current());
        assert!(host.shared.inputs[0].snapshot().1.is_none());
    }

    #[test]
    fn tui_publication_validates_grid_bounds_and_exact_cell_count() {
        let host = HydraHost::new();
        let sink = host.tui_sink();
        let valid = HydraTuiFrame {
            cols: 2,
            rows: 2,
            cells: vec![crate::HydraTuiCell::default(); 4],
            ..HydraTuiFrame::default()
        };
        assert!(
            !sink.publish(&valid).unwrap(),
            "an unwanted frame is not cloned"
        );

        let malformed = HydraTuiFrame {
            cells: vec![crate::HydraTuiCell::default(); 3],
            ..valid.clone()
        };
        assert!(sink.publish(&malformed).is_err());
        let hostile = HydraTuiFrame {
            cols: u16::MAX,
            rows: u16::MAX,
            cells: Vec::new(),
            ..HydraTuiFrame::default()
        };
        assert!(sink.publish(&hostile).is_err());

        host.shared.wants_tui.store(true, Ordering::Relaxed);
        assert!(sink.publish(&valid).unwrap());
        assert_eq!(host.shared.tui.grid.lock().unwrap().as_ref(), Some(&valid));
    }

    #[test]
    fn clearing_a_slot_invalidates_its_lease() {
        let host = HydraHost::new();
        let sink = host.input_sink();
        let lease = sink.bind(2).expect("s2 lease");
        assert!(lease.publish(1, 1, vec![1, 2, 3, 4]).unwrap());
        sink.clear(2);
        assert!(!lease.publish(1, 1, vec![5, 6, 7, 8]).unwrap());
        assert!(host.shared.inputs[2].snapshot().1.is_none());
    }

    #[test]
    fn input_publication_requires_one_exact_bounded_rgba_frame() {
        let host = HydraHost::new();
        let lease = host.input_sink().bind(0).unwrap();
        assert!(lease.publish(0, 1, Vec::new()).is_err());
        assert!(lease.publish(1, 1, vec![0; 3]).is_err());
        assert!(lease.publish(2049, 1, vec![0; 2049 * 4]).is_err());
    }

    #[test]
    fn configured_source_slots_are_explicit_per_program() {
        let mut program = HydraProgram {
            statements: vec![HydraStatement::ConfigureSource {
                slot: 3,
                source: HydraSource::Camera { device: None },
            }],
            ..HydraProgram::default()
        };
        assert_eq!(
            configured_source_slots(&program),
            [false, false, false, true]
        );
        assert_eq!(
            configured_source_slots(&HydraProgram::default()),
            [false; 4]
        );
        program.options.feed_strudel = true;
        assert!(feed_strudel_owns_s0(
            &program,
            &configured_source_slots(&program)
        ));
        program.statements.push(HydraStatement::ConfigureSource {
            slot: 0,
            source: HydraSource::ImageUrl {
                url: "https://example.com/texture.png".into(),
            },
        });
        assert!(!feed_strudel_owns_s0(
            &program,
            &configured_source_slots(&program)
        ));
        program
            .statements
            .push(HydraStatement::ClearSource { slot: 0 });
        assert!(feed_strudel_owns_s0(
            &program,
            &configured_source_slots(&program)
        ));
        program.statements.push(command("hush"));
        assert_eq!(configured_source_slots(&program), [false; 4]);
    }

    #[test]
    fn a_new_input_revision_retires_an_unowned_feed_strudel_texture_once() {
        let feed_strudel_upload = SourceUploadState {
            revision: 7,
            owned: false,
        };
        assert!(source_needs_clear(false, 8, feed_strudel_upload));
        assert!(!source_needs_clear(false, 7, feed_strudel_upload));
        assert!(!source_needs_clear(true, 8, feed_strudel_upload));
    }

    #[test]
    fn final_output_plan_keeps_only_the_last_chain_and_ticks_o0_through_o3() {
        let program = HydraProgram {
            statements: vec![output("osc", 2), output("noise", 0), output("shape", 2)],
            ..HydraProgram::default()
        };
        let chains = chains_of(&program).expect("output plan");
        let heads: Vec<_> = chains
            .iter()
            .map(|node| match node {
                HydraNode::Chain { head, .. } => head.as_str(),
                _ => panic!("an output source is a chain"),
            })
            .collect();
        assert_eq!(heads, ["noise", "shape"]);
    }

    #[test]
    fn hush_discards_earlier_output_sources_but_not_later_ones() {
        let program = HydraProgram {
            statements: vec![output("osc", 0), command("hush"), output("noise", 1)],
            ..HydraProgram::default()
        };
        let chains = chains_of(&program).expect("output plan");
        assert!(matches!(
            chains.as_slice(),
            [HydraNode::Chain { head, .. }] if head == "noise"
        ));
        assert!(program_has_hush(&program));
    }

    #[test]
    fn signal_schedule_advances_by_step_and_holds_its_last_sample() {
        let received = Instant::now();
        let schedule = SignalSchedule::from_frame(
            &HydraSignalFrame {
                step: 10.0,
                slots: vec![
                    vec![1.into(), 2.into(), 3.into()],
                    vec![serde_json::Value::Null, 7.into()],
                ],
            },
            received,
        );
        assert_eq!(schedule.values_at(received), [1.0, 0.0]);
        assert_eq!(
            schedule.values_at(received + Duration::from_millis(19)),
            [2.0, 7.0]
        );
        assert_eq!(
            schedule.values_at(received + Duration::from_secs(1)),
            [3.0, 7.0],
            "a late renderer holds the last lookahead value"
        );
    }

    #[test]
    fn audio_smoothing_matches_upstream_and_detect_audio_gates_only_the_score() {
        let host = HydraHost::new();
        let mut frame = HydraAudioFrame::default();
        frame.bins.fill(1.0);
        host.audio(&frame);
        assert_eq!(host.shared.audio.lock().unwrap()[0], 0.6);
        host.audio(&HydraAudioFrame::default());
        assert!((host.shared.audio.lock().unwrap()[0] - 0.24).abs() < 1e-6);

        let bands = [0.5; crate::HYDRA_AUDIO_BINS];
        let mut muted = HydraProgram {
            options: HydraOptions {
                detect_audio: false,
                ..HydraOptions::default()
            },
            ..HydraProgram::default()
        };
        assert_eq!(score_audio(&muted, bands), [0.0; crate::HYDRA_AUDIO_BINS]);
        muted.options.detect_audio = true;
        assert_eq!(score_audio(&muted, bands), bands);
    }

    #[test]
    fn render_accepts_the_recorders_pristine_chain_output_reference() {
        let program = HydraProgram {
            statements: vec![HydraStatement::Evaluate {
                node: HydraNode::Chain {
                    head: "render".into(),
                    args: vec![HydraNode::Chain {
                        head: "o3".into(),
                        args: Vec::new(),
                        calls: Vec::new(),
                    }],
                    calls: Vec::new(),
                },
            }],
            ..HydraProgram::default()
        };
        assert_eq!(selected_output(&program), CanvasSelection::Output(3));
    }

    #[test]
    fn repeated_bare_render_commands_collapse_to_one_final_composite_decision() {
        let program = HydraProgram {
            statements: (0..crate::MAX_HYDRA_STATEMENTS)
                .map(|_| command("render"))
                .collect(),
            ..HydraProgram::default()
        };
        assert!(
            wants_display_composite(&program),
            "all repeated calls reduce to one boolean composite pass"
        );

        let mut ordered = HydraProgram {
            statements: vec![command("render"), command("render"), render_selection(2)],
            ..HydraProgram::default()
        };
        assert!(
            !wants_display_composite(&ordered),
            "a later render(o2) makes every earlier composite irrelevant"
        );
        ordered.statements.push(command("render"));
        assert!(wants_display_composite(&ordered));
        ordered.statements.push(command("hush"));
        assert!(
            !wants_display_composite(&ordered),
            "hush finishes by selecting o0"
        );
    }

    #[test]
    fn hush_resets_canvas_selection_to_o0_until_a_later_render() {
        let program = HydraProgram {
            statements: vec![
                HydraStatement::Evaluate {
                    node: HydraNode::Chain {
                        head: "render".into(),
                        args: vec![HydraNode::Global { name: "o3".into() }],
                        calls: Vec::new(),
                    },
                },
                command("hush"),
            ],
            ..HydraProgram::default()
        };
        assert_eq!(selected_output(&program), CanvasSelection::Output(0));
    }

    /// A renderer at a given size with half of its ten output textures
    /// written, as a one-output sketch leaves them. `in_process` puts the
    /// textures in the process (software rasteriser, Apple silicon).
    /// Otherwise they are a GPU driver's.
    fn footprint(width: u32, height: u32, in_process: bool) -> NativeFootprint {
        let texture = width as usize * height as usize * 4;
        NativeFootprint {
            width,
            height,
            textures: 10 * texture,
            resident: 5 * texture,
            readback: texture,
            textures_in_process: in_process,
        }
    }

    /// The breakdown reads what the worker last recorded for each picture:
    /// an opened renderer shows, a resize shows at its new size, and a
    /// retired renderer or an ended worker reads as nothing held.
    #[test]
    fn the_gauges_follow_a_renderer_opening_resizing_and_retiring() {
        let host = HydraHost::new();
        let frames = host.theme_frames();
        assert!(host.memory().is_empty(), "nothing has been opened");

        let [score, _, theme] = &host.shared.renderers;
        theme.record(Some(footprint(1280, 720, true)));
        assert_eq!(frames.memory(), host.memory(), "any handle reads the host");
        assert_eq!(host.memory().theme, Some(footprint(1280, 720, true)));
        assert_eq!(host.memory().score, None);
        assert_eq!(
            host.memory().process_bytes(),
            6 * 1280 * 720 * 4,
            "a software adapter's written textures are the process's"
        );
        assert_eq!(host.memory().gpu_bytes(), 0);

        theme.record(Some(footprint(640, 360, true)));
        assert_eq!(host.memory().theme, Some(footprint(640, 360, true)));

        score.record(Some(footprint(1024, 576, false)));
        let memory = host.memory();
        assert_eq!(
            memory
                .renderers()
                .map(|(picture, _)| picture)
                .collect::<Vec<_>>(),
            ["score", "theme"]
        );
        assert_eq!(
            memory.process_bytes(),
            6 * 640 * 360 * 4 + 1024 * 576 * 4,
            "a GPU's textures are not, integrated or discrete"
        );
        assert_eq!(
            memory.gpu_bytes(),
            10 * 1024 * 576 * 4,
            "they are said beside the figure, as allocated"
        );

        theme.record(None);
        assert_eq!(host.memory().theme, None, "retired");
        {
            let _worker = GaugesClearedOnExit(&host.shared);
        }
        assert!(host.memory().is_empty(), "a worker that ends holds nothing");
    }

    /// A camera or image frame waiting for the GPU is counted while its
    /// slot holds it, and not once the slot is cleared or rebound - read
    /// without the lock a camera publishes under.
    #[test]
    fn input_frames_are_counted_while_their_slot_holds_them() {
        let host = HydraHost::new();
        let sink = host.input_sink();
        let image = sink.bind(1).expect("s1 lease");
        assert!(image.publish(4, 2, vec![0; 32]).unwrap());
        let camera = sink.bind(0).expect("s0 lease");
        assert!(camera.publish(2, 2, vec![0; 16]).unwrap());
        assert_eq!(host.memory().inputs, 48);
        assert!(
            !host.memory().is_empty(),
            "an input alone is something held"
        );

        assert!(camera.publish(1, 1, vec![0; 4]).unwrap());
        assert_eq!(
            host.memory().inputs,
            36,
            "the newest frame replaces the last"
        );
        sink.clear(1);
        assert_eq!(host.memory().inputs, 4);
        assert!(camera.clear_if_current());
        assert_eq!(host.memory().inputs, 0);
        assert!(host.memory().is_empty());
    }

    /// Failure messages sent on `events` while `count` frames run through
    /// `frame`, each ended on `once` as the worker ends it.
    fn failures_across(
        count: usize,
        once: &mut SaidOnce<'_>,
        events: &mpsc::Receiver<HydraEvent>,
        mut frame: impl FnMut(usize, &mut SaidOnce<'_>),
    ) -> Vec<String> {
        for step in 0..count {
            frame(step, once);
            once.end_frame();
        }
        events
            .try_iter()
            .filter_map(|event| match event {
                HydraEvent::Failed { message } => Some(message),
                _ => None,
            })
            .collect()
    }

    /// A source upload that keeps failing is reported once while it keeps
    /// failing, whether the camera publishes every frame or less often, and
    /// once again after a clean upload.
    #[test]
    fn a_broken_source_is_said_once_while_it_stays_broken() {
        let Ok(mut open) = NativeRenderer::new(64, 64) else {
            eprintln!("skipped: no GPU and no software rasteriser");
            return;
        };
        let refusal = open
            .set_source(0, 0, 1, &[])
            .expect_err("a zero-width source is refused")
            .to_string();
        let mut renderer = Some(open);
        let shared = Arc::new(Shared::default());
        let (events, heard) = mpsc::channel();
        let mut once = SaidOnce::new(&events);
        let size = AtomicU32::new(0);
        let slot = FrameSlot::default();
        let mut uploads = [SourceUploadState::default(); crate::HYDRA_SOURCE_SLOTS];
        let program = HydraProgram {
            statements: vec![
                HydraStatement::ConfigureSource {
                    slot: 0,
                    source: HydraSource::Camera { device: None },
                },
                output("osc", 0),
            ],
            options: HydraOptions {
                width: 64,
                height: 64,
                ..HydraOptions::default()
            },
            ..HydraProgram::default()
        };
        let publish = |width: u32, rgba: Vec<u8>| {
            *shared.inputs[0].frame.lock().unwrap() = Some(InputFrame {
                width,
                height: 1,
                rgba: Arc::from(rgba),
            });
            shared.inputs[0].revision.fetch_add(1, Ordering::Release);
        };
        let mut draw_frame = |step: usize, once: &mut SaidOnce<'_>| {
            draw(
                &mut renderer,
                &program,
                &shared,
                &events,
                once,
                step as f32 * 0.016,
                &size,
                &slot,
                false,
                &mut uploads,
                false,
            );
        };

        for period in [1, 3] {
            let said = failures_across(12 * period, &mut once, &heard, |step, once| {
                if step % period == 0 {
                    publish(0, Vec::new());
                }
                draw_frame(step, once);
            });
            assert_eq!(
                said,
                [refusal.as_str()],
                "a camera publishing every {period} frames"
            );
            let said = failures_across(1, &mut once, &heard, |step, once| {
                publish(1, vec![9, 18, 27, 255]);
                draw_frame(step, once);
            });
            assert_eq!(said, Vec::<String>::new(), "a clean upload says nothing");
        }
    }

    /// A frame that will not reduce is reported once while it keeps failing,
    /// and once again after a frame that reduces. It is not counted as
    /// delivered.
    #[test]
    fn a_frame_that_will_not_reduce_is_said_once_and_not_delivered() {
        let (events, heard) = mpsc::channel();
        let mut once = SaidOnce::new(&events);
        let shared = Shared::default();
        shared.smoothing.store(true, Ordering::Relaxed);
        let slot = FrameSlot::default();
        // A 4x4 render for a 2x2 delivery: 64 bytes reduce, 10 do not.
        let geometry = FrameGeometry {
            width: 2,
            height: 2,
            render_width: 4,
            render_height: 4,
            factor: 2,
        };
        let refusal = downsample(&[0; 10], 4, 4, 2).expect_err("10 bytes are not a 4x4 frame");

        for round in 1..=2 {
            let said = failures_across(12, &mut once, &heard, |_, once| {
                deliver(vec![0; 10], geometry, &shared, &slot, once);
            });
            assert_eq!(said, [refusal.as_str()], "round {round}");
            let said = failures_across(1, &mut once, &heard, |_, once| {
                deliver(vec![0; 64], geometry, &shared, &slot, once);
            });
            assert_eq!(
                said,
                Vec::<String>::new(),
                "a frame that reduces says nothing"
            );
            assert_eq!(
                shared.delivered.load(Ordering::Relaxed),
                round,
                "only the frames that reduce are delivered"
            );
        }
    }

    /// The pace follows the score program's own fps: 60 halves the default
    /// wait, 15 doubles it, and no score program keeps the default.
    #[test]
    fn the_worker_pace_follows_the_score_program_fps() {
        assert_eq!(super::frame_interval(None), super::FRAME_INTERVAL);
        assert_eq!(
            super::frame_interval(Some(crate::HYDRA_FRAMES_PER_SECOND)),
            super::FRAME_INTERVAL
        );
        assert_eq!(super::frame_interval(Some(60)), super::FRAME_INTERVAL / 2);
        assert_eq!(super::frame_interval(Some(15)), super::FRAME_INTERVAL * 2);
    }
}
