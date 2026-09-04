use std::sync::{atomic::AtomicUsize, Arc, Mutex};

use openvm_circuit::{
    primitives::Chip, system::poseidon2::columns::Poseidon2PeripheryCols,
    utils::next_power_of_two_or_zero,
};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::{
    copy::{MemCopyD2H, MemCopyH2D},
    d_buffer::DeviceBuffer,
    stream::GpuDeviceCtx,
};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::prover::{AirProvingContext, MatrixDimensions};

use crate::cuda_abi::poseidon2;

#[derive(Clone)]
pub struct SharedBuffer<T> {
    buffer: Arc<Mutex<Option<Arc<DeviceBuffer<T>>>>>,
    pub idx: Arc<DeviceBuffer<u32>>,
}

impl<T> SharedBuffer<T> {
    pub fn records(&self) -> Arc<DeviceBuffer<T>> {
        let records = self.buffer.lock().unwrap();
        records
            .clone()
            .expect("Poseidon2 records buffer must be prepared before tracegen")
    }
}

pub struct Poseidon2ChipGPU<const SBOX_REGISTERS: usize> {
    pub device_ctx: GpuDeviceCtx,
    pub records: Arc<Mutex<Option<Arc<DeviceBuffer<F>>>>>,
    pub idx: Arc<DeviceBuffer<u32>>,
    /// Trace height pinned by [`Self::set_forced_height`], or 0 for "derive it from the records".
    ///
    /// This chip is the only one in the system whose exact height cannot be predicted from metered
    /// execution: tracegen deduplicates equal hash inputs by value. The metered pass counts one
    /// initial hash for every touched memory leaf/node and a possible distinct final hash only for
    /// written leaves and their ancestors. This is much tighter than counting both trees in full,
    /// but remains an upper bound because a write can restore the initial value and unrelated
    /// inputs can coincide. Pinning the height lets a caller plan from that sound bound instead of
    /// running trace generation twice just to learn this one number.
    forced_height: Arc<AtomicUsize>,
    #[cfg(feature = "metrics")]
    pub(crate) current_trace_height: Arc<AtomicUsize>,
}

impl<const SBOX_REGISTERS: usize> Poseidon2ChipGPU<SBOX_REGISTERS> {
    pub fn new(device_ctx: GpuDeviceCtx) -> Self {
        let idx = Arc::new(DeviceBuffer::<u32>::with_capacity_on(1, &device_ctx));
        idx.fill_zero_on(&device_ctx).unwrap();
        Self {
            device_ctx: device_ctx.clone(),
            records: Arc::new(Mutex::new(None)),
            idx,
            forced_height: Arc::new(AtomicUsize::new(0)),
            #[cfg(feature = "metrics")]
            current_trace_height: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Pins the next generated trace to `height` rows. `0` restores the default.
    ///
    /// The padding rows this adds are the same dummy permutation of the zero state that trace
    /// generation already writes between the deduplicated record count and the next power of
    /// two, so a larger height widens the padding region without changing its contents.
    pub fn set_forced_height(&self, height: usize) {
        assert!(
            height.is_power_of_two() || height == 0,
            "forced Poseidon2 trace height {height} must be a power of two"
        );
        self.forced_height
            .store(height, std::sync::atomic::Ordering::Relaxed);
    }

    fn take_forced_height(&self) -> usize {
        self.forced_height
            .swap(0, std::sync::atomic::Ordering::Relaxed)
    }

    /// Prepare an exact one-segment scratch buffer for Poseidon2 records.
    ///
    /// Each Poseidon2 record occupies `POSEIDON2_WIDTH` field elements.
    pub fn prepare_records(&self, num_records: usize) {
        self.idx.fill_zero_on(&self.device_ctx).unwrap();
        let mut records = self.records.lock().unwrap();
        assert!(
            records.is_none(),
            "Poseidon2 records buffer already prepared"
        );
        if num_records == 0 {
            return;
        }
        let num_elements = num_records
            .checked_mul(POSEIDON2_WIDTH)
            .expect("Poseidon2 records buffer size overflow");
        records.replace(Arc::new(DeviceBuffer::<F>::with_capacity_on(
            num_elements,
            &self.device_ctx,
        )));
    }

    pub fn shared_buffer(&self) -> SharedBuffer<F> {
        SharedBuffer {
            buffer: self.records.clone(),
            idx: self.idx.clone(),
        }
    }

    pub fn trace_width() -> usize {
        Poseidon2PeripheryCols::<F, SBOX_REGISTERS>::width()
    }

    /// A trace of `forced_height` pure padding rows, or the empty matrix when nothing is pinned.
    ///
    /// A chip that hashed nothing this segment still owes its pinned rows: the plan named a
    /// height and the shape catalog is bound to it. Returning the dummy matrix here would make
    /// the generated shape disagree with the plan, which the height gate rejects.
    fn padding_only_ctx(&self, forced_height: usize) -> AirProvingContext<GpuBackend> {
        if forced_height == 0 {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        let trace = DeviceMatrix::<F>::with_capacity_on(
            forced_height,
            Self::trace_width(),
            &self.device_ctx,
        );
        trace.buffer().fill_zero_on(&self.device_ctx).unwrap();
        let empty_records = DeviceBuffer::<F>::new();
        let empty_counts = DeviceBuffer::<u32>::new();
        unsafe {
            poseidon2::tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &empty_records,
                &empty_counts,
                0,
                SBOX_REGISTERS,
                self.device_ctx.stream.as_raw(),
            )
            .expect("Failed to generate Poseidon2 padding trace");
        }
        AirProvingContext::simple_no_pis(trace)
    }
}

impl<RA, const SBOX_REGISTERS: usize> Chip<RA, GpuBackend> for Poseidon2ChipGPU<SBOX_REGISTERS> {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        // Taken, not read: a pinned height applies to exactly one segment, so leaving it set
        // would silently carry into the next one.
        let forced_height = self.take_forced_height();
        let Some(records) = self.records.lock().unwrap().take() else {
            self.idx.fill_zero_on(&self.device_ctx).unwrap();
            return self.padding_only_ctx(forced_height);
        };
        debug_assert_eq!(records.len() % POSEIDON2_WIDTH, 0);
        let capacity_records = records.len() / POSEIDON2_WIDTH;
        let mut num_records = self.idx.to_host_on(&self.device_ctx).unwrap()[0] as usize;
        assert!(
            num_records <= capacity_records,
            "Poseidon2 records buffer overflow: pushed {num_records} records into capacity {capacity_records}"
        );
        if num_records == 0 {
            self.idx.fill_zero_on(&self.device_ctx).unwrap();
            return self.padding_only_ctx(forced_height);
        }
        let counts = DeviceBuffer::<u32>::with_capacity_on(num_records, &self.device_ctx);
        let dedup_records =
            DeviceBuffer::<F>::with_capacity_on(num_records * POSEIDON2_WIDTH, &self.device_ctx);
        let dedup_counts = DeviceBuffer::<u32>::with_capacity_on(num_records, &self.device_ctx);
        unsafe {
            let d_num_records = [num_records].to_device_on(&self.device_ctx).unwrap();
            let mut temp_bytes = 0;
            poseidon2::deduplicate_records_get_temp_bytes(
                &records,
                &counts,
                num_records,
                &d_num_records,
                &mut temp_bytes,
                self.device_ctx.stream.as_raw(),
            )
            .expect("Failed to get temp bytes");
            let d_temp_storage = if temp_bytes == 0 {
                DeviceBuffer::<u8>::new()
            } else {
                DeviceBuffer::<u8>::with_capacity_on(temp_bytes, &self.device_ctx)
            };
            poseidon2::deduplicate_records(
                &records,
                &counts,
                &dedup_records,
                &dedup_counts,
                num_records,
                &d_num_records,
                &d_temp_storage,
                temp_bytes,
                self.device_ctx.stream.as_raw(),
            )
            .expect("Failed to deduplicate records");
            num_records = *d_num_records
                .to_host_on(&self.device_ctx)
                .unwrap()
                .first()
                .unwrap();
        }
        drop(records);
        drop(counts);
        #[cfg(feature = "metrics")]
        self.current_trace_height
            .store(num_records, std::sync::atomic::Ordering::Relaxed);
        let natural_height = next_power_of_two_or_zero(num_records);
        let trace_height = if forced_height == 0 {
            natural_height
        } else {
            assert!(
                forced_height >= natural_height,
                "forced Poseidon2 trace height {forced_height} is below the {natural_height} \
                 rows the deduplicated records need"
            );
            forced_height
        };
        let trace = DeviceMatrix::<F>::with_capacity_on(
            trace_height,
            Self::trace_width(),
            &self.device_ctx,
        );
        trace.buffer().fill_zero_on(&self.device_ctx).unwrap();
        unsafe {
            poseidon2::tracegen(
                trace.buffer(),
                trace.height(),
                trace.width(),
                &dedup_records,
                &dedup_counts,
                num_records,
                SBOX_REGISTERS,
                self.device_ctx.stream.as_raw(),
            )
            .expect("Failed to generate trace");
        }
        // Reset state of this chip.
        self.idx.fill_zero_on(&self.device_ctx).unwrap();
        AirProvingContext::simple_no_pis(trace)
    }
}

pub enum Poseidon2PeripheryChipGPU {
    Register0(Poseidon2ChipGPU<0>),
    Register1(Poseidon2ChipGPU<1>),
}

impl Poseidon2PeripheryChipGPU {
    pub fn new(sbox_registers: usize, device_ctx: GpuDeviceCtx) -> Self {
        match sbox_registers {
            0 => Self::Register0(Poseidon2ChipGPU::new(device_ctx)),
            1 => Self::Register1(Poseidon2ChipGPU::new(device_ctx)),
            _ => panic!("Invalid number of sbox registers: {sbox_registers}"),
        }
    }

    pub fn prepare_records(&self, num_records: usize) {
        match self {
            Self::Register0(chip) => chip.prepare_records(num_records),
            Self::Register1(chip) => chip.prepare_records(num_records),
        }
    }

    /// See [`Poseidon2ChipGPU::set_forced_height`].
    pub fn set_forced_height(&self, height: usize) {
        match self {
            Self::Register0(chip) => chip.set_forced_height(height),
            Self::Register1(chip) => chip.set_forced_height(height),
        }
    }

    pub fn shared_buffer(&self) -> SharedBuffer<F> {
        match self {
            Self::Register0(chip) => chip.shared_buffer(),
            Self::Register1(chip) => chip.shared_buffer(),
        }
    }

    pub fn device_ctx(&self) -> &GpuDeviceCtx {
        match self {
            Self::Register0(chip) => &chip.device_ctx,
            Self::Register1(chip) => &chip.device_ctx,
        }
    }
}

impl<RA> Chip<RA, GpuBackend> for Poseidon2PeripheryChipGPU {
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<GpuBackend> {
        match self {
            Self::Register0(chip) => chip.generate_proving_ctx(()),
            Self::Register1(chip) => chip.generate_proving_ctx(()),
        }
    }
}
