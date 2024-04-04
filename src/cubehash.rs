use digest::{
    array::{Array, ArraySize}, block_buffer::Eager, core_api::{
        AlgorithmName, Block, BlockSizeUser, Buffer, BufferKindUser, FixedOutputCore, UpdateCore
    }, typenum::{IsGreater, IsLessOrEqual, True, Unsigned, U0, U32, U64}, HashMarker, Output, OutputSizeUser
};

mod soft;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unstable-avx512"))]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
cpufeatures::new!(cpu_sse2, "sse2");
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
cpufeatures::new!(cpu_avx2, "avx", "avx2");
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unstable-avx512"))]
cpufeatures::new!(cpu_avx512, "avx", "avx512f");
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
cpufeatures::new!(cpu_neon, "neon");

#[derive(Clone)]
enum Backend<const I: u16, const R: u16, const F: u16, H> {
    Soft(soft::Soft<I, R, F, H>),
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2(sse2::Sse2<I, R, F, H>),
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2(avx2::Avx2<I, R, F, H>),
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unstable-avx512"))]
    Avx512(avx512::Avx512<I, R, F, H>),
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    Neon(neon::Neon<I, R, F, H>),
}

#[derive(Clone)]
pub struct CubeHashCore<const I: u16, const R: u16, const F: u16, H>(Backend<I, R, F, H>);

trait CubeHashBackend<const I: u16, const R: u16, const F: u16, H> {
    unsafe fn init() -> Self where H: Unsigned;
    unsafe fn update_block(&mut self, block: &Array<u8, U32>);
    unsafe fn finalize(&mut self, out: &mut Array<u8, H>) where H: ArraySize + IsGreater<U0, Output = True> + IsLessOrEqual<U64, Output = True>;
}

impl<const I: u16, const R: u16, const F: u16, H> HashMarker for CubeHashCore<I, R, F, H> {}

impl<const I: u16, const R: u16, const F: u16, H: Unsigned> AlgorithmName for CubeHashCore<I, R, F, H> {
    fn write_alg_name(f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "CubeHash<{}, {}, {}, {}>", I, R, F, H::USIZE)
    }
}

impl<const I: u16, const R: u16, const F: u16, H> BlockSizeUser for CubeHashCore<I, R, F, H> {
    type BlockSize = U32;
}

impl<const I: u16, const R: u16, const F: u16, H: ArraySize> OutputSizeUser for CubeHashCore<I, R, F, H> {
    type OutputSize = H;
}

impl<const I: u16, const R: u16, const F: u16, H> BufferKindUser for CubeHashCore<I, R, F, H> {
    type BufferKind = Eager;
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BackendSelector {
    Soft,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Sse2,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2,
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unstable-avx512"))]
    Avx512,
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    Neon
}

impl<const I: u16, const R: u16, const F: u16, H: Unsigned> CubeHashCore<I, R, F, H> {
    pub fn new_with_backend(backend: BackendSelector) -> Option<Self> {
        match backend {
            BackendSelector::Soft => Some(Self(Backend::Soft(unsafe { soft::Soft::init() }))),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            BackendSelector::Sse2 => cpu_sse2::get().then(|| Self(Backend::Sse2(unsafe { sse2::Sse2::init() }))),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            BackendSelector::Avx2 => cpu_avx2::get().then(|| Self(Backend::Avx2(unsafe { avx2::Avx2::init() }))),
            #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unstable-avx512"))]
            BackendSelector::Avx512 => cpu_avx512::get().then(|| Self(Backend::Avx512(unsafe { avx512::Avx512::init() }))),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))] 
            BackendSelector::Neon => cpu_neon::get().then(|| Self(Backend::Neon(unsafe { neon::Neon::init() }))),
        }
    }
}

impl<const I: u16, const R: u16, const F: u16, H: Unsigned> Default for CubeHashCore<I, R, F, H> {
    fn default() -> Self {
        cfg_if::cfg_if! {
            if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
                /*
                 * Some interesting notes:
                 * 
                 * On znver4 (my desktop), when compiled with -C target-cpu=native,
                 * SSE2 blows AVX2 and AVX512 out of the water. Otherwise, it's AVX2.
                 * (My guess is AMD went hard into AVX512VL instructions, and therefore
                 *  SSE2 under VL runs fastest because it maximizes register spread
                 *  (enabling parallelism) and has less overall operations (because two
                 *  swaps become nothing))
                 * 
                 * On icelake-server, AVX512 is consistently the winner and performs
                 * equally with and without -C target-cpu=native. With it, SSE2 is
                 * competitive with AVX512. Without, SSE2 and AVX2 perform equally.
                 * (this was virtualized so YMMV)
                 * 
                 * Conclusion: AVX512VL is a boon to existing SSE2-intrinsic-using code
                 *             that performs frequent swapping at a granularity of at
                 *             least 128 bits (at least on Zen 4). Consider messing with
                 *             target_feature for performance.
                 * 
                 * Numbers:
                 * 
                 * Core           | Native | Set    | Cycles/byte
                 * ---------------+--------+--------+------------
                 * znver4         | No     | SSE2   | 6.3
                 * znver4         | No     | AVX2   | 5.7
                 * znver4         | No     | AVX512 | 6.5
                 * znver4         | Yes    | SSE2   | 4.5
                 * znver4         | Yes    | AVX2   | 5.2
                 * znver4         | Yes    | AVX512 | 6.5
                 * icelake-server | No     | SSE2   | 6.4
                 * icelake-server | No     | AVX2   | 6.8
                 * icelake-server | No     | AVX512 | 3.9
                 * icelake-server | Yes    | SSE2   | 4.3
                 * icelake-server | Yes    | AVX2   | 5.3
                 * icelake-server | Yes    | AVX512 | 3.9
                 */

                #[cfg(feature = "unstable-avx512")]
                if cpu_avx512::get() {
                    return Self(Backend::Avx512(unsafe { avx512::Avx512::init() }))
                }

                if cpu_avx2::get() {
                    return Self(Backend::Avx2(unsafe { avx2::Avx2::init() }))
                }

                if cpu_sse2::get() {
                    return Self(Backend::Sse2(unsafe { sse2::Sse2::init() }))
                }
            } else if #[cfg(all(target_arch = "aarch64", target_endian = "little"))] {
                if cpu_neon::get() {
                    return Self(Backend::Neon(unsafe { neon::Neon::init() }))
                }
            }
        }

        Self(Backend::Soft(unsafe { soft::Soft::init() }))
    }
}

impl<const I: u16, const R: u16, const F: u16, H> UpdateCore for CubeHashCore<I, R, F, H> {
    fn update_blocks(&mut self, blocks: &[Block<Self>]) {
        match self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2(ref mut b) => unsafe {
                for block in blocks.iter() { b.update_block(block) } }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx2(ref mut b) => unsafe {
                for block in blocks.iter() { b.update_block(block) } }
            #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unstable-avx512"))]
            Backend::Avx512(ref mut b) => unsafe {
                for block in blocks.iter() { b.update_block(block) } },
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Backend::Neon(ref mut b) => unsafe {
                for block in blocks.iter() { b.update_block(block) } },
            Backend::Soft(ref mut b) => unsafe {
                for block in blocks.iter() { b.update_block(block) } }
        }
    }
}

impl<const I: u16, const R: u16, const F: u16, H: ArraySize + IsGreater<U0, Output = True> + IsLessOrEqual<U64, Output = True>> FixedOutputCore for CubeHashCore<I, R, F, H> {
    fn finalize_fixed_core(&mut self, buffer: &mut Buffer<Self>, out: &mut Output<Self>) {
        buffer.digest_pad(0x80, &[], |block| match self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2(ref mut b) => unsafe { b.update_block(block) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx2(ref mut b) => unsafe { b.update_block(block) },
            #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unstable-avx512"))]
            Backend::Avx512(ref mut b) => unsafe { b.update_block(block) },
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Backend::Neon(ref mut b) => unsafe { b.update_block(block) },
            Backend::Soft(ref mut b) => unsafe { b.update_block(block) }
        });
        match self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Sse2(ref mut b) => unsafe { b.finalize(out) },
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Backend::Avx2(ref mut b) => unsafe { b.finalize(out) },
            #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unstable-avx512"))]
            Backend::Avx512(ref mut b) => unsafe { b.finalize(out) },
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Backend::Neon(ref mut b) => unsafe { b.finalize(out) },
            Backend::Soft(ref mut b) => unsafe { b.finalize(out) }
        }
    }
}

#[cfg(test)]
mod test {
    extern crate alloc;

    use digest::{Digest as _, core_api::CoreWrapper, consts::U56};

    use super::*;

    type CubeHash448 = CoreWrapper<CubeHashCore<16, 16, 32, U56>>;

    #[inline]
    fn control() -> CubeHash448 {
        CubeHash448::from_core(CubeHashCore(Backend::Soft(unsafe { soft::Soft::init() })))
    }

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unstable-avx512"))]
    #[test]
    fn avx512_consistent() {
        if !cpu_avx512::get() {
            panic!("test cannot run on this system");
        }

        let data = alloc::vec![69; 1048574];

        let chash = {
            let mut control = control();
            control.update(&data);
            control.finalize()
        };

        let thash = {
            let mut uut = CubeHash448::from_core(CubeHashCore(Backend::Avx512(unsafe { avx512::Avx512::init() })));
            uut.update(&data);
            uut.finalize()
        };

        assert_eq!(chash, thash);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn sse2_consistent() {
        if !cpu_sse2::get() {
            panic!("test cannot run on this system");
        }

        let data = alloc::vec![69; 1048574];

        let chash = {
            let mut control = control();
            control.update(&data);
            control.finalize()
        };

        let thash = {
            let mut uut = CubeHash448::from_core(CubeHashCore(Backend::Sse2(unsafe { sse2::Sse2::init() })));
            uut.update(&data);
            uut.finalize()
        };

        assert_eq!(chash, thash);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn avx2_consistent() {
        if !cpu_avx2::get() {
            panic!("test cannot run on this system");
        }

        let data = alloc::vec![69; 1048574];

        let chash = {
            let mut control = control();
            control.update(&data);
            control.finalize()
        };

        let thash = {
            let mut uut = CubeHash448::from_core(CubeHashCore(Backend::Avx2(unsafe { avx2::Avx2::init() })));
            uut.update(&data);
            uut.finalize()
        };

        assert_eq!(chash, thash);
    }

    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    #[test]
    fn neon_consistent() {
        if !cpu_neon::get() {
            panic!("test cannot run on this system");
        }

        let data = alloc::vec![69; 1048574];

        let chash = {
            let mut control = control();
            control.update(&data);
            control.finalize()
        };

        let thash = {
            let mut uut = CubeHash448::from_core(CubeHashCore(Backend::Neon(unsafe { neon::Neon::init() })));
            uut.update(&data);
            uut.finalize()
        };

        assert_eq!(chash, thash);
    }
}