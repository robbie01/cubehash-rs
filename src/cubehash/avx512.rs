#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::{marker::PhantomData, mem};

use super::{CubeHashCore, CubeHashBackend};
use digest::{array::{Array, ArraySize}, consts::U32, core_api::BlockSizeUser, typenum::{consts::{U0, U64}, IsGreater, IsLessOrEqual, True, Unsigned}};
use static_assertions::const_assert_eq;

const_assert_eq!(<CubeHashCore::<16, 16, 32, U64> as BlockSizeUser>::BlockSize::USIZE, mem::size_of::<__m256i>());

#[derive(Clone)]
pub struct Avx512<const I: u16, const R: u16, const F: u16, H> {
    r0: __m512i,
    r1: __m512i,
    _phantom: PhantomData<H>
}

impl<const I: u16, const R: u16, const F: u16, H> Avx512<I, R, F, H> {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn round(&mut self) {
        let Self { r0, r1, .. } = self;
        *r1 = _mm512_add_epi32(*r0, *r1);
        *r0 = _mm512_shuffle_i32x4(*r0, *r0, 0x4E);
        *r0 = _mm512_rol_epi32(*r0, 7);
        *r0 = _mm512_xor_epi32(*r0, *r1);
        *r1 = _mm512_shuffle_epi32(*r1, 0x4E);
        *r1 = _mm512_add_epi32(*r0, *r1);
        *r0 = _mm512_permutex_epi64(*r0, 0x4E);
        *r0 = _mm512_rol_epi32(*r0, 11);
        *r0 = _mm512_xor_epi32(*r0, *r1);
        *r1 = _mm512_shuffle_epi32(*r1, 0xB1);
    }
}

impl<const I: u16, const R: u16, const F: u16, H> CubeHashBackend<I, R, F, H> for Avx512<I, R, F, H> {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn init() -> Self where H: Unsigned {
        let mut init = Self {
            r0: _mm512_setr_epi32(
                H::I32,
                <CubeHashCore<I, R, F, H> as BlockSizeUser>::BlockSize::I32,
                R.into(),
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ),
            r1: _mm512_setzero_epi32(),
            _phantom: PhantomData
        };
        // the compiler precomputes this?? incredible
        for _ in 0..I {
            init.round();
        }
        init
    }

    #[inline]
    #[target_feature(enable = "avx,avx512f")]
    unsafe fn update_block(&mut self, block: &Array<u8, U32>) {
        let c = _mm512_zextsi256_si512(_mm256_loadu_si256(block.as_ptr() as *const __m256i));
        self.r0 = _mm512_xor_epi32(self.r0, c);
        for _ in 0..R {
            self.round();
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn finalize(&mut self, out: &mut Array<u8, H>) where H: ArraySize + IsGreater<U0, Output = True> + IsLessOrEqual<U64, Output = True> {
        self.r1 = _mm512_xor_epi32(
            self.r1,
            _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
        );
        for _ in 0..F {
            self.round();
        }
        if H::USIZE == 64 {
            _mm512_storeu_epi32(out.as_mut_ptr() as *mut i32, self.r0);
        } else {
            let mut buf = [0; 64];
            _mm512_storeu_epi32(buf.as_mut_ptr() as *mut i32, self.r0);
            let l = out.len();
            out.copy_from_slice(&buf[..l]);
        }
    }
}