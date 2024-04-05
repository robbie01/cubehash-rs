#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::{iter, marker::PhantomData, mem};

use super::{CubeHashCore, CubeHashBackend};
use digest::{array::{Array, ArraySize}, consts::U32, core_api::BlockSizeUser, typenum::{consts::{U0, U64}, IsGreater, IsLessOrEqual, True, Unsigned}};
use static_assertions::const_assert_eq;

const_assert_eq!(<CubeHashCore::<16, 16, 32, U64> as BlockSizeUser>::BlockSize::USIZE, mem::size_of::<__m256i>());

#[derive(Clone)]
pub struct Avx2<const I: u16, const R: u16, const F: u16, H> {
    r00: __m256i,
    r01: __m256i,
    r10: __m256i,
    r11: __m256i,
    _phantom: PhantomData<H>
}

impl<const I: u16, const R: u16, const F: u16, H> Avx2<I, R, F, H> {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn round(&mut self) {
        let Self { r00, r01, r10, r11, .. } = *self;

        // 1. add
        let r10 = _mm256_add_epi32(r10, r00);
        let r11 = _mm256_add_epi32(r11, r01);

        // 2. rotate
        let r00 = _mm256_or_si256(
            _mm256_slli_epi32(r00, 7),
            _mm256_srli_epi32(r00, 25)
        );
        let r01 = _mm256_or_si256(
            _mm256_slli_epi32(r01, 7),
            _mm256_srli_epi32(r01, 25)
        );

        // 3. swap
        let (r00, r01) = (r01, r00);

        // 4. xor
        let r00 = _mm256_xor_si256(r00, r10);
        let r01 = _mm256_xor_si256(r01, r11);

        // 5. swap
        let r10 = _mm256_shuffle_epi32(r10, 0x4E);
        let r11 = _mm256_shuffle_epi32(r11, 0x4E);

        // 6. add
        let r10 = _mm256_add_epi32(r10, r00);
        let r11 = _mm256_add_epi32(r11, r01);

        // 7. rotate
        let r00 = _mm256_or_si256(
            _mm256_slli_epi32(r00, 11),
            _mm256_srli_epi32(r00, 21)
        );
        let r01 = _mm256_or_si256(
            _mm256_slli_epi32(r01, 11),
            _mm256_srli_epi32(r01, 21)
        );

        // 8. swap
        let r00 = _mm256_permute4x64_epi64(r00, 0x4E);
        let r01 = _mm256_permute4x64_epi64(r01, 0x4E);

        // 9. xor
        let r00 = _mm256_xor_si256(r00, r10);
        let r01 = _mm256_xor_si256(r01, r11);

        // 10. swap
        let r10 = _mm256_shuffle_epi32(r10, 0xB1);
        let r11 = _mm256_shuffle_epi32(r11, 0xB1);

        self.r00 = r00;
        self.r01 = r01;
        self.r10 = r10;
        self.r11 = r11;
    }
}

impl<const I: u16, const R: u16, const F: u16, H> CubeHashBackend<I, R, F, H> for Avx2<I, R, F, H> {
    #[inline]
    #[target_feature(enable = "avx,avx2")]
    unsafe fn init() -> Self where H: Unsigned {
        let mut init = Self {
            r00: _mm256_setr_epi32(
                H::I32,
                <CubeHashCore<I, R, F, H> as BlockSizeUser>::BlockSize::I32,
                R.into(),
                0, 0, 0, 0, 0
            ),
            r01: _mm256_setzero_si256(),
            r10: _mm256_setzero_si256(),
            r11: _mm256_setzero_si256(),
            _phantom: PhantomData
        };
        // the compiler precomputes this?? incredible
        for _ in 0..I {
            init.round();
        }
        init
    }

    #[inline]
    #[target_feature(enable = "avx,avx2")]
    unsafe fn update_block(&mut self, block: &Array<u8, U32>) {
        self.r00 = _mm256_xor_si256(self.r00, _mm256_loadu_si256(block.as_ptr() as *const __m256i));
        for _ in 0..R {
            self.round();
        }
    }

    #[inline]
    #[target_feature(enable = "avx,avx2")]
    unsafe fn finalize(&mut self, out: &mut Array<u8, H>) where H: ArraySize + IsGreater<U0, Output = True> + IsLessOrEqual<U64, Output = True> {
        self.r11 = _mm256_xor_si256(
            self.r11,
            _mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, 1)
        );

        for _ in 0..F {
            self.round();
        }

        let &mut Self { r00, r01, .. } = self;
        
        for (chunk, r) in iter::zip(out.chunks_mut(32), [r00, r01]) {
            if chunk.len() == 32 {
                _mm256_storeu_si256(chunk.as_mut_ptr() as *mut __m256i, r);
            } else {
                let mut buf = [0; 32];
                _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, r);
                let l = chunk.len();
                chunk.copy_from_slice(&buf[..l]);
            }
        }
    }
}

#[cfg(feature = "zeroize")]
impl<const I: u16, const R: u16, const F: u16, H> digest::zeroize::Zeroize for Avx2<I, R, F, H> {
    fn zeroize(&mut self) {
        let Self { r00, r01, r10, r11, .. } = self;
        r00.zeroize();
        r01.zeroize();
        r10.zeroize();
        r11.zeroize();
    }
}