#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::{iter, marker::PhantomData, mem};

use super::{CubeHashCore, CubeHashBackend};
use digest::{array::{Array, ArraySize}, consts::U32, core_api::BlockSizeUser, typenum::{consts::{U0, U64}, IsGreater, IsLessOrEqual, True, Unsigned}};
use static_assertions::const_assert_eq;

const_assert_eq!(<CubeHashCore::<16, 16, 32, U64> as BlockSizeUser>::BlockSize::USIZE, 2*mem::size_of::<__m128i>());

#[derive(Clone)]
pub struct Sse2<const I: u16, const R: u16, const F: u16, H> {
    r000: __m128i,
    r001: __m128i,
    r010: __m128i,
    r011: __m128i,
    r100: __m128i,
    r101: __m128i,
    r110: __m128i,
    r111: __m128i,
    _phantom: PhantomData<H>
}

impl<const I: u16, const R: u16, const F: u16, H> Sse2<I, R, F, H> {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn round(&mut self) {
        let Self { r000, r001, r010, r011, r100, r101, r110, r111, .. } = *self;

        // 1. add
        let r100 = _mm_add_epi32(r100, r000);
        let r101 = _mm_add_epi32(r101, r001);
        let r110 = _mm_add_epi32(r110, r010);
        let r111 = _mm_add_epi32(r111, r011);

        // 2. rotate
        let r000 = _mm_or_si128(
            _mm_slli_epi32(r000, 7),
            _mm_srli_epi32(r000, 25)
        );
        let r001 = _mm_or_si128(
            _mm_slli_epi32(r001, 7),
            _mm_srli_epi32(r001, 25)
        );
        let r010 = _mm_or_si128(
            _mm_slli_epi32(r010, 7),
            _mm_srli_epi32(r010, 25)
        );
        let r011 = _mm_or_si128(
            _mm_slli_epi32(r011, 7),
            _mm_srli_epi32(r011, 25)
        );
        
        // 3. swap
        let (r000, r010) = (r010, r000);
        let (r001, r011) = (r011, r001);

        // 4. xor
        let r000 = _mm_xor_si128(r000, r100);
        let r001 = _mm_xor_si128(r001, r101);
        let r010 = _mm_xor_si128(r010, r110);
        let r011 = _mm_xor_si128(r011, r111);

        // 5. swap
        let r100 = _mm_shuffle_epi32(r100, 0x4E);
        let r101 = _mm_shuffle_epi32(r101, 0x4E);
        let r110 = _mm_shuffle_epi32(r110, 0x4E);
        let r111 = _mm_shuffle_epi32(r111, 0x4E);

        // 6. add
        let r100 = _mm_add_epi32(r100, r000);
        let r101 = _mm_add_epi32(r101, r001);
        let r110 = _mm_add_epi32(r110, r010);
        let r111 = _mm_add_epi32(r111, r011);

        // 7. rotate
        let r000 = _mm_or_si128(
            _mm_slli_epi32(r000, 11),
            _mm_srli_epi32(r000, 21)
        );
        let r001 = _mm_or_si128(
            _mm_slli_epi32(r001, 11),
            _mm_srli_epi32(r001, 21)
        );
        let r010 = _mm_or_si128(
            _mm_slli_epi32(r010, 11),
            _mm_srli_epi32(r010, 21)
        );
        let r011 = _mm_or_si128(
            _mm_slli_epi32(r011, 11),
            _mm_srli_epi32(r011, 21)
        );

        // 8. swap
        let (r000, r001) = (r001, r000);
        let (r010, r011) = (r011, r010);

        // 9. xor
        let r000 = _mm_xor_si128(r000, r100);
        let r001 = _mm_xor_si128(r001, r101);
        let r010 = _mm_xor_si128(r010, r110);
        let r011 = _mm_xor_si128(r011, r111);

        // 10. swap
        let r100 = _mm_shuffle_epi32(r100, 0xB1);
        let r101 = _mm_shuffle_epi32(r101, 0xB1);
        let r110 = _mm_shuffle_epi32(r110, 0xB1);
        let r111 = _mm_shuffle_epi32(r111, 0xB1);

        self.r000 = r000;
        self.r001 = r001;
        self.r010 = r010;
        self.r011 = r011;
        self.r100 = r100;
        self.r101 = r101;
        self.r110 = r110;
        self.r111 = r111;
    }
}

impl<const I: u16, const R: u16, const F: u16, H> CubeHashBackend<I, R, F, H> for Sse2<I, R, F, H> {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn init() -> Self where H: Unsigned {
        let mut init = Self {
            r000: _mm_setr_epi32(
                H::I32,
                <CubeHashCore<I, R, F, H> as BlockSizeUser>::BlockSize::I32,
                R.into(),
                0
            ),
            r001: _mm_setzero_si128(),
            r010: _mm_setzero_si128(),
            r011: _mm_setzero_si128(),
            r100: _mm_setzero_si128(),
            r101: _mm_setzero_si128(),
            r110: _mm_setzero_si128(),
            r111: _mm_setzero_si128(),
            _phantom: PhantomData
        };
        // the compiler precomputes this?? incredible
        for _ in 0..I {
            init.round();
        }
        init
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn update_block(&mut self, block: &Array<u8, U32>) {
        let x000 = _mm_loadu_si128(block.as_ptr() as *const __m128i);
        let x001 = _mm_loadu_si128(block[16..].as_ptr() as *const __m128i);
        self.r000 = _mm_xor_si128(self.r000, x000);
        self.r001 = _mm_xor_si128(self.r001, x001);
        for _ in 0..R {
            self.round();
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn finalize(&mut self, out: &mut Array<u8, H>) where H: ArraySize + IsGreater<U0, Output = True> + IsLessOrEqual<U64, Output = True> {
        self.r111 = _mm_xor_si128(
            self.r111,
            _mm_setr_epi32(0, 0, 0, 1)
        );

        for _ in 0..F {
            self.round();
        }

        let &mut Self { r000, r001, r010, r011, .. } = self;
        
        for (chunk, r) in iter::zip(out.chunks_mut(16), [r000, r001, r010, r011]) {
            if chunk.len() == 16 {
                _mm_storeu_si128(chunk.as_mut_ptr() as *mut __m128i, r);
            } else {
                let mut buf = [0; 16];
                _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, r);
                let l = chunk.len();
                chunk.copy_from_slice(&buf[..l]);
            }
        }
    }
}