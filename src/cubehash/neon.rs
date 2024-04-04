use core::{arch::aarch64::*, marker::PhantomData, iter, mem};

use super::{CubeHashCore, CubeHashBackend};
use digest::{core_api::BlockSizeUser, array::{Array, ArraySize}, typenum::{consts::{U0, U32, U64}, IsGreater, IsLessOrEqual, True, Unsigned}};
use static_assertions::const_assert_eq;

const_assert_eq!(<CubeHashCore::<16, 16, 32, U64> as BlockSizeUser>::BlockSize::USIZE, 2*mem::size_of::<uint32x4_t>());

#[derive(Clone)]
pub struct Neon<const I: u16, const R: u16, const F: u16, H> {
    r000: uint32x4_t,
    r001: uint32x4_t,
    r010: uint32x4_t,
    r011: uint32x4_t,
    r100: uint32x4_t,
    r101: uint32x4_t,
    r110: uint32x4_t,
    r111: uint32x4_t,
    _phantom: PhantomData<H>
}

impl<const I: u16, const R: u16, const F: u16, H> Neon<I, R, F, H> {
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn round(&mut self) {
        let Self { r000, r001, r010, r011, r100, r101, r110, r111, .. } = self;

        // 1. add
        *r100 = vaddq_u32(*r100, *r000);
        *r101 = vaddq_u32(*r101, *r001);
        *r110 = vaddq_u32(*r110, *r010);
        *r111 = vaddq_u32(*r111, *r011);

        // 2-3. rotate, swap
        {
            let p000 = *r000;
            let p001 = *r001;
            let p010 = *r010;
            let p011 = *r011;

            *r010 = vsriq_n_u32(vshlq_n_u32(p000, 7), p000, 25);
            *r011 = vsriq_n_u32(vshlq_n_u32(p001, 7), p001, 25);
            *r000 = vsriq_n_u32(vshlq_n_u32(p010, 7), p010, 25);
            *r001 = vsriq_n_u32(vshlq_n_u32(p011, 7), p011, 25);
        }

        // 4. xor
        *r000 = veorq_u32(*r000, *r100);
        *r001 = veorq_u32(*r001, *r101);
        *r010 = veorq_u32(*r010, *r110);
        *r011 = veorq_u32(*r011, *r111);

        // 5. swap
        {
            let f100 = vreinterpretq_f32_u32(*r100);
            let f101 = vreinterpretq_f32_u32(*r101);
            let f110 = vreinterpretq_f32_u32(*r110);
            let f111 = vreinterpretq_f32_u32(*r111);

            *r100 = vreinterpretq_u32_f32(vcombine_f32(
                vget_high_f32(f100),
                vget_low_f32(f100)
            ));
            *r101 = vreinterpretq_u32_f32(vcombine_f32(
                vget_high_f32(f101),
                vget_low_f32(f101)
            ));
            *r110 = vreinterpretq_u32_f32(vcombine_f32(
                vget_high_f32(f110),
                vget_low_f32(f110)
            ));
            *r111 = vreinterpretq_u32_f32(vcombine_f32(
                vget_high_f32(f111),
                vget_low_f32(f111)
            ));
        }

        // 6. add
        *r100 = vaddq_u32(*r100, *r000);
        *r101 = vaddq_u32(*r101, *r001);
        *r110 = vaddq_u32(*r110, *r010);
        *r111 = vaddq_u32(*r111, *r011);

        // 7-8. rotate, swap
        {
            let p000 = *r000;
            let p001 = *r001;
            let p010 = *r010;
            let p011 = *r011;

            *r001 = vsriq_n_u32(vshlq_n_u32(p000, 11), p000, 21);
            *r000 = vsriq_n_u32(vshlq_n_u32(p001, 11), p001, 21);
            *r011 = vsriq_n_u32(vshlq_n_u32(p010, 11), p010, 21);
            *r010 = vsriq_n_u32(vshlq_n_u32(p011, 11), p011, 21);
        }

        // 9. xor
        *r000 = veorq_u32(*r000, *r100);
        *r001 = veorq_u32(*r001, *r101);
        *r010 = veorq_u32(*r010, *r110);
        *r011 = veorq_u32(*r011, *r111);

        // 10. swap
        *r100 = vrev64q_u32(*r100);
        *r101 = vrev64q_u32(*r101);
        *r110 = vrev64q_u32(*r110);
        *r111 = vrev64q_u32(*r111);
    }
}

impl<const I: u16, const R: u16, const F: u16, H> CubeHashBackend<I, R, F, H> for Neon<I, R, F, H> {
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn init() -> Self where H: Unsigned {
        let mut init = Self {
            r000: vld1q_u32([
                H::U32,
                <CubeHashCore<I, R, F, H> as BlockSizeUser>::BlockSize::U32,
                R.into(),
                0
            ].as_ptr()),
            r001: vmovq_n_u32(0),
            r010: vmovq_n_u32(0),
            r011: vmovq_n_u32(0),
            r100: vmovq_n_u32(0),
            r101: vmovq_n_u32(0),
            r110: vmovq_n_u32(0),
            r111: vmovq_n_u32(0),
            _phantom: PhantomData
        };
        // the compiler precomputes this?? incredible
        for _ in 0..I {
            init.round();
        }
        init
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn update_block(&mut self, block: &Array<u8, U32>) {
        let x000 = vld1q_u32(block.as_ptr() as *const u32);
        let x001 = vld1q_u32(block[16..].as_ptr() as *const u32);
        self.r000 = veorq_u32(self.r000, x000);
        self.r001 = veorq_u32(self.r001, x001);
        for _ in 0..R {
            self.round();
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn finalize(&mut self, out: &mut Array<u8, H>) where H: ArraySize + IsGreater<U0, Output = True> + IsLessOrEqual<U64, Output = True> {
        self.r111 = veorq_u32(
            self.r111,
            vld1q_u32([0, 0, 0, 1].as_ptr())
        );

        for _ in 0..F {
            self.round();
        }

        let &mut Self { r000, r001, r010, r011, .. } = self;

        for (chunk, r) in iter::zip(out.chunks_mut(16), [r000, r001, r010, r011]) {
            if chunk.len() == 16 {
                vst1q_u32(chunk.as_mut_ptr() as *mut u32, r);
            } else {
                let mut buf = [0; 16];
                vst1q_u32(buf.as_mut_ptr() as *mut u32, r);
                let l = chunk.len();
                chunk.copy_from_slice(&buf[..l]);
            }
        }
    }
}