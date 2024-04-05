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
        let Self { r000, r001, r010, r011, r100, r101, r110, r111, .. } = *self;

        // 1. add
        let r100 = vaddq_u32(r100, r000);
        let r101 = vaddq_u32(r101, r001);
        let r110 = vaddq_u32(r110, r010);
        let r111 = vaddq_u32(r111, r011);

        // 2. rotate
        let r000 = vsriq_n_u32(vshlq_n_u32(r000, 7), r000, 25);
        let r001 = vsriq_n_u32(vshlq_n_u32(r001, 7), r001, 25);
        let r010 = vsriq_n_u32(vshlq_n_u32(r010, 7), r010, 25);
        let r011 = vsriq_n_u32(vshlq_n_u32(r011, 7), r011, 25);

        // 3. swap
        let (r000, r010) = (r010, r000);
        let (r001, r011) = (r011, r001);

        // 4. xor
        let r000 = veorq_u32(r000, r100);
        let r001 = veorq_u32(r001, r101);
        let r010 = veorq_u32(r010, r110);
        let r011 = veorq_u32(r011, r111);

        // 5. swap
        let r100 = {
            let f100 = vreinterpretq_f32_u32(r100);
            vreinterpretq_u32_f32(vcombine_f32(
                vget_high_f32(f100),
                vget_low_f32(f100)
            ))
        };
        let r101 = {
            let f101 = vreinterpretq_f32_u32(r101);
            vreinterpretq_u32_f32(vcombine_f32(
                vget_high_f32(f101),
                vget_low_f32(f101)
            ))
        };
        let r110 = {
            let f110 = vreinterpretq_f32_u32(r110);
            vreinterpretq_u32_f32(vcombine_f32(
                vget_high_f32(f110),
                vget_low_f32(f110)
            ))
        };
        let r111 = {
            let f111 = vreinterpretq_f32_u32(r111);
            vreinterpretq_u32_f32(vcombine_f32(
                vget_high_f32(f111),
                vget_low_f32(f111)
            ))
        };

        // 6. add
        let r100 = vaddq_u32(r100, r000);
        let r101 = vaddq_u32(r101, r001);
        let r110 = vaddq_u32(r110, r010);
        let r111 = vaddq_u32(r111, r011);

        // 7. rotate
        let r000 = vsriq_n_u32(vshlq_n_u32(r000, 11), r000, 21);
        let r001 = vsriq_n_u32(vshlq_n_u32(r001, 11), r001, 21);
        let r010 = vsriq_n_u32(vshlq_n_u32(r010, 11), r010, 21);
        let r011 = vsriq_n_u32(vshlq_n_u32(r011, 11), r011, 21);

        // 8. swap
        let (r000, r001) = (r001, r000);
        let (r010, r011) = (r011, r010);

        // 9. xor
        let r000 = veorq_u32(r000, r100);
        let r001 = veorq_u32(r001, r101);
        let r010 = veorq_u32(r010, r110);
        let r011 = veorq_u32(r011, r111);

        // 10. swap
        let r100 = vrev64q_u32(r100);
        let r101 = vrev64q_u32(r101);
        let r110 = vrev64q_u32(r110);
        let r111 = vrev64q_u32(r111);

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

#[inline(always)]
fn zeroize_reg(r: &mut uint32x4_t) {
    use core::{ptr, sync::atomic::{self, Ordering}};

    unsafe { ptr::write_volatile(r, mem::zeroed()) };
    atomic::compiler_fence(Ordering::SeqCst);
}

#[cfg(feature = "zeroize")]
impl<const I: u16, const R: u16, const F: u16, H> digest::zeroize::Zeroize for Neon<I, R, F, H> {
    fn zeroize(&mut self) {
        let Self { r000, r001, r010, r011, r100, r101, r110, r111, .. } = self;
        zeroize_reg(r000);
        zeroize_reg(r001);
        zeroize_reg(r010);
        zeroize_reg(r011);
        zeroize_reg(r100);
        zeroize_reg(r101);
        zeroize_reg(r110);
        zeroize_reg(r111);
    }
}