use core::{iter, marker::PhantomData};

use super::{CubeHashCore, CubeHashBackend};
use digest::{array::{Array, ArraySize}, consts::U32, core_api::BlockSizeUser, typenum::{consts::{U0, U64}, IsGreater, IsLessOrEqual, True, Unsigned}};

#[derive(Clone)]
pub struct Soft<const I: u16, const R: u16, const F: u16, H> {
    r: [u32; 32],
    _phantom: PhantomData<H>
}

impl<const I: u16, const R: u16, const F: u16, H> Soft<I, R, F, H> {
    #[inline]
    fn round(&mut self) {
        // Direct calque of the eBASH simple version
        let Self { r, .. } = self;
        let mut tmp = [0; 16];

        for i in 0..16 {
            r[i+16] = r[i+16].wrapping_add(r[i]);
        }
        for i in 0..16 {
            tmp[i ^ 0b1000] = r[i];
        }
        for i in 0..16 {
            r[i] = tmp[i].rotate_left(7);
        }
        for i in 0..16 {
            r[i] ^= r[i+16];
        }
        for i in 0..16 {
            tmp[i ^ 0b10] = r[i | 0b10000];
        }
        r[16..].copy_from_slice(&tmp);
        for i in 0..16 {
            r[i+16] = r[i+16].wrapping_add(r[i]);
        }
        for i in 0..16 {
            tmp[i ^ 0b100] = r[i];
        }
        for i in 0..16 {
            r[i] = tmp[i].rotate_left(11);
        }
        for i in 0..16 {
            r[i] ^= r[i+16];
        }
        for i in 0..16 {
            tmp[i ^ 0b1] = r[i | 0b10000];
        }
        r[16..].copy_from_slice(&tmp);
    }
}

impl<const I: u16, const R: u16, const F: u16, H> CubeHashBackend<I, R, F, H> for Soft<I, R, F, H> {
    #[inline]
    unsafe fn init() -> Self where H: Unsigned {
        let mut init = Self {
            r: [
                H::U32,
                <CubeHashCore<I, R, F, H> as BlockSizeUser>::BlockSize::U32,
                R.into(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ],
            _phantom: PhantomData
        };
        for _ in 0..I {
            init.round();
        }
        init
    }

    #[inline]
    unsafe fn update_block(&mut self, block: &Array<u8, U32>) {
        for (word, chunk) in iter::zip(self.r.iter_mut(), block.chunks_exact(4)) {
            *word ^= u32::from_le_bytes(chunk.try_into().unwrap())
        }

        for _ in 0..R {
            self.round();
        }
    }

    #[inline]
    unsafe fn finalize(&mut self, out: &mut Array<u8, H>) where H: ArraySize + IsGreater<U0, Output = True> + IsLessOrEqual<U64, Output = True> {
        self.r[31] ^= 1;

        for _ in 0..F {
            self.round();
        }

        for (chunk, word) in iter::zip(out.iter_mut(), self.r.iter().flat_map(|x| x.to_le_bytes())) {
            *chunk = word;
        }
    }
}