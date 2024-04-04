use core::slice;

use digest::{block_buffer::Eager, core_api::{AlgorithmName, Block, BlockSizeUser, BufferKindUser, FixedOutputCore, UpdateCore}, crypto_common::KeySizeUser, array::ArraySize, typenum::{IsGreater, IsLessOrEqual, True, Unsigned, U0, U64}, KeyInit, MacMarker, OutputSizeUser};

use super::cubehash::CubeHashCore;

#[derive(Clone)]
pub struct CubeMacCore<const I: u16, const R: u16, const F: u16, H>(CubeHashCore<I, R, F, H>);

impl<const I: u16, const R: u16, const F: u16, H> MacMarker for CubeMacCore<I, R, F, H> {}

impl<const I: u16, const R: u16, const F: u16, H: Unsigned> AlgorithmName for CubeMacCore<I, R, F, H> {
    fn write_alg_name(f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "CubeMac<{}, {}, {}, {}>", I, R, F, H::USIZE)
    }
}

impl<const I: u16, const R: u16, const F: u16, H> BlockSizeUser for CubeMacCore<I, R, F, H> {
    type BlockSize = <CubeHashCore<I, R, F, H> as BlockSizeUser>::BlockSize;
}

impl<const I: u16, const R: u16, const F: u16, H: ArraySize> OutputSizeUser for CubeMacCore<I, R, F, H> {
    type OutputSize = H;
}

impl<const I: u16, const R: u16, const F: u16, H> BufferKindUser for CubeMacCore<I, R, F, H> {
    type BufferKind = Eager;
}

impl<const I: u16, const R: u16, const F: u16, H> KeySizeUser for CubeMacCore<I, R, F, H> {
    type KeySize = U64;
}

impl<const I: u16, const R: u16, const F: u16, H: Unsigned> KeyInit for CubeMacCore<I, R, F, H> {
    #[inline]
    fn new(key: &digest::Key<Self>) -> Self {
        let mut init = CubeHashCore::default();
        let (low, high) = key.split_ref();
        init.update_blocks(slice::from_ref(low));
        init.update_blocks(slice::from_ref(high));
        Self(init)
    }
}

impl<const I: u16, const R: u16, const F: u16, H> UpdateCore for CubeMacCore<I, R, F, H> {
    #[inline(always)]
    fn update_blocks(&mut self, blocks: &[Block<Self>]) {
        self.0.update_blocks(blocks)
    }
}

impl<const I: u16, const R: u16, const F: u16, H: ArraySize + IsGreater<U0, Output = True> + IsLessOrEqual<U64, Output = True>> FixedOutputCore for CubeMacCore<I, R, F, H> {
    #[inline(always)]
    fn finalize_fixed_core(&mut self, buffer: &mut digest::core_api::Buffer<Self>, out: &mut digest::Output<Self>) {
        self.0.finalize_fixed_core(buffer, out)
    }
}