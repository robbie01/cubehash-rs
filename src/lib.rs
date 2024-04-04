#![no_std]
#![cfg_attr(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unstable-avx512"), feature(stdarch_x86_avx512))]
#![cfg_attr(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unstable-avx512"), feature(avx512_target_feature))]

mod cubehash;
mod cubemac;

use digest::{core_api::CoreWrapper, typenum::consts::{U16, U20, U28, U32, U48, U64}};

pub use digest::{self, Digest, Mac, KeyInit};

#[cfg(feature = "selectable-backend")]
pub use cubehash::BackendSelector as CubeHashBackend;

pub use cubehash::CubeHashCore;
type CubeHash<H> = CoreWrapper<CubeHashCore<16, 16, 32, H>>;
pub type CubeHash128 = CubeHash<U16>;
pub type CubeHash160 = CubeHash<U20>;
pub type CubeHash224 = CubeHash<U28>;
pub type CubeHash256 = CubeHash<U32>;
pub type CubeHash384 = CubeHash<U48>;
pub type CubeHash512 = CubeHash<U64>;

pub use cubemac::CubeMacCore;
pub type CubeMac128 = CoreWrapper<CubeMacCore<16, 16, 32, U16>>;

#[cfg(test)]
mod test {
    extern crate alloc;
    use super::*;

    #[test]
    fn cubemac_consistent() {
        let k = [10; 64];
        let d = alloc::vec![0; 1048574];

        let cmac = {
            let mut h = CubeHash128::new();
            h.update(&k);
            h.update(&d);
            h.finalize()
        };

        let mut m = CubeMac128::new_from_slice(&k).unwrap();
        m.update(&d);
        m.verify(&cmac).unwrap();
    }
}