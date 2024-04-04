use std::{arch::x86_64::*, sync::{OnceLock, atomic::{compiler_fence, Ordering}}};
use cubehash::{CubeHash512, CubeHashBackend, CubeHashCore, Digest};

#[inline(always)]
fn square<T: std::ops::Mul<T> + Copy>(x: T) -> <T as std::ops::Mul<T>>::Output {
    x*x
}

fn profile(gen: impl Digest + Clone) {
    static DATA: OnceLock<Vec<u8>> = OnceLock::new();
    let mut hsh = None;

    let data = DATA.get_or_init(|| vec![11; 1024*1024*1024]);
    let mut cpb = Vec::new();

    for _ in 0..20 {
        compiler_fence(Ordering::SeqCst);
        unsafe { __cpuid(0) };
        compiler_fence(Ordering::SeqCst);
        let t0 = unsafe { _rdtsc() };
        compiler_fence(Ordering::SeqCst);
        let mut h = gen.clone();
        h.update(data);
        let hash = h.finalize();
        compiler_fence(Ordering::SeqCst);
        let t1 = unsafe { __rdtscp(&mut 0) };
        compiler_fence(Ordering::SeqCst);
        unsafe { __cpuid(0) };
        compiler_fence(Ordering::SeqCst);
        match hsh {
            None => hsh = Some(hash),
            Some(ref b) => assert_eq!(*b, hash)
        }
        cpb.push((t1-t0) as f32 / data.len() as f32);
    }

    let mean = cpb.iter().copied().sum::<f32>() / cpb.len() as f32;
    let stdev = (cpb.iter().map(|&z| square(z-mean)).sum::<f32>() / cpb.len() as f32).sqrt();

    println!("mean = {mean}, stdev = {stdev}");
}

fn main() {
    //profile(CubeHash512::from_core(CubeHashCore::new_with_backend(CubeHashBackend::Soft).unwrap()));
    let sse2 = CubeHash512::from_core(CubeHashCore::new_with_backend(CubeHashBackend::Sse2).unwrap());
    let avx2 = CubeHash512::from_core(CubeHashCore::new_with_backend(CubeHashBackend::Avx2).unwrap());
    let avx512 = CubeHash512::from_core(CubeHashCore::new_with_backend(CubeHashBackend::Avx512).unwrap());
    // let sha2 = sha2::Sha512::new();

    // print!("SHA-512:         ");
    // profile(sha2);
    print!("cubehash/SSE2:   ");
    profile(sse2);
    print!("cubehash/AVX2:   ");
    profile(avx2);
    print!("cubehash/AVX512: ");
    profile(avx512);
}