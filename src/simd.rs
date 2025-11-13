#[cfg(not(any(
    target_feature = "sse",
    target_feature = "simd128",
    target_feature = "avx512f",
    target_feature = "avx",
    all(target_feature = "neon", target_arch = "aarch64")
)))]
compile_error!("Cannot use optimized SIMD code on this platform");

#[cfg(target_feature = "avx512f")]
mod exports {
    pub(crate) type F32Vec = wide::f32x16;
    pub(crate) type I32Vec = wide::i32x16;

    pub(crate) const CHUNK_SIZE: usize = 16;
}

#[cfg(all(not(target_feature = "avx512f"), target_feature = "avx"))]
mod exports {
    pub(crate) type F32Vec = wide::f32x8;
    pub(crate) type I32Vec = wide::i32x8;

    pub(crate) const CHUNK_SIZE: usize = 8;
}

#[cfg(all(not(target_feature = "avx512f"), not(target_feature = "avx")))]
mod exports {
    pub(crate) type F32Vec = wide::f32x4;
    pub(crate) type I32Vec = wide::i32x4;

    pub(crate) const CHUNK_SIZE: usize = 4;
}

pub(crate) use exports::*;

#[inline]
#[allow(unreachable_code)]
const fn horizontal_max(v: [f32; CHUNK_SIZE]) -> f32 {
    #[cfg(target_feature = "avx512f")]
    return f32::max(
        f32::max(
            f32::max(f32::max(v[0], v[1]), f32::max(v[2], v[3])),
            f32::max(f32::max(v[4], v[5]), f32::max(v[6], v[7])),
        ),
        f32::max(
            f32::max(f32::max(v[8], v[9]), f32::max(v[10], v[11])),
            f32::max(f32::max(v[12], v[13]), f32::max(v[14], v[15])),
        ),
    );
    #[cfg(all(not(target_feature = "avx512f"), target_feature = "avx"))]
    return f32::max(
        f32::max(f32::max(v[0], v[1]), f32::max(v[2], v[3])),
        f32::max(f32::max(v[4], v[5]), f32::max(v[6], v[7])),
    );

    f32::max(f32::max(v[0], v[1]), f32::max(v[2], v[3]))
}

#[inline]
pub fn simd_align(x: &[f32]) -> &[[f32; CHUNK_SIZE]] {
    unsafe { x.align_to::<[f32; CHUNK_SIZE]>().1 }
}

#[inline]
pub(crate) fn absmax(x: &[f32]) -> f32 {
    let mut f = F32Vec::splat(0.0);

    for c in simd_align(x) {
        f = f.fast_max(F32Vec::new(*c).abs());
    }

    horizontal_max(f.to_array())
}
