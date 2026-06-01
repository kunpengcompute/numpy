#include "fast_loop_macros.h"
#include "numpy/npy_math.h"
#include "simd/simd.h"
#include "loops_utils.h"
#include "loops.h"
#include "npy_svml.h"
#include "simd/simd.hpp"
#include <hwy/contrib/math/math-inl.h>
#include <cmath>
#include <cstring>

#if NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)

static void
simd_pow_f32(const npyv_lanetype_f32 *src1, npy_intp ssrc1,
             const npyv_lanetype_f32 *src2, npy_intp ssrc2,
                   npyv_lanetype_f32 *dst, npy_intp sdst, npy_intp len)
{
    const int vstep = npyv_nlanes_f32;
    for (; len > 0; len -= vstep, src1 += ssrc1*vstep, src2 += ssrc2*vstep, dst += sdst*vstep) {
        npyv_f32 x1;
        if (ssrc1 == 1) {
            x1 = npyv_load_till_f32(src1, len, 1);
        }
        else {
            x1 = npyv_loadn_till_f32(src1, ssrc1, len, 1);
        }

        npyv_f32 x2;
        if (ssrc2 == 1) {
            x2 = npyv_load_till_f32(src2, len, 1);
        }
        else {
            x2 = npyv_loadn_till_f32(src2, ssrc2, len, 1);
        }

        npyv_f32 out = __svml_powf16(x1, x2);
        if (sdst == 1) {
            npyv_store_till_f32(dst, len, out);
        }
        else {
            npyv_storen_till_f32(dst, sdst, len, out);
        }
    }
    npyv_cleanup();
}

static void
simd_pow_f64(const npyv_lanetype_f64 *src1, npy_intp ssrc1,
             const npyv_lanetype_f64 *src2, npy_intp ssrc2,
                   npyv_lanetype_f64 *dst, npy_intp sdst, npy_intp len)
{
    const int vstep = npyv_nlanes_f64;
    for (; len > 0; len -= vstep, src1 += ssrc1*vstep, src2 += ssrc2*vstep, dst += sdst*vstep) {
        npyv_f64 x1;
        if (ssrc1 == 1) {
            x1 = npyv_load_till_f64(src1, len, 1);
        }
        else {
            x1 = npyv_loadn_till_f64(src1, ssrc1, len, 1);
        }

        npyv_f64 x2;
        if (ssrc2 == 1) {
            x2 = npyv_load_till_f64(src2, len, 1);
        }
        else {
            x2 = npyv_loadn_till_f64(src2, ssrc2, len, 1);
        }

        npyv_f64 out = __svml_pow8_ha(x1, x2);
        if (sdst == 1) {
            npyv_store_till_f64(dst, len, out);
        }
        else {
            npyv_storen_till_f64(dst, sdst, len, out);
        }
    }
    npyv_cleanup();
}

#elif NPY_SIMD_FMA3

namespace hn = hwy::HWY_NAMESPACE;

// ============================================================
// Double-precision pow - following SVE algorithm from optimized-routines
// ============================================================

#define V_POW_EXP_TABLE_BITS 8
#define V_POW_LOG_TABLE_BITS 7
#define V_POW_N_LOG (1 << V_POW_LOG_TABLE_BITS)
#define V_POW_N_EXP (1 << V_POW_EXP_TABLE_BITS)
#define V_POW_OFF 0x3fe6955500000000ULL

/* SVE exp special case constants */
#define V_POW_SIGN_BIAS (0x800ULL << V_POW_EXP_TABLE_BITS)
#define V_POW_SMALL_EXP 0x3c9  /* top12(0x1p-54) */
#define V_POW_BIG_EXP   0x408  /* top12(512.) */
#define V_POW_THRES_EXP 0x03f  /* BigExp - SmallExp */
#define V_POW_HUGE_EXP  0x409  /* top12(1024.) */

struct PowExpData {
    npy_double poly[3];
    npy_double n_over_ln2, ln2_over_n_hi, ln2_over_n_lo, shift;
    uint64_t sbits[V_POW_N_EXP];
};

struct PowLogData {
    npy_double poly[7];
    npy_double ln2_hi, ln2_lo;
    npy_double invc[V_POW_N_LOG];
    npy_double logc[V_POW_N_LOG];
    npy_double logctail[V_POW_N_LOG];
};

static const PowExpData _pow_exp_data = {
    .poly = { 0x1.fffffffffffd4p-2, 0x1.5555571d6ef9p-3, 0x1.5555576a5adcep-5, },
    .n_over_ln2 = 0x1.71547652b82fep0 * V_POW_N_EXP,
    .ln2_over_n_hi = 0x1.62e42fefc0000p-9,
    .ln2_over_n_lo = -0x1.c610ca86c3899p-45,
    .shift = 0x1.8p52,
    .sbits = {
        0x3ff0000000000000, 0x3feffb1afa5abcbf, 0x3feff63da9fb3335, 0x3feff168143b0281, 0x3fefec9a3e778061,
        0x3fefe7d42e11bbcc, 0x3fefe315e86e7f85, 0x3fefde5f72f654b1, 0x3fefd9b0d3158574, 0x3fefd50a0e3c1f89,
        0x3fefd06b29ddf6de, 0x3fefcbd42b72a836, 0x3fefc74518759bc8, 0x3fefc2bdf66607e0, 0x3fefbe3ecac6f383,
        0x3fefb9c79b1f3919, 0x3fefb5586cf9890f, 0x3fefb0f145e46c85, 0x3fefac922b7247f7, 0x3fefa83b23395dec,
        0x3fefa3ec32d3d1a2, 0x3fef9fa55fdfa9c5, 0x3fef9b66affed31b, 0x3fef973028d7233e, 0x3fef9301d0125b51,
        0x3fef8edbab5e2ab6, 0x3fef8abdc06c31cc, 0x3fef86a814f204ab, 0x3fef829aaea92de0, 0x3fef7e95934f312e,
        0x3fef7a98c8a58e51, 0x3fef76a45471c3c2, 0x3fef72b83c7d517b, 0x3fef6ed48695bbc0, 0x3fef6af9388c8dea,
        0x3fef672658375d2f, 0x3fef635beb6fcb75, 0x3fef5f99f8138a1c, 0x3fef5be084045cd4, 0x3fef582f95281c6b,
        0x3fef54873168b9aa, 0x3fef50e75eb44027, 0x3fef4d5022fcd91d, 0x3fef49c18438ce4d, 0x3fef463b88628cd6,
        0x3fef42be3578a819, 0x3fef3f49917ddc96, 0x3fef3bdda27912d1, 0x3fef387a6e756238, 0x3fef351ffb82140a,
        0x3fef31ce4fb2a63f, 0x3fef2e85711ece75, 0x3fef2b4565e27cdd, 0x3fef280e341ddf29, 0x3fef24dfe1f56381,
        0x3fef21ba7591bb70, 0x3fef1e9df51fdee1, 0x3fef1b8a66d10f13, 0x3fef187fd0dad990, 0x3fef157e39771b2f,
        0x3fef1285a6e4030b, 0x3fef0f961f641589, 0x3fef0cafa93e2f56, 0x3fef09d24abd886b, 0x3fef06fe0a31b715,
        0x3fef0432edeeb2fd, 0x3fef0170fc4cd831, 0x3feefeb83ba8ea32, 0x3feefc08b26416ff, 0x3feef96266e3fa2d,
        0x3feef6c55f929ff1, 0x3feef431a2de883b, 0x3feef1a7373aa9cb, 0x3feeef26231e754a, 0x3feeecae6d05d866,
        0x3feeea401b7140ef, 0x3feee7db34e59ff7, 0x3feee57fbfec6cf4, 0x3feee32dc313a8e5, 0x3feee0e544ede173,
        0x3feedea64c123422, 0x3feedc70df1c5175, 0x3feeda4504ac801c, 0x3feed822c367a024, 0x3feed60a21f72e2a,
        0x3feed3fb2709468a, 0x3feed1f5d950a897, 0x3feecffa3f84b9d4, 0x3feece086061892d, 0x3feecc2042a7d232,
        0x3feeca41ed1d0057, 0x3feec86d668b3237, 0x3feec6a2b5c13cd0, 0x3feec4e1e192aed2, 0x3feec32af0d7d3de,
        0x3feec17dea6db7d7, 0x3feebfdad5362a27, 0x3feebe41b817c114, 0x3feebcb299fddd0d, 0x3feebb2d81d8abff,
        0x3feeb9b2769d2ca7, 0x3feeb8417f4531ee, 0x3feeb6daa2cf6642, 0x3feeb57de83f4eef, 0x3feeb42b569d4f82,
        0x3feeb2e2f4f6ad27, 0x3feeb1a4ca5d920f, 0x3feeb070dde910d2, 0x3feeaf4736b527da, 0x3feeae27dbe2c4cf,
        0x3feead12d497c7fd, 0x3feeac0827ff07cc, 0x3feeab07dd485429, 0x3feeaa11fba87a03, 0x3feea9268a5946b7,
        0x3feea84590998b93, 0x3feea76f15ad2148, 0x3feea6a320dceb71, 0x3feea5e1b976dc09, 0x3feea52ae6cdf6f4,
        0x3feea47eb03a5585, 0x3feea3dd1d1929fd, 0x3feea34634ccc320, 0x3feea2b9febc8fb7, 0x3feea23882552225,
        0x3feea1c1c70833f6, 0x3feea155d44ca973, 0x3feea0f4b19e9538, 0x3feea09e667f3bcd, 0x3feea052fa75173e,
        0x3feea012750bdabf, 0x3fee9fdcddd47645, 0x3fee9fb23c651a2f, 0x3fee9f9298593ae5, 0x3fee9f7df9519484,
        0x3fee9f7466f42e87, 0x3fee9f75e8ec5f74, 0x3fee9f8286ead08a, 0x3fee9f9a48a58174, 0x3fee9fbd35d7cbfd,
        0x3fee9feb564267c9, 0x3feea024b1ab6e09, 0x3feea0694fde5d3f, 0x3feea0b938ac1cf6, 0x3feea11473eb0187,
        0x3feea17b0976cfdb, 0x3feea1ed0130c132, 0x3feea26a62ff86f0, 0x3feea2f336cf4e62, 0x3feea3878491c491,
        0x3feea427543e1a12, 0x3feea4d2add106d9, 0x3feea589994cce13, 0x3feea64c1eb941f7, 0x3feea71a4623c7ad,
        0x3feea7f4179f5b21, 0x3feea8d99b4492ed, 0x3feea9cad931a436, 0x3feeaac7d98a6699, 0x3feeabd0a478580f,
        0x3feeace5422aa0db, 0x3feeae05bad61778, 0x3feeaf3216b5448c, 0x3feeb06a5e0866d9, 0x3feeb1ae99157736,
        0x3feeb2fed0282c8a, 0x3feeb45b0b91ffc6, 0x3feeb5c353aa2fe2, 0x3feeb737b0cdc5e5, 0x3feeb8b82b5f98e5,
        0x3feeba44cbc8520f, 0x3feebbdd9a7670b3, 0x3feebd829fde4e50, 0x3feebf33e47a22a2, 0x3feec0f170ca07ba,
        0x3feec2bb4d53fe0d, 0x3feec49182a3f090, 0x3feec674194bb8d5, 0x3feec86319e32323, 0x3feeca5e8d07f29e,
        0x3feecc667b5de565, 0x3feece7aed8eb8bb, 0x3feed09bec4a2d33, 0x3feed2c980460ad8, 0x3feed503b23e255d,
        0x3feed74a8af46052, 0x3feed99e1330b358, 0x3feedbfe53c12e59, 0x3feede6b5579fdbf, 0x3feee0e521356eba,
        0x3feee36bbfd3f37a, 0x3feee5ff3a3c2774, 0x3feee89f995ad3ad, 0x3feeeb4ce622f2ff, 0x3feeee07298db666,
        0x3feef0ce6c9a8952, 0x3feef3a2b84f15fb, 0x3feef68415b749b1, 0x3feef9728de5593a, 0x3feefc6e29f1c52a,
        0x3feeff76f2fb5e47, 0x3fef028cf22749e4, 0x3fef05b030a1064a, 0x3fef08e0b79a6f1f, 0x3fef0c1e904bc1d2,
        0x3fef0f69c3f3a207, 0x3fef12c25bd71e09, 0x3fef16286141b33d, 0x3fef199bdd85529c, 0x3fef1d1cd9fa652c,
        0x3fef20ab5fffd07a, 0x3fef244778fafb22, 0x3fef27f12e57d14b, 0x3fef2ba88988c933, 0x3fef2f6d9406e7b5,
        0x3fef33405751c4db, 0x3fef3720dcef9069, 0x3fef3b0f2e6d1675, 0x3fef3f0b555dc3fa, 0x3fef43155b5bab74,
        0x3fef472d4a07897c, 0x3fef4b532b08c968, 0x3fef4f87080d89f2, 0x3fef53c8eacaa1d6, 0x3fef5818dcfba487,
        0x3fef5c76e862e6d3, 0x3fef60e316c98398, 0x3fef655d71ff6075, 0x3fef69e603db3285, 0x3fef6e7cd63a8315,
        0x3fef7321f301b460, 0x3fef77d5641c0658, 0x3fef7c97337b9b5f, 0x3fef81676b197d17, 0x3fef864614f5a129,
        0x3fef8b333b16ee12, 0x3fef902ee78b3ff6, 0x3fef953924676d76, 0x3fef9a51fbc74c83, 0x3fef9f7977cdb740,
        0x3fefa4afa2a490da, 0x3fefa9f4867cca6e, 0x3fefaf482d8e67f1, 0x3fefb4aaa2188510, 0x3fefba1bee615a27,
        0x3fefbf9c1cb6412a, 0x3fefc52b376bba97, 0x3fefcac948dd7274, 0x3fefd0765b6e4540, 0x3fefd632798844f8,
        0x3fefdbfdad9cbe14, 0x3fefe1d802243c89, 0x3fefe7c1819e90d8, 0x3fefedba3692d514, 0x3feff3c22b8f71f1,
        0x3feff9d96b2a23d9,
    },
};

static const PowLogData _pow_log_data = {
    .poly = { -0x1p-1, -0x1.555555555556p-1, 0x1.0000000000006p-1, 0x1.999999959554ep-1,
              -0x1.555555529a47ap-1, -0x1.2495b9b4845e9p0, 0x1.0002b8b263fc3p0, },
    .ln2_hi = 0x1.62e42fefa3800p-1,
    .ln2_lo = 0x1.ef35793c76730p-45,
    .invc = {
        0x1.6a00000000000p+0, 0x1.6800000000000p+0, 0x1.6600000000000p+0, 0x1.6400000000000p+0,
        0x1.6200000000000p+0, 0x1.6000000000000p+0, 0x1.5e00000000000p+0, 0x1.5c00000000000p+0,
        0x1.5a00000000000p+0, 0x1.5800000000000p+0, 0x1.5600000000000p+0, 0x1.5600000000000p+0,
        0x1.5400000000000p+0, 0x1.5200000000000p+0, 0x1.5000000000000p+0, 0x1.4e00000000000p+0,
        0x1.4c00000000000p+0, 0x1.4a00000000000p+0, 0x1.4a00000000000p+0, 0x1.4800000000000p+0,
        0x1.4600000000000p+0, 0x1.4400000000000p+0, 0x1.4200000000000p+0, 0x1.4000000000000p+0,
        0x1.4000000000000p+0, 0x1.3e00000000000p+0, 0x1.3c00000000000p+0, 0x1.3a00000000000p+0,
        0x1.3a00000000000p+0, 0x1.3800000000000p+0, 0x1.3600000000000p+0, 0x1.3400000000000p+0,
        0x1.3400000000000p+0, 0x1.3200000000000p+0, 0x1.3000000000000p+0, 0x1.3000000000000p+0,
        0x1.2e00000000000p+0, 0x1.2c00000000000p+0, 0x1.2c00000000000p+0, 0x1.2a00000000000p+0,
        0x1.2800000000000p+0, 0x1.2600000000000p+0, 0x1.2600000000000p+0, 0x1.2400000000000p+0,
        0x1.2400000000000p+0, 0x1.2200000000000p+0, 0x1.2000000000000p+0, 0x1.2000000000000p+0,
        0x1.1e00000000000p+0, 0x1.1c00000000000p+0, 0x1.1c00000000000p+0, 0x1.1a00000000000p+0,
        0x1.1a00000000000p+0, 0x1.1800000000000p+0, 0x1.1600000000000p+0, 0x1.1600000000000p+0,
        0x1.1400000000000p+0, 0x1.1400000000000p+0, 0x1.1200000000000p+0, 0x1.1000000000000p+0,
        0x1.1000000000000p+0, 0x1.0e00000000000p+0, 0x1.0e00000000000p+0, 0x1.0c00000000000p+0,
        0x1.0c00000000000p+0, 0x1.0a00000000000p+0, 0x1.0a00000000000p+0, 0x1.0800000000000p+0,
        0x1.0800000000000p+0, 0x1.0600000000000p+0, 0x1.0400000000000p+0, 0x1.0400000000000p+0,
        0x1.0200000000000p+0, 0x1.0200000000000p+0, 0x1.0000000000000p+0, 0x1.0000000000000p+0,
        0x1.fc00000000000p-1, 0x1.f800000000000p-1, 0x1.f400000000000p-1, 0x1.f000000000000p-1,
        0x1.ec00000000000p-1, 0x1.e800000000000p-1, 0x1.e400000000000p-1, 0x1.e200000000000p-1,
        0x1.de00000000000p-1, 0x1.da00000000000p-1, 0x1.d600000000000p-1, 0x1.d400000000000p-1,
        0x1.d000000000000p-1, 0x1.cc00000000000p-1, 0x1.ca00000000000p-1, 0x1.c600000000000p-1,
        0x1.c400000000000p-1, 0x1.c000000000000p-1, 0x1.be00000000000p-1, 0x1.ba00000000000p-1,
        0x1.b800000000000p-1, 0x1.b400000000000p-1, 0x1.b200000000000p-1, 0x1.ae00000000000p-1,
        0x1.ac00000000000p-1, 0x1.aa00000000000p-1, 0x1.a600000000000p-1, 0x1.a400000000000p-1,
        0x1.a000000000000p-1, 0x1.9e00000000000p-1, 0x1.9c00000000000p-1, 0x1.9a00000000000p-1,
        0x1.9600000000000p-1, 0x1.9400000000000p-1, 0x1.9200000000000p-1, 0x1.9000000000000p-1,
        0x1.8c00000000000p-1, 0x1.8a00000000000p-1, 0x1.8800000000000p-1, 0x1.8600000000000p-1,
        0x1.8400000000000p-1, 0x1.8200000000000p-1, 0x1.7e00000000000p-1, 0x1.7c00000000000p-1,
        0x1.7a00000000000p-1, 0x1.7800000000000p-1, 0x1.7600000000000p-1, 0x1.7400000000000p-1,
        0x1.7200000000000p-1, 0x1.7000000000000p-1, 0x1.6e00000000000p-1, 0x1.6c00000000000p-1,
    },
    .logc = {
        -0x1.62c82f2b9c800p-2, -0x1.5d1bdbf580800p-2, -0x1.5767717455800p-2, -0x1.51aad872df800p-2,
        -0x1.4be5f95777800p-2, -0x1.4618bc21c6000p-2, -0x1.404308686a800p-2, -0x1.3a64c55694800p-2,
        -0x1.347dd9a988000p-2, -0x1.2e8e2bae12000p-2, -0x1.2895a13de8800p-2, -0x1.2895a13de8800p-2,
        -0x1.22941fbcf7800p-2, -0x1.1c898c1699800p-2, -0x1.1675cababa800p-2, -0x1.1058bf9ae4800p-2,
        -0x1.0a324e2739000p-2, -0x1.0402594b4d000p-2, -0x1.0402594b4d000p-2, -0x1.fb9186d5e4000p-3,
        -0x1.ef0adcbdc6000p-3, -0x1.e27076e2af000p-3, -0x1.d5c216b4fc000p-3, -0x1.c8ff7c79aa000p-3,
        -0x1.c8ff7c79aa000p-3, -0x1.bc286742d9000p-3, -0x1.af3c94e80c000p-3, -0x1.a23bc1fe2b000p-3,
        -0x1.a23bc1fe2b000p-3, -0x1.9525a9cf45000p-3, -0x1.87fa06520d000p-3, -0x1.7ab890210e000p-3,
        -0x1.7ab890210e000p-3, -0x1.6d60fe719d000p-3, -0x1.5ff3070a79000p-3, -0x1.5ff3070a79000p-3,
        -0x1.526e5e3a1b000p-3, -0x1.44d2b6ccb8000p-3, -0x1.44d2b6ccb8000p-3, -0x1.371fc201e9000p-3,
        -0x1.29552f81ff000p-3, -0x1.1b72ad52f6000p-3, -0x1.1b72ad52f6000p-3, -0x1.0d77e7cd09000p-3,
        -0x1.0d77e7cd09000p-3, -0x1.fec9131dbe000p-4, -0x1.e27076e2b0000p-4, -0x1.e27076e2b0000p-4,
        -0x1.c5e548f5bc000p-4, -0x1.a926d3a4ae000p-4, -0x1.a926d3a4ae000p-4, -0x1.8c345d631a000p-4,
        -0x1.8c345d631a000p-4, -0x1.6f0d28ae56000p-4, -0x1.51b073f062000p-4, -0x1.51b073f062000p-4,
        -0x1.341d7961be000p-4, -0x1.341d7961be000p-4, -0x1.16536eea38000p-4, -0x1.f0a30c0118000p-5,
        -0x1.f0a30c0118000p-5, -0x1.b42dd71198000p-5, -0x1.b42dd71198000p-5, -0x1.77458f632c000p-5,
        -0x1.77458f632c000p-5, -0x1.39e87b9fec000p-5, -0x1.39e87b9fec000p-5, -0x1.f829b0e780000p-6,
        -0x1.f829b0e780000p-6, -0x1.7b91b07d58000p-6, -0x1.fc0a8b0fc0000p-7, -0x1.fc0a8b0fc0000p-7,
        -0x1.fe02a6b100000p-8, -0x1.fe02a6b100000p-8, 0x0.0000000000000p+0,  0x0.0000000000000p+0,
        0x1.0101575890000p-7,  0x1.0205658938000p-6,  0x1.8492528c90000p-6,  0x1.0415d89e74000p-5,
        0x1.466aed42e0000p-5,  0x1.894aa149fc000p-5,  0x1.ccb73cdddc000p-5,  0x1.eea31c006c000p-5,
        0x1.1973bd1466000p-4,  0x1.3bdf5a7d1e000p-4,  0x1.5e95a4d97a000p-4,  0x1.700d30aeac000p-4,
        0x1.9335e5d594000p-4,  0x1.b6ac88dad6000p-4,  0x1.c885801bc4000p-4,  0x1.ec739830a2000p-4,
        0x1.fe89139dbe000p-4,  0x1.1178e8227e000p-3,  0x1.1aa2b7e23f000p-3,  0x1.2d1610c868000p-3,
        0x1.365fcb0159000p-3,  0x1.4913d8333b000p-3,  0x1.527e5e4a1b000p-3,  0x1.6574ebe8c1000p-3,
        0x1.6f0128b757000p-3,  0x1.7898d85445000p-3,  0x1.8beafeb390000p-3,  0x1.95a5adcf70000p-3,
        0x1.a93ed3c8ae000p-3,  0x1.b31d8575bd000p-3,  0x1.bd087383be000p-3,  0x1.c6ffbc6f01000p-3,
        0x1.db13db0d49000p-3,  0x1.e530effe71000p-3,  0x1.ef5ade4dd0000p-3,  0x1.f991c6cb3b000p-3,
        0x1.07138604d5800p-2,  0x1.0c42d67616000p-2,  0x1.1178e8227e800p-2,  0x1.16b5ccbacf800p-2,
        0x1.1bf99635a6800p-2,  0x1.214456d0eb800p-2,  0x1.2bef07cdc9000p-2,  0x1.314f1e1d36000p-2,
        0x1.36b6776be1000p-2,  0x1.3c25277333000p-2,  0x1.419b423d5e800p-2,  0x1.4718dc271c800p-2,
        0x1.4c9e09e173000p-2,  0x1.522ae0738a000p-2,  0x1.57bf753c8d000p-2,  0x1.5d5bddf596000p-2,
    },
    .logctail = {
        0x1.ab42428375680p-48,  -0x1.ca508d8e0f720p-46, -0x1.362a4d5b6506dp-45, -0x1.684e49eb067d5p-49,
        -0x1.41b6993293ee0p-47, 0x1.3d82f484c84ccp-46,  0x1.c42f3ed820b3ap-50,  0x1.0b1c686519460p-45,
        0x1.5594dd4c58092p-45,  0x1.67b1e99b72bd8p-45,  0x1.5ca14b6cfb03fp-46,  0x1.5ca14b6cfb03fp-46,
        -0x1.65a242853da76p-46, -0x1.fafbc68e75404p-46, 0x1.f1fc63382a8f0p-46,  -0x1.6a8c4fd055a66p-45,
        -0x1.c6bee7ef4030ep-47, -0x1.036b89ef42d7fp-48, -0x1.036b89ef42d7fp-48, 0x1.d572aab993c87p-47,
        0x1.b26b79c86af24p-45,  -0x1.72f4f543fff10p-46, 0x1.1ba91bbca681bp-45,  0x1.7794f689f8434p-45,
        0x1.7794f689f8434p-45,  0x1.94eb0318bb78fp-46,  0x1.a4e633fcd9066p-52,  -0x1.58c64dc46c1eap-45,
        -0x1.58c64dc46c1eap-45, -0x1.ad1d904c1d4e3p-45, 0x1.bbdbf7fdbfa09p-45,  0x1.bdb9072534a58p-45,
        0x1.bdb9072534a58p-45,  -0x1.0e46aa3b2e266p-46, -0x1.e9e439f105039p-46, -0x1.e9e439f105039p-46,
        -0x1.0de8b90075b8fp-45, 0x1.70cc16135783cp-46,  0x1.70cc16135783cp-46,  0x1.178864d27543ap-48,
        -0x1.48d301771c408p-45, -0x1.e80a41811a396p-45, -0x1.e80a41811a396p-45, 0x1.a699688e85bf4p-47,
        0x1.a699688e85bf4p-47,  -0x1.575545ca333f2p-45, 0x1.a342c2af0003cp-45,  0x1.a342c2af0003cp-45,
        -0x1.d0c57585fbe06p-46, 0x1.53935e85baac8p-45,  0x1.53935e85baac8p-45,  0x1.37c294d2f5668p-46,
        0x1.37c294d2f5668p-46,  -0x1.69737c93373dap-45, 0x1.f025b61c65e57p-46,  0x1.f025b61c65e57p-46,
        0x1.c5edaccf913dfp-45,  0x1.c5edaccf913dfp-45,  0x1.47c5e768fa309p-46,  0x1.d599e83368e91p-45,
        0x1.d599e83368e91p-45,  0x1.c827ae5d6704cp-46,  0x1.c827ae5d6704cp-46,  -0x1.cfc4634f2a1eep-45,
        -0x1.cfc4634f2a1eep-45, 0x1.502b7f526feaap-48,  0x1.502b7f526feaap-48,  -0x1.980267c7e09e4p-45,
        -0x1.980267c7e09e4p-45, -0x1.88d5493faa639p-45, -0x1.f1e7cf6d3a69cp-50, -0x1.f1e7cf6d3a69cp-50,
        -0x1.9e23f0dda40e4p-46, -0x1.9e23f0dda40e4p-46, 0x0.0000000000000p+0,   0x0.0000000000000p+0,
        -0x1.0c76b999d2be8p-46, -0x1.3dc5b06e2f7d2p-45, -0x1.aa0ba325a0c34p-45, 0x1.111c05cf1d753p-47,
        -0x1.c167375bdfd28p-45, -0x1.97995d05a267dp-46, -0x1.a68f247d82807p-46, -0x1.e113e4fc93b7bp-47,
        -0x1.5325d560d9e9bp-45, 0x1.cc85ea5db4ed7p-45,  -0x1.c69063c5d1d1ep-45, 0x1.c1e8da99ded32p-49,
        0x1.3115c3abd47dap-45,  -0x1.390802bf768e5p-46, 0x1.646d1c65aacd3p-45,  -0x1.dc068afe645e0p-45,
        -0x1.534d64fa10afdp-45, 0x1.1ef78ce2d07f2p-45,  0x1.ca78e44389934p-45,  0x1.39d6ccb81b4a1p-47,
        0x1.62fa8234b7289p-51,  0x1.5837954fdb678p-45,  0x1.633e8e5697dc7p-45,  0x1.9cf8b2c3c2e78p-46,
        -0x1.5118de59c21e1p-45, -0x1.c661070914305p-46, -0x1.73d54aae92cd1p-47, 0x1.7f22858a0ff6fp-47,
        -0x1.8724350562169p-45, -0x1.c358d4eace1aap-47, -0x1.d4bc4595412b6p-45, -0x1.1ec72c5962bd2p-48,
        -0x1.aff2af715b035p-45, 0x1.212276041f430p-51,  -0x1.a211565bb8e11p-51, 0x1.bcbecca0cdf30p-46,
        0x1.89cdb16ed4e91p-48,  0x1.7188b163ceae9p-45,  -0x1.c210e63a5f01cp-45, 0x1.b9acdf7a51681p-45,
        0x1.ca6ed5147bdb7p-45,  0x1.a87deba46baeap-47,  0x1.a9cfa4a5004f4p-45,  -0x1.8e27ad3213cb8p-45,
        0x1.16ecdb0f177c8p-46,  0x1.83b54b606bd5cp-46,  0x1.8e436ec90e09dp-47,  -0x1.f27ce0967d675p-45,
        -0x1.e20891b0ad8a4p-45, 0x1.ebe708164c759p-45,  0x1.fadedee5d40efp-46,  -0x1.a0b2a08a465dcp-47,
    },
};

// ============================================================
// Helper functions matching SVE algorithm
// ============================================================

// Check if x is an integer.
template <class D>
HWY_INLINE auto IsInt(D d, hn::Vec<D> x)
{
    return hn::Eq(hn::Trunc(x), x);
}

// Check if x is real not integer valued.
template <class D>
HWY_INLINE auto IsNotInt(D d, hn::Vec<D> x)
{
    return hn::Ne(hn::Trunc(x), x);
}

// Check if x is an odd integer.
template <class D>
HWY_INLINE auto IsOdd(D d, hn::Vec<D> x)
{
    hn::Vec<D> y = hn::Mul(x, hn::Set(d, 0.5));
    return IsNotInt(d, y);
}

// Top 12 bits (sign and exponent of each double lane).
template <class D>
HWY_INLINE hn::Vec<hn::RebindToUnsigned<D>> Top12(D d, hn::Vec<D> x)
{
    using DU = hn::RebindToUnsigned<D>;
    return hn::ShiftRight<52>(hn::BitCast(DU(), x));
}

// Returns true if input is the bit representation of 0, infinity or nan.
template <class D>
HWY_INLINE auto IsZeroInfNan(D d, hn::Vec<hn::RebindToUnsigned<D>> i)
{
    using VU = hn::Vec<hn::RebindToUnsigned<D>>;
    using DU = hn::RebindToUnsigned<D>;
    const DU du;
    VU one = hn::Set(du, 1);
    VU bound = hn::Set(du, 2u * 0x7ff0000000000000ULL - 1);
    return hn::Ge(hn::Sub(hn::Add(i, i), one), bound);
}

// ============================================================
// Double-precision log - matching SVE sv_log_inline
// ============================================================

template <class D>
HWY_INLINE hn::Vec<D> v_log_inline_hwy(D d, hn::Vec<hn::RebindToUnsigned<D>> ix, hn::Vec<D> *tail)
{
    using DU = hn::RebindToUnsigned<D>;
    using DS = hn::RebindToSigned<D>;
    using VD = hn::Vec<D>;
    using VU = hn::Vec<DU>;
    using VS = hn::Vec<DS>;

    const DU du;
    const DS ds;

    // x = 2^k z; where z is in range [Off,2*Off) and exact.
    // The range is split into N subintervals.
    // The ith subinterval contains z and c is near its center.
    VU offset = hn::Set(du, V_POW_OFF);
    VU tmp = hn::Sub(ix, offset);
    VU idx_mask = hn::Set(du, (uint64_t)(V_POW_N_LOG - 1));
    VU idx_u = hn::And(hn::ShiftRight<52 - V_POW_LOG_TABLE_BITS>(tmp), idx_mask);
    VS k = hn::ShiftRight<52>(hn::BitCast(ds, tmp));
    VD kd = hn::ConvertTo(d, k);
    VU iz = hn::Sub(ix, hn::ShiftLeft<52>(hn::BitCast(du, k)));
    VD z = hn::BitCast(d, iz);

    // log(x) = k*Ln2 + log(c) + log1p(z/c-1).
    // SVE lookup requires 3 separate lookup tables.
    VD invc = hn::GatherIndex(d, _pow_log_data.invc, hn::BitCast(ds, idx_u));
    VD logc = hn::GatherIndex(d, _pow_log_data.logc, hn::BitCast(ds, idx_u));
    VD logctail = hn::GatherIndex(d, _pow_log_data.logctail, hn::BitCast(ds, idx_u));

    // Note: 1/c is j/N or j/N/2 where j is an integer in [N,2N) and
    // |z/c - 1| < 1/N, so r = z/c - 1 is exactly representible.
    VD r = hn::MulAdd(z, invc, hn::Set(d, -1.0));

    const npy_double ln2_hi = _pow_log_data.ln2_hi;
    const npy_double ln2_lo = _pow_log_data.ln2_lo;
    VD t1 = hn::MulAdd(kd, hn::Set(d, ln2_hi), logc);
    VD t2 = hn::Add(t1, r);
    VD lo1 = hn::MulAdd(kd, hn::Set(d, ln2_lo), logctail);
    VD lo2 = hn::Add(hn::Sub(t1, t2), r);

    // Evaluation is optimized assuming superscalar pipelined execution.
    VD ar = hn::Mul(hn::Set(d, -0.5), r);
    VD ar2 = hn::Mul(r, ar);
    VD ar3 = hn::Mul(r, ar2);
    VD hi = hn::Add(t2, ar2);
    VD lo3 = hn::NegMulAdd(ar, r, ar2);
    VD lo4 = hn::Add(hn::Sub(t2, hi), ar2);

    const npy_double A1 = _pow_log_data.poly[1];
    const npy_double A2 = _pow_log_data.poly[2];
    const npy_double A3 = _pow_log_data.poly[3];
    const npy_double A4 = _pow_log_data.poly[4];
    const npy_double A5 = _pow_log_data.poly[5];
    const npy_double A6 = _pow_log_data.poly[6];

    VD a56 = hn::MulAdd(r, hn::Set(d, A6), hn::Set(d, A5));
    VD a34 = hn::MulAdd(r, hn::Set(d, A4), hn::Set(d, A3));
    VD a12 = hn::MulAdd(r, hn::Set(d, A2), hn::Set(d, A1));
    VD p = hn::MulAdd(ar2, a56, a34);
    p = hn::MulAdd(ar2, p, a12);
    p = hn::Mul(ar3, p);
    VD lo = hn::Add(hn::Add(hn::Add(lo1, lo2), hn::Add(lo3, lo4)), p);
    VD y = hn::Add(hi, lo);
    *tail = hn::Add(hn::Sub(hi, y), lo);
    return y;
}

// ============================================================
// Double-precision exp - matching SVE sv_exp_inline / sv_exp_core
// ============================================================

template <class D>
HWY_INLINE hn::Vec<D> ExpSpecialCase(D d, hn::Vec<D> tmp, hn::Vec<hn::RebindToUnsigned<D>> sbits,
                                     hn::Vec<hn::RebindToSigned<D>> ki)
{
    using DU = hn::RebindToUnsigned<D>;
    using DS = hn::RebindToSigned<D>;
    using VD = hn::Vec<D>;
    using VU = hn::Vec<DU>;
    using VS = hn::Vec<DS>;

    const DS ds;
    auto p_pos = hn::Ge(hn::BitCast(ds, ki), hn::Zero(ds));

    // Scale up or down depending on sign of k.
    VS offset = hn::IfThenElse(p_pos, hn::Set(ds, 1009LL << 52), hn::Set(ds, -1022LL << 52));
    VD factor = hn::IfThenElse(hn::RebindMask(d, p_pos), hn::Set(d, 0x1p1009), hn::Set(d, 0x1p-1022));

    VU offset_sbits = hn::Sub(hn::BitCast(DU(), sbits), hn::BitCast(DU(), offset));
    VD scale = hn::BitCast(d, offset_sbits);
    VD res = hn::MulAdd(scale, tmp, scale);
    return hn::Mul(res, factor);
}

template <class D>
HWY_INLINE hn::Vec<D> ExpCore(D d, hn::Vec<D> x, hn::Vec<D> xtail,
                              hn::Vec<hn::RebindToUnsigned<D>> sign_bias,
                              hn::Vec<D> *out_tmp, hn::Vec<hn::RebindToUnsigned<D>> *out_sbits,
                              hn::Vec<hn::RebindToSigned<D>> *out_ki)
{
    using DU = hn::RebindToUnsigned<D>;
    using DS = hn::RebindToSigned<D>;
    using VD = hn::Vec<D>;
    using VU = hn::Vec<DU>;
    using VS = hn::Vec<DS>;

    const DU du;
    const DS ds;

    // exp(x) = 2^(k/N) * exp(r)
    VD n_over_ln2 = hn::Set(d, _pow_exp_data.n_over_ln2);
    VD z = hn::Mul(x, n_over_ln2);
    VD kd = hn::Round(z);
    VS ki = hn::ConvertTo(ds, kd);
    VU ki_u = hn::BitCast(du, ki);

    VD ln2_hi_n = hn::Set(d, _pow_exp_data.ln2_over_n_hi);
    VD ln2_lo_n = hn::Set(d, _pow_exp_data.ln2_over_n_lo);
    VD r = hn::NegMulAdd(kd, ln2_hi_n, x);
    r = hn::NegMulAdd(kd, ln2_lo_n, r);
    r = hn::Add(r, xtail);

    VU idx = hn::And(ki_u, hn::Set(du, (uint64_t)(V_POW_N_EXP - 1)));
    VU top = hn::ShiftLeft<52 - V_POW_EXP_TABLE_BITS>(hn::BitCast(du, hn::Add(ki, hn::BitCast(ds, sign_bias))));
    VU sbits = hn::GatherIndex(du, _pow_exp_data.sbits, hn::BitCast(ds, idx));
    sbits = hn::Add(sbits, top);

    VD r2 = hn::Mul(r, r);
    VD exp_c0 = hn::Set(d, _pow_exp_data.poly[0]);
    VD exp_c1 = hn::Set(d, _pow_exp_data.poly[1]);
    VD exp_c2 = hn::Set(d, _pow_exp_data.poly[2]);
    VD tmp = hn::MulAdd(r, exp_c2, exp_c1);
    tmp = hn::MulAdd(r, tmp, exp_c0);
    tmp = hn::MulAdd(r2, tmp, r);
    VD scale = hn::BitCast(d, sbits);
    z = hn::MulAdd(scale, tmp, scale);

    *out_tmp = tmp;
    *out_sbits = sbits;
    *out_ki = ki;
    return z;
}

template <class D>
HWY_INLINE hn::Vec<D> v_exp_inline_hwy(D d, hn::Vec<D> x, hn::Vec<D> xtail,
                                       hn::Vec<hn::RebindToUnsigned<D>> sign_bias)
{
    using DU = hn::RebindToUnsigned<D>;
    using VD = hn::Vec<D>;
    using VU = hn::Vec<DU>;

    const DU du;

    // 3 types of special cases: tiny, huge, and scale*(1+TMP) overflow.
    VU abstop = hn::And(Top12(d, x), hn::Set(du, 0x7ffULL));
    auto uoflow = hn::Ge(hn::Sub(abstop, hn::Set(du, V_POW_SMALL_EXP)), hn::Set(du, V_POW_THRES_EXP));

    VD tmp;
    VU sbits;
    hn::Vec<hn::RebindToSigned<D>> ki;
    if (HWY_UNLIKELY(!hn::AllFalse(du, uoflow))) {
        VD z = ExpCore(d, x, xtail, sign_bias, &tmp, &sbits, &ki);

        // |x| is tiny (|x| <= 0x1p-54).
        auto uflow = hn::Ge(hn::Sub(abstop, hn::Set(du, V_POW_SMALL_EXP)), hn::Set(du, 0x80000000ULL));
        uflow = hn::And(uoflow, uflow);
        if ((!hn::AllFalse(du, uflow))) {
            npy_set_floatstatus_underflow();
        }
        // |x| is huge (|x| >= 1024).
        auto oflow = hn::Ge(abstop, hn::Set(du, V_POW_HUGE_EXP));
        oflow = hn::And(uoflow, hn::AndNot(uflow, oflow));
        if ((!hn::AllFalse(du, oflow))) {
            npy_set_floatstatus_overflow();
        }
        // Handle scale*(1+TMP) overflow for intermediate values
        auto special = hn::AndNot(hn::Or(uflow, oflow), uoflow);
        if (HWY_UNLIKELY(!hn::AllFalse(du, special))) {
            z = hn::IfThenElse(hn::RebindMask(d, special), ExpSpecialCase(d, tmp, sbits, ki), z);
        }
        
        // For huge values, return inf directly without calling ExpCore
        // (ExpCore would overflow the exponent shift into the sign bit)
        auto x_is_neg = hn::Lt(x, hn::Zero(d));
        VU sign_mask = hn::ShiftLeft<52 - V_POW_EXP_TABLE_BITS>(sign_bias);
        VD res_oflow = hn::IfThenElse(x_is_neg, hn::Zero(d), hn::Set(d, NPY_INFINITY));
        res_oflow = hn::BitCast(d, hn::Or(hn::BitCast(du, res_oflow), sign_mask));
        // Avoid spurious underflow for tiny x.
        VD res_spurious_uflow = hn::BitCast(d, hn::Or(sign_mask, hn::Set(du, 0x3ff0000000000000ULL)));

        z = hn::IfThenElse(hn::RebindMask(d, oflow), res_oflow, z);
        z = hn::IfThenElse(hn::RebindMask(d, uflow), res_spurious_uflow, z);
        return z;
    }

    return ExpCore(d, x, xtail, sign_bias, &tmp, &sbits, &ki);
}

// ============================================================
// Main Double-precision pow - matching SVE SV_NAME_D2(pow)
// ============================================================

template <class D>
HWY_INLINE hn::Vec<D> pow_hwy_impl(D d, hn::Vec<D> x, hn::Vec<D> y)
{
    using DU = hn::RebindToUnsigned<D>;
    using VD = hn::Vec<D>;
    using VU = hn::Vec<DU>;

    const DU du;
    HWY_LANES_CONSTEXPR size_t N = hn::Lanes(d);

    VU vix0 = hn::BitCast(du, x);
    VU viy0 = hn::BitCast(du, y);
    // Special cases of x or y: zero, inf and nan.
    auto xspecial = IsZeroInfNan(du, vix0);
    auto yspecial = IsZeroInfNan(du, viy0);
    auto special = hn::Or(xspecial, yspecial);

    VD x_comp = hn::IfThenElse(hn::RebindMask(d, special), hn::Set(d, 1.0), x);
    VD y_comp = hn::IfThenElse(hn::RebindMask(d, special), hn::Set(d, 0.0), y);
    VU vix = hn::BitCast(du, x_comp);

    // Negative x cases.
    auto xisneg = hn::Lt(x_comp, hn::Zero(d));

    // Set sign_bias and ix depending on sign of x and nature of y.
    auto yint_or_xpos = hn::FirstN(d, N);
    VU sign_bias = hn::Zero(du);
    if (HWY_UNLIKELY(!hn::AllFalse(d, xisneg))) {
        yint_or_xpos = hn::Or(hn::Not(xisneg), IsInt(d, y_comp));
        auto yisodd_xisneg = hn::And(IsOdd(d, y_comp), xisneg);
        // Always use abs(x) for core computation to avoid log(negative) = NaN
        vix = hn::And(vix0, hn::Set(du, 0x7fffffffffffffffULL));
        sign_bias = hn::IfThenElse(hn::RebindMask(du, yisodd_xisneg), hn::Set(du, V_POW_SIGN_BIAS), hn::Zero(du));
    }

    // Cases of subnormal x: |x| < 0x1p-1022.
    auto x_is_subnormal = hn::Lt(hn::Abs(x_comp), hn::Set(d, 0x1p-1022));
    if (HWY_UNLIKELY(!hn::AllFalse(d, hn::RebindMask(d, hn::And(yint_or_xpos, x_is_subnormal))))) {
        VD x_scaled = hn::Mul(hn::Abs(x_comp), hn::Set(d, 0x1p52));
        VU vix_norm = hn::Sub(hn::BitCast(du, x_scaled), hn::Set(du, 52ULL << 52));
        vix = hn::IfThenElse(hn::RebindMask(du, x_is_subnormal), vix_norm, vix);
    }

    // y_hi = log(ix, &y_lo).
    VD vlo;
    VD vhi = v_log_inline_hwy(d, vix, &vlo);

    // z = exp(y_hi, y_lo, sign_bias).
    VD vehi = hn::Mul(y_comp, vhi);
    VD vemi = hn::NegMulAdd(y_comp, vhi, vehi);
    VD velo = hn::Neg(hn::NegMulAdd(y_comp, vlo, vemi));
    VD vz = v_exp_inline_hwy(d, vehi, velo, sign_bias);

    // Cases of finite y and finite negative x.
    vz = hn::IfThenElse(yint_or_xpos, vz, hn::Set(d, NPY_NAN));

    if (HWY_UNLIKELY(!hn::AllFalse(du, special))) {
        HWY_ALIGN npy_double xbuf[HWY_MAX_LANES_D(D)];
        HWY_ALIGN npy_double ybuf[HWY_MAX_LANES_D(D)];
        HWY_ALIGN npy_double rbuf[HWY_MAX_LANES_D(D)];
        HWY_ALIGN uint64_t mask[HWY_MAX_LANES_D(D)];
        hn::Store(hn::VecFromMask(du, special), du, mask);
        hn::Store(x, d, xbuf);
        hn::Store(y, d, ybuf);
        hn::Store(vz, d, rbuf);
        for (size_t i = 0; i < N; i++) {
            if (mask[i]) {
                rbuf[i] = npy_pow(xbuf[i], ybuf[i]);
            }
        }
        vz = hn::Load(d, rbuf);
    }
    return vz;
}

// ============================================================
// Main Single-precision pow - following SVE algorithm
// ============================================================

#define V_POWF_SIGN_BIAS (1 << (5 + 11))
#define V_POWF_SMALL_BOUND 0x1p-126f
#define V_POWF_INF_BITS 0x7f800000

template <class D>
HWY_INLINE auto IsZeroInfNanF(D d, hn::Vec<hn::RebindToUnsigned<D>> i)
{
    using VU = hn::Vec<hn::RebindToUnsigned<D>>;
    using DU = hn::RebindToUnsigned<D>;
    const DU du;
    VU one = hn::Set(du, 1);
    VU bound = hn::Set(du, 2u * 0x7f800000 - 1);
    return hn::Ge(hn::Sub(hn::Add(i, i), one), bound);
}

// ============================================================
// Single-precision pow constants
// ============================================================

#define V_POWF_SIGN_BIAS (1 << (5 + 11))
#define V_POWF_SMALL_BOUND 0x1p-126f
#define V_POWF_SUBNORMAL_BIAS 0x0b800000
#define V_POWF_SMALL_NORM 0x1p23f
#define V_POWF_OFF 0x3f35d000
#define V_POWF_MANTISSA_MASK 0x007fffff
#define V_POWF_INF_BITS 0x7f800000

// float pow tables (matching optimized-routines sv_powf_inline.h)
#define V_POWF_EXP2_TABLE_BITS 5
#define V_POWF_EXP2_N (1 << V_POWF_EXP2_TABLE_BITS)
#define V_POWF_LOG2_TABLE_BITS 5
#define V_POWF_LOG2_N (1 << V_POWF_LOG2_TABLE_BITS)
#define V_POWF_LOG2_IDX_MASK (V_POWF_LOG2_N - 1)
#define V_POWF_EXP2_IDX_MASK (V_POWF_EXP2_N - 1)
#define V_POWF_UFLOW_BOUND -0x1.2cp+12f
#define V_POWF_OFLOW_BOUND 0x1p+12f

static const npy_double __powf_log2_tab_invc[V_POWF_LOG2_N] = {
    0x1.6489890582816p+0, 0x1.5cf19b35e3472p+0, 0x1.55aac0e956d65p+0, 0x1.4eb0022977e01p+0,
    0x1.47fcccda1dd1fp+0, 0x1.418ceabab68c1p+0, 0x1.3b5c788f1edb3p+0, 0x1.3567de48e9c9ap+0,
    0x1.2fabc80fd19bap+0, 0x1.2a25200ce536bp+0, 0x1.24d108e0152e3p+0, 0x1.1facd8ab2fbe1p+0,
    0x1.1ab614a03efdfp+0, 0x1.15ea6d03af9ffp+0, 0x1.1147b994bb776p+0, 0x1.0ccbf650593aap+0,
    0x1.0875408477302p+0, 0x1.0441d42a93328p+0, 0x1p+0, 0x1.f1d006c855e86p-1,
    0x1.e28c3341aa301p-1, 0x1.d4bdf9aa64747p-1, 0x1.c7b45a24e5803p-1, 0x1.bb5f5eb2ed60ap-1,
    0x1.afb0bff8fe6b4p-1, 0x1.a49badf7ab1f5p-1, 0x1.9a14a111fc4c9p-1, 0x1.901131f5b2fdcp-1,
    0x1.8687f73f6d865p-1, 0x1.7d7067eb77986p-1, 0x1.74c2c1cf97b65p-1, 0x1.6c77f37cff2a1p-1,
};

static const npy_double __powf_log2_tab_logc[V_POWF_LOG2_N] = {
    -0x1.e960f97b22702p+3, -0x1.c993406cd4db6p+3, -0x1.aa711d9a7d0f3p+3, -0x1.8bf37bacdce9bp+3,
    -0x1.6e13b3519946ep+3, -0x1.50cb8281e4089p+3, -0x1.341504a237e2bp+3, -0x1.17eaab624ffbbp+3,
    -0x1.f88e708f8c853p+2, -0x1.c24b6da113914p+2, -0x1.8d02ee397cb1dp+2, -0x1.58ac1223408b3p+2,
    -0x1.253e6fd190e89p+2, -0x1.e5641882c12ffp+1, -0x1.81fea712926f7p+1, -0x1.203e240de64a3p+1,
    -0x1.8029b86a78281p0, -0x1.85d713190fb9p-1, 0x0p+0, 0x1.4c1cc07312997p0,
    0x1.5e1848ccec948p+1, 0x1.04cfcb7f1196fp+2, 0x1.582813d463c21p+2, 0x1.a936fa68760ccp+2,
    0x1.f81bc31d6cc4ep+2, 0x1.2279a09fae6b1p+3, 0x1.47ec0b6df5526p+3, 0x1.6c71762280f1p+3,
    0x1.90155070798dap+3, 0x1.b2e23b1d3068cp+3, 0x1.d4e21b0daa86ap+3, 0x1.f61e2a2f67f3fp+3,
};

static const uint64_t __powf_exp2_tab[V_POWF_EXP2_N] = {
    0x3ff0000000000000, 0x3fefd9b0d3158574, 0x3fefb5586cf9890f, 0x3fef9301d0125b51, 0x3fef72b83c7d517b,
    0x3fef54873168b9aa, 0x3fef387a6e756238, 0x3fef1e9df51fdee1, 0x3fef06fe0a31b715, 0x3feef1a7373aa9cb,
    0x3feedea64c123422, 0x3feece086061892d, 0x3feebfdad5362a27, 0x3feeb42b569d4f82, 0x3feeab07dd485429,
    0x3feea47eb03a5585, 0x3feea09e667f3bcd, 0x3fee9f75e8ec5f74, 0x3feea11473eb0187, 0x3feea589994cce13,
    0x3feeace5422aa0db, 0x3feeb737b0cdc5e5, 0x3feec49182a3f090, 0x3feed503b23e255d, 0x3feee89f995ad3ad,
    0x3feeff76f2fb5e47, 0x3fef199bdd85529c, 0x3fef3720dcef9069, 0x3fef5818dcfba487, 0x3fef7c97337b9b5f,
    0x3fefa4afa2a490da, 0x3fefd0765b6e4540,
};

static const npy_double __powf_log2_poly[4] = {
    -0x1.6ff5daa3b3d7cp+3, 0x1.ec81d03c01aebp+3, -0x1.71547bb43f101p+4, 0x1.7154764a815cbp+5,
};

static const npy_double __powf_exp2_poly[3] = {
    0x1.c6af84b912394p-20, 0x1.ebfce50fac4f3p-13, 0x1.62e42ff0c52d6p-6,
};

// ============================================================
// float pow core - widens to double, computes, narrows back
// Matching SVE sv_powf_core_ext and sv_powf_core from optimized-routines
// ============================================================

// Compute core for half of the lanes in double precision.
template <class D64>
HWY_INLINE hn::Vec<D64> PowfCoreExt(
    D64 d64, hn::Vec<hn::RebindToUnsigned<D64>> i, hn::Vec<D64> z, hn::Vec<hn::RebindToSigned<D64>> k,
    hn::Vec<D64> y, hn::Vec<hn::RebindToUnsigned<D64>> sign_bias, hn::Vec<D64> *pylogx)
{
    using DU64 = hn::RebindToUnsigned<D64>;
    using DS64 = hn::RebindToSigned<D64>;
    using VD64 = hn::Vec<D64>;
    using VU64 = hn::Vec<DU64>;

    const DU64 du64;
    const DS64 ds64;

    VD64 invc = hn::GatherIndex(d64, __powf_log2_tab_invc, hn::BitCast(ds64, i));
    VD64 logc = hn::GatherIndex(d64, __powf_log2_tab_logc, hn::BitCast(ds64, i));

    // log2(x) = log1p(z/c-1)/ln2 + log2(c) + k.
    VD64 r = hn::MulAdd(z, invc, hn::Set(d64, -1.0));
    VD64 y0 = hn::Add(logc, hn::ConvertTo(d64, k));

    // Polynomial to approximate log1p(r)/ln2.
    VD64 logx = hn::Set(d64, __powf_log2_poly[0]);
    logx = hn::MulAdd(r, logx, hn::Set(d64, __powf_log2_poly[1]));
    logx = hn::MulAdd(r, logx, hn::Set(d64, __powf_log2_poly[2]));
    logx = hn::MulAdd(r, logx, hn::Set(d64, __powf_log2_poly[3]));
    logx = hn::MulAdd(r, logx, y0);
    *pylogx = hn::Mul(y, logx);

    // exp2(x) = 2^(k/N) * 2^r
    VD64 kd = hn::Round(*pylogx);
    VU64 ki = hn::BitCast(du64, hn::ConvertTo(ds64, kd));
    r = hn::Sub(*pylogx, kd);

    VU64 ki_idx = hn::And(ki, hn::Set(du64, (uint64_t)(V_POWF_EXP2_N - 1)));
    VU64 t = hn::GatherIndex(du64, __powf_exp2_tab, hn::BitCast(ds64, ki_idx));
    VU64 ski = hn::Add(ki, sign_bias);
    t = hn::Add(t, hn::ShiftLeft<52 - V_POWF_EXP2_TABLE_BITS>(ski));
    VD64 s = hn::BitCast(d64, t);

    VD64 p = hn::Set(d64, __powf_exp2_poly[0]);
    p = hn::MulAdd(p, r, hn::Set(d64, __powf_exp2_poly[1]));
    p = hn::MulAdd(p, r, hn::Set(d64, __powf_exp2_poly[2]));
    p = hn::MulAdd(p, hn::Mul(s, r), s);

    return p;
}

// Widen vector to double precision and compute core on both halves.
// Only available on SVE where Highway supports proper DemoteTo from double to float.
template <class D32>
HWY_INLINE hn::Vec<D32> powf_core(D32 d, hn::Vec<D32> *ylogx_out, hn::Vec<hn::RebindToUnsigned<D32>> tmp,
                                       hn::Vec<D32> iz, hn::Vec<D32> y, hn::Vec<hn::RebindToSigned<D32>> k,
                                       hn::Vec<hn::RebindToUnsigned<D32>> sign_bias)
{
    using DU32 = hn::RebindToUnsigned<D32>;
    using D64 = hn::Repartition<npy_double, D32>;
    using DU64 = hn::RebindToUnsigned<D64>;
    using DS64 = hn::RebindToSigned<D64>;
    auto dh = hn::Half<D32>();
    auto du32 = DU32();
    auto d64 = D64();
    auto du64 = DU64();
    auto ds64 = DS64();

    auto tmp_idx = hn::And(hn::ShiftRight<(23 - V_POWF_LOG2_TABLE_BITS)>(tmp), hn::Set(du32, V_POWF_LOG2_IDX_MASK));
    auto idx_lo = hn::PromoteLowerTo(du64, tmp_idx);
    auto idx_hi = hn::PromoteUpperTo(du64, tmp_idx);

    auto iz_lo = hn::PromoteLowerTo(d64, iz);
    auto iz_hi = hn::PromoteUpperTo(d64, iz);

    auto k_lo = hn::PromoteLowerTo(ds64, k);
    auto k_hi = hn::PromoteUpperTo(ds64, k);

    auto y_lo = hn::PromoteLowerTo(d64, y);
    auto y_hi = hn::PromoteUpperTo(d64, y);

    auto sign_bias_lo = hn::PromoteLowerTo(du64, sign_bias);
    auto sign_bias_hi = hn::PromoteUpperTo(du64, sign_bias);

    hn::Vec<D64> ylogx_lo, ylogx_hi;
    hn::Vec<D64> lo = PowfCoreExt(d64, idx_lo, iz_lo, k_lo, y_lo, sign_bias_lo, &ylogx_lo);
    hn::Vec<D64> hi = PowfCoreExt(d64, idx_hi, iz_hi, k_hi, y_hi, sign_bias_hi, &ylogx_hi);

    auto ylogx_lo_32 = hn::DemoteTo(dh, ylogx_lo);
    auto ylogx_hi_32 = hn::DemoteTo(dh, ylogx_hi);
    *ylogx_out = hn::Combine(d, ylogx_hi_32, ylogx_lo_32);
    auto lo_32 = hn::DemoteTo(dh, lo);
    auto hi_32 = hn::DemoteTo(dh, hi);
    return hn::Combine(d, hi_32, lo_32);
}

// SVE float pow implementation - matches SVE SV_NAME_F2(pow) algorithm
template <class D>
HWY_INLINE hn::Vec<D> powf_hwy_impl(D d, hn::Vec<D> x, hn::Vec<D> y)
{
    using DU = hn::RebindToUnsigned<D>;
    using DS = hn::RebindToSigned<D>;
    using VD = hn::Vec<D>;
    using VU = hn::Vec<DU>;
    using VS = hn::Vec<DS>;

    const DU du;
    const DS ds;
    HWY_LANES_CONSTEXPR size_t N = hn::Lanes(d);

    VU vix0 = hn::BitCast(du, x);
    VU viy0 = hn::BitCast(du, y);
    // Special cases of x or y: zero, inf and nan.
    auto xspecial = IsZeroInfNanF(du, vix0);
    auto yspecial = IsZeroInfNanF(du, viy0);
    auto special = hn::Or(xspecial, yspecial);

    VD x_comp = hn::IfThenElse(hn::RebindMask(d, special), hn::Set(d, 1.0), x);
    VD y_comp = hn::IfThenElse(hn::RebindMask(d, special), hn::Set(d, 0.0), y);
    VU vix = hn::BitCast(du, x_comp);

    // Negative x cases.
    auto xisneg = hn::Lt(x_comp, hn::Zero(d));

    // Set sign_bias and ix depending on sign of x and nature of y.
    auto yint_or_xpos = hn::FirstN(d, N);
    VU sign_bias = hn::Zero(du);
    // VU vix = vix0;
    if (HWY_UNLIKELY(!hn::AllFalse(d, xisneg))) {
        yint_or_xpos = hn::Or(hn::Not(xisneg), IsInt(d, y_comp));
        auto yisodd_xisneg = hn::And(IsOdd(d, y_comp), xisneg);
        // Always use abs(x) for core computation to avoid log(negative) = NaN
        vix = hn::And(vix0, hn::Set(du, 0x7fffffff));
        sign_bias = hn::IfThenElse(hn::RebindMask(du, yisodd_xisneg), hn::Set(du, V_POWF_SIGN_BIAS), hn::Zero(du));
    }

    // Cases of subnormal x: |x| < 0x1p-126.
    auto x_is_subnormal = hn::Lt(hn::Abs(x_comp), hn::Set(d, V_POWF_SMALL_BOUND));
    if (HWY_UNLIKELY(!hn::AllFalse(d, hn::RebindMask(d, hn::And(yint_or_xpos, x_is_subnormal))))) {
        VD x_scaled = hn::Mul(hn::Abs(x_comp), hn::Set(d, V_POWF_SMALL_NORM));
        VU vix_norm = hn::Sub(hn::BitCast(du, x_scaled), hn::Set(du, V_POWF_SUBNORMAL_BIAS));
        vix = hn::IfThenElse(hn::RebindMask(du, x_is_subnormal), vix_norm, vix);
    }

    // Part of core computation carried in working precision.
    VU off = hn::Set(du, V_POWF_OFF);
    VU tmp = hn::Sub(vix, off);
    VU top = hn::AndNot(hn::Set(du, V_POWF_MANTISSA_MASK), tmp);
    VD iz = hn::BitCast(d, hn::Sub(vix, top));
    VS k = hn::ShiftRight<23 - V_POWF_EXP2_TABLE_BITS>(hn::BitCast(ds, top));

    // Compute core in extended precision and return intermediate ylogx results.
    VD ylogx;
    VD ret = powf_core(d, &ylogx, tmp, iz, y_comp, k, sign_bias);

    // Handle exp special cases of underflow and overflow.
    VU sign = hn::ShiftLeft<20 - V_POWF_EXP2_TABLE_BITS>(sign_bias);
    VD ret_oflow = hn::BitCast(d, hn::Or(sign, hn::Set(du, V_POWF_INF_BITS)));
    VD ret_uflow = hn::BitCast(d, sign);
    ret = hn::IfThenElse(hn::Le(ylogx, hn::Set(d, V_POWF_UFLOW_BOUND)), ret_uflow, ret);
    ret = hn::IfThenElse(hn::Gt(ylogx, hn::Set(d, V_POWF_OFLOW_BOUND)), ret_oflow, ret);
    // Cases of finite y and finite negative x.
    ret = hn::IfThenElse(yint_or_xpos, ret, hn::Set(d, NPY_NANF));

    // Scalar fallback for special cases.
    if (HWY_UNLIKELY(!hn::AllFalse(du, special))) {
        HWY_ALIGN npy_float xbuf[HWY_MAX_LANES_D(D)];
        HWY_ALIGN npy_float ybuf[HWY_MAX_LANES_D(D)];
        HWY_ALIGN npy_float rbuf[HWY_MAX_LANES_D(D)];
        HWY_ALIGN uint32_t mask[HWY_MAX_LANES_D(D)];
        hn::Store(hn::VecFromMask(du, special), du, mask);
        hn::Store(x, d, xbuf);
        hn::Store(y, d, ybuf);
        hn::Store(ret, d, rbuf);
        for (size_t i = 0; i < N; i++) {
            if (mask[i]) {
                rbuf[i] = npy_powf(xbuf[i], ybuf[i]);
            }
        }
        ret = hn::Load(d, rbuf);
    }

    return ret;
}

// ============================================================
// Unified Pow entry point
// ============================================================

template <typename D>
HWY_INLINE hn::VFromD<D> Pow(D d, hn::VFromD<D> x, hn::VFromD<D> y)
{
    using T = hn::TFromD<D>;
    static_assert(std::is_same_v<T, npy_double> || std::is_same_v<T, npy_float>, "pow not implemented for this type");
    if constexpr (std::is_same_v<T, npy_double>) {
        return pow_hwy_impl(d, x, y);
    } else {
        return powf_hwy_impl(d, x, y);
    }
}

// ============================================================
// Highway SIMD wrapper functions
// ============================================================

template <typename T>
static HWY_ATTR void simd_power(const T *src1, npy_intp ssrc1,
                                const T *src2, npy_intp ssrc2,
                                      T *dst, npy_intp sdst, npy_intp len)
{
    using D = hn::ScalableTag<T>;
    const D d = D();
    using V = hn::Vec<decltype(d)>;
    using DI = hn::RebindToSigned<decltype(d)>;
    using TI = hn::TFromD<DI>;
    const DI di = DI();
    HWY_LANES_CONSTEXPR size_t N = hn::Lanes(d);

    auto idx_in1 = hn::Mul(hn::Iota(di, 0), hn::Set(di, static_cast<TI>(ssrc1)));
    auto idx_in2 = hn::Mul(hn::Iota(di, 0), hn::Set(di, static_cast<TI>(ssrc2)));
    auto idx_out1 = hn::Mul(hn::Iota(di, 0), hn::Set(di, static_cast<TI>(sdst)));

    for (; len >= (npy_intp)N; len -= N, src1 += ssrc1*N, src2 += ssrc2*N, dst += sdst*N) {
        V x1, x2;
        if (ssrc1 == 1) {
            x1 = hn::LoadU(d, src1);
        } else {
            x1 = hn::GatherIndex(d, src1, idx_in1);
        }
        if (ssrc2 == 1) {
            x2 = hn::LoadU(d, src2);
        } else {
            x2 = hn::GatherIndex(d, src2, idx_in2);
        }
        V out = Pow(d, x1, x2);
        if (sdst == 1) {
            hn::StoreU(out, d, dst);
        } else {
            hn::ScatterIndex(out, d, dst, idx_out1);
        }
    }
    if (len > 0) {
        HWY_ALIGN T buf1[HWY_MAX_LANES_D(decltype(d))] {1};
        HWY_ALIGN T buf2[HWY_MAX_LANES_D(decltype(d))] {1};
        for (npy_intp i = 0; i < len; ++i) {
            buf1[i] = src1[i * ssrc1];
            buf2[i] = src2[i * ssrc2];
        }
        V x1 = hn::LoadU(d, buf1);
        V x2 = hn::LoadU(d, buf2);
        V out = Pow(d, x1, x2);
        hn::StoreU(out, d, buf1);
        for (npy_intp i = 0; i < len; ++i) {
            dst[i * sdst] = buf1[i];
        }
    }
    npyv_cleanup();
}
#endif // NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)

extern "C" {

// ============================================================
// Dispatch loop functions
// ============================================================

#ifdef __aarch64__
#define POWER_FAST_PATH(TYPE, type, sqrt_func)                              \
    int stride_zero = steps[1]==0;                                          \
    if (stride_zero) {                                                      \
        BINARY_DEFS                                                         \
        const type in2 = *(type *)ip2;                                      \
        char* new_args[2];                                                  \
        npy_intp new_steps[2];                                              \
        new_args[0] = args[0];                                              \
        new_args[1] = args[2];                                              \
        new_steps[0] = steps[0];                                            \
        new_steps[1] = steps[2];                                            \
        if (in2 == -1.0) {                                                  \
            TYPE##_reciprocal(new_args, dimensions, new_steps, NULL);       \
            return;                                                         \
        }                                                                   \
        if (in2 == 0.0) {                                                   \
            TYPE##__ones_like(new_args, dimensions, new_steps, NULL);       \
            return;                                                         \
        }                                                                   \
        if (in2 == 0.5) {                                                   \
            TYPE##_sqrt(new_args, dimensions, new_steps, NULL);             \
            return;                                                         \
        }                                                                   \
        if (in2 == 1.0) {                                                   \
            BINARY_LOOP_SLIDING {                                           \
                const type in1 = *(type *)ip1;                              \
                *(type *)op1 = in1;                                         \
            }                                                               \
            return;                                                         \
        }                                                                   \
        if (in2 == 2.0) {                                                   \
            TYPE##_square(new_args, dimensions, new_steps, NULL);           \
            return;                                                         \
        }                                                                   \
    }
#else
#define POWER_FAST_PATH(TYPE, type, sqrt_func)                              \
    int stride_zero = steps[1]==0;                                          \
    if (stride_zero) {                                                      \
        BINARY_DEFS                                                         \
        const type in2 = *(type *)ip2;                                      \
        int fastop_found = 1;                                               \
        BINARY_LOOP_SLIDING {                                               \
            const type in1 = *(type *)ip1;                                  \
            if (in2 == -1.0) {                                              \
                *(type *)op1 = 1.0 / in1;                                   \
            }                                                               \
            else if (in2 == 0.0) {                                          \
                *(type *)op1 = 1.0;                                         \
            }                                                               \
            else if (in2 == 0.5) {                                          \
                *(type *)op1 = sqrt_func(in1);                              \
            }                                                               \
            else if (in2 == 1.0) {                                          \
                *(type *)op1 = in1;                                         \
            }                                                               \
            else if (in2 == 2.0) {                                          \
                *(type *)op1 = in1 * in1;                                   \
            }                                                               \
            else {                                                          \
                fastop_found = 0;                                           \
                break;                                                      \
            }                                                               \
        }                                                                   \
        if (fastop_found) {                                                 \
            return;                                                         \
        }                                                                   \
    }
#endif

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_power)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
{
    POWER_FAST_PATH(DOUBLE, npy_double, npy_sqrt)

#if NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)
    const npy_intp len = dimensions[0];
    const npy_double *src1 = (npy_double*)args[0];
    const npy_double *src2 = (npy_double*)args[1];
          npy_double *dst  = (npy_double*)args[2];

    if (!is_mem_overlap(src1, steps[0], dst, steps[2], len) &&
        !is_mem_overlap(src2, steps[1], dst, steps[2], len) &&
        npyv_loadable_stride_f64(steps[0]) &&
        npyv_loadable_stride_f64(steps[1]) &&
        npyv_storable_stride_f64(steps[2])
    ) {
        const npy_intp ssrc1 = steps[0] / sizeof(npy_double);
        const npy_intp ssrc2 = steps[1] / sizeof(npy_double);
        const npy_intp sdst  = steps[2] / sizeof(npy_double);

        simd_pow_f64(src1, ssrc1, src2, ssrc2, dst, sdst, len);
        return;
    }
#elif NPY_SIMD_FMA3
    const npy_intp len = dimensions[0];
    const npy_double *src1_hwy = (npy_double*)args[0];
    const npy_double *src2_hwy = (npy_double*)args[1];
          npy_double *dst_hwy  = (npy_double*)args[2];

    if (!is_mem_overlap(src1_hwy, steps[0], dst_hwy, steps[2], len) &&
        !is_mem_overlap(src2_hwy, steps[1], dst_hwy, steps[2], len))
    {
        const npy_intp ssrc1 = steps[0] / sizeof(npy_double);
        const npy_intp ssrc2 = steps[1] / sizeof(npy_double);
        const npy_intp sdst  = steps[2] / sizeof(npy_double);

        simd_power<npy_double>(src1_hwy, ssrc1, src2_hwy, ssrc2, dst_hwy, sdst, len);
        return;
    }
#endif

    BINARY_LOOP {
        const npy_double in1 = *(npy_double *)ip1;
        const npy_double in2 = *(npy_double *)ip2;
        *(npy_double *)op1 = npy_pow(in1, in2);
    }
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_power)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(data))
{
    POWER_FAST_PATH(FLOAT, npy_float, npy_sqrtf)

#if NPY_SIMD && defined(NPY_HAVE_AVX512_SKX) && defined(NPY_CAN_LINK_SVML)
    const npy_intp len = dimensions[0];
    const npy_float *src1 = (npy_float*)args[0];
    const npy_float *src2 = (npy_float*)args[1];
          npy_float *dst  = (npy_float*)args[2];

    if (!is_mem_overlap(src1, steps[0], dst, steps[2], len) &&
        !is_mem_overlap(src2, steps[1], dst, steps[2], len) &&
        npyv_loadable_stride_f32(steps[0]) &&
        npyv_loadable_stride_f32(steps[1]) &&
        npyv_storable_stride_f32(steps[2])
    ) {
        const npy_intp ssrc1 = steps[0] / sizeof(npy_float);
        const npy_intp ssrc2 = steps[1] / sizeof(npy_float);
        const npy_intp sdst  = steps[2] / sizeof(npy_float);

        simd_pow_f32(src1, ssrc1, src2, ssrc2, dst, sdst, len);
        return;
    }
#elif NPY_SIMD_FMA3
    const npy_intp len = dimensions[0];
    const npy_float *src1_hwy = (npy_float*)args[0];
    const npy_float *src2_hwy = (npy_float*)args[1];
          npy_float *dst_hwy  = (npy_float*)args[2];

    if (!is_mem_overlap(src1_hwy, steps[0], dst_hwy, steps[2], len) &&
        !is_mem_overlap(src2_hwy, steps[1], dst_hwy, steps[2], len))
    {
        const npy_intp ssrc1 = steps[0] / sizeof(npy_float);
        const npy_intp ssrc2 = steps[1] / sizeof(npy_float);
        const npy_intp sdst  = steps[2] / sizeof(npy_float);

        simd_power<npy_float>(src1_hwy, ssrc1, src2_hwy, ssrc2, dst_hwy, sdst, len);
        return;
    }
#endif

    BINARY_LOOP {
        const npy_float in1 = *(npy_float *)ip1;
        const npy_float in2 = *(npy_float *)ip2;
        *(npy_float *)op1 = npy_powf(in1, in2);
    }
}

}