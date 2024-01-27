#include <riscv_vector.h>
#include <stdio.h>
#include "utils.h"
#define PRINT_MAT 1
#define PACK_8 1

#if PACK_8

#else
#undef __aarch64__    
#endif

#define VL 8
// #define vl 8
int vl;

static void pack_A_tile_fp32_to_fp16(const Mat<>& A, Mat<__fp16>& AT, int i, int max_ii, int k, int max_kk)
{
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    __fp16* pp = AT;

    int ii = 0;
#if __riscv_vector
    for (; ii + 7 < max_ii; ii += 8)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;
        const float* p4 = (const float*)A + (i + ii + 4) * A_hstep + k;
        const float* p5 = (const float*)A + (i + ii + 5) * A_hstep + k;
        const float* p6 = (const float*)A + (i + ii + 6) * A_hstep + k;
        const float* p7 = (const float*)A + (i + ii + 7) * A_hstep + k;

        int kk = 0;

        int n = max_kk;
        while (n > 0) {
            vl = vsetvl_e32m2(n);
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vfloat16m1_t _r1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p1, vl), vl);
            vfloat16m1_t _r2 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p2, vl), vl);
            vfloat16m1_t _r3 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p3, vl), vl);
            vfloat16m1_t _r4 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p4, vl), vl);
            vfloat16m1_t _r5 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p5, vl), vl);
            vfloat16m1_t _r6 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p6, vl), vl);
            vfloat16m1_t _r7 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p7, vl), vl);
            vsseg8e16_v_f16m1(pp, _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);
            pp += 8 * vl;
            p0 += vl;
            p1 += vl;
            p2 += vl;
            p3 += vl;
            n -= vl;
        }

        // for (; kk + 7 < max_kk; kk += 8)
        // {
        //     vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
        //     vfloat16m1_t _r1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p1, vl), vl);
        //     vfloat16m1_t _r2 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p2, vl), vl);
        //     vfloat16m1_t _r3 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p3, vl), vl);
        //     vfloat16m1_t _r4 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p4, vl), vl);
        //     vfloat16m1_t _r5 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p5, vl), vl);
        //     vfloat16m1_t _r6 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p6, vl), vl);
        //     vfloat16m1_t _r7 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p7, vl), vl);

        //     // uint16x8_t _r0 = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p0, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(p0 + 4, vl / 2)), vl / 2);
        //     // uint16x8_t _r1 = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p1, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(p1 + 4, vl / 2)), vl / 2);
        //     // uint16x8_t _r2 = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p2, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(p2 + 4, vl / 2)), vl / 2);
        //     // uint16x8_t _r3 = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p3, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(p3 + 4, vl / 2)), vl / 2);
        //     // uint16x8_t _r4 = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p4, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(p4 + 4, vl / 2)), vl / 2);
        //     // uint16x8_t _r5 = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p5, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(p5 + 4, vl / 2)), vl / 2);
        //     // uint16x8_t _r6 = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p6, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(p6 + 4, vl / 2)), vl / 2);
        //     // uint16x8_t _r7 = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p7, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(p7 + 4, vl / 2)), vl / 2);
        //     transpose8x8_f16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
        //     vse16_v_f16m1(pp, _r0, vl);
        //     vse16_v_f16m1(pp + 8, _r1, vl);
        //     vse16_v_f16m1(pp + 8 * 2, _r2, vl);
        //     vse16_v_f16m1(pp + 8 * 3, _r3, vl);
        //     vse16_v_f16m1(pp + 8 * 4, _r4, vl);
        //     vse16_v_f16m1(pp + 8 * 5, _r5, vl);
        //     vse16_v_f16m1(pp + 8 * 6, _r6, vl);
        //     vse16_v_f16m1(pp + 8 * 7, _r7, vl);

        //     // vst1q_u16(pp, _r0);
        //     // vst1q_u16(pp + 8, _r1);
        //     // vst1q_u16(pp + 8 * 2, _r2);
        //     // vst1q_u16(pp + 8 * 3, _r3);
        //     // vst1q_u16(pp + 8 * 4, _r4);
        //     // vst1q_u16(pp + 8 * 5, _r5);
        //     // vst1q_u16(pp + 8 * 6, _r6);
        //     // vst1q_u16(pp + 8 * 7, _r7);
        //     pp += 64;
        //     p0 += 8;
        //     p1 += 8;
        //     p2 += 8;
        //     p3 += 8;
        //     p4 += 8;
        //     p5 += 8;
        //     p6 += 8;
        //     p7 += 8;
        // }
        // for (; kk < max_kk; kk++)
        // {
        //     pp[0] = float32_to_float16(p0[0]);
        //     pp[1] = float32_to_float16(p1[0]);
        //     pp[2] = float32_to_float16(p2[0]);
        //     pp[3] = float32_to_float16(p3[0]);
        //     pp[4] = float32_to_float16(p4[0]);
        //     pp[5] = float32_to_float16(p5[0]);
        //     pp[6] = float32_to_float16(p6[0]);
        //     pp[7] = float32_to_float16(p7[0]);
        //     pp += 8;
        //     p0++;
        //     p1++;
        //     p2++;
        //     p3++;
        //     p4++;
        //     p5++;
        //     p6++;
        //     p7++;
        // }
    }
#endif // __riscv_vector
    for (; ii + 3 < max_ii; ii += 4)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
        const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
        const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

        int kk = 0;

        int n = max_kk;
        while (n > 0) {
            vl = vsetvl_e32m2(n);
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vfloat16m1_t _r1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p1, vl), vl);
            vfloat16m1_t _r2 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p2, vl), vl);
            vfloat16m1_t _r3 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p3, vl), vl);
            vsseg4e16_v_f16m1(pp, _r0, _r1, _r2, _r3, vl);
            pp += 4 * vl;
            p0 += vl;
            p1 += vl;
            p2 += vl;
            p3 += vl;
            n -= vl;
        }


        // for (; kk + 7 < max_kk; kk += 8)
        // {
        //     vfloat16m1_t v0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
        //     vfloat16m1_t v1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p1, vl), vl);
        //     vfloat16m1_t v2 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p2, vl), vl);
        //     vfloat16m1_t v3 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p3, vl), vl);
        //     vsseg4e16_v_f16m1(pp, v0, v1, v2, v3, vl);

        //     // uint16x8x4_t _r0123;
        //     // _r0123.val[0] = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p0, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(p0 + 4, vl / 2)), vl / 2);
        //     // _r0123.val[1] = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p1, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(p1 + 4, vl / 2)), vl / 2);
        //     // _r0123.val[2] = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p2, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(p2 + 4, vl / 2)), vl / 2);
        //     // _r0123.val[3] = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p3, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(p3 + 4, vl / 2)), vl / 2);
        //     // vst4q_u16(pp, _r0123);
        //     pp += 32;
        //     p0 += 8;
        //     p1 += 8;
        //     p2 += 8;
        //     p3 += 8;
        // }
        // for (; kk + 3 < max_kk; kk += 4)
        // {
        //     vfloat16mf2_t v0 = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p0, vl / 2), vl / 2);
        //     vfloat16mf2_t v1 = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p1, vl / 2), vl / 2);
        //     vfloat16mf2_t v2 = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p2, vl / 2), vl / 2);
        //     vfloat16mf2_t v3 = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p3, vl / 2), vl / 2);
        //     vsseg4e16_v_f16mf2(pp, v0, v1, v2, v3, vl / 2);

        //     // uint16x4x4_t _r0123;
        //     // _r0123.val[0] = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p0, vl / 2), vl / 2);
        //     // _r0123.val[1] = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p1, vl / 2), vl / 2);
        //     // _r0123.val[2] = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p2, vl / 2), vl / 2);
        //     // _r0123.val[3] = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p3, vl / 2), vl / 2);
        //     // vst4_u16(pp, _r0123);
        //     pp += 16;
        //     p0 += 4;
        //     p1 += 4;
        //     p2 += 4;
        //     p3 += 4;
        // }
        // for (; kk < max_kk; kk++)
        // {
        //     pp[0] = float32_to_float16(p0[0]);
        //     pp[1] = float32_to_float16(p1[0]);
        //     pp[2] = float32_to_float16(p2[0]);
        //     pp[3] = float32_to_float16(p3[0]);
        //     pp += 4;
        //     p0++;
        //     p1++;
        //     p2++;
        //     p3++;
        // }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

        int kk = 0;
        int n = max_kk;
        while (n > 0) {
            vl = vsetvl_e32m2(n);
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vfloat16m1_t _r1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p1, vl), vl);
            vsseg2e16_v_f16m1(pp, _r0, _r1, vl);
            pp += 2 * vl;
            p0 += vl;
            p1 += vl;
            n -= vl;
        }
        // for (; kk + 7 < max_kk; kk += 8)
        // {
        //     vfloat16m1_t v0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
        //     vfloat16m1_t v1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p1, vl), vl);
        //     vsseg2e16_v_f16m1(pp, v0, v1, vl);

        //     // uint16x8x2_t _r01;
        //     // _r01.val[0] = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p0, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(p0 + 4, vl / 2)), vl / 2);
        //     // _r01.val[1] = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p1, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(p1 + 4, vl / 2)), vl / 2);
        //     // vst2q_u16(pp, _r01);
        //     pp += 16;
        //     p0 += 8;
        //     p1 += 8;
        // }
        // for (; kk + 3 < max_kk; kk += 4)
        // {
        //     vfloat16mf2_t v0 = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p0, vl / 2), vl / 2);
        //     vfloat16mf2_t v1 = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p1, vl / 2), vl / 2);
        //     vsseg2e16_v_f16mf2(pp, v0, v1, vl / 2);
        //     // uint16x4x2_t _r01;
        //     // _r01.val[0] = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p0, vl / 2), vl / 2);
        //     // _r01.val[1] = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p1, vl / 2), vl / 2);
        //     // vst2_u16(pp, _r01);
        //     pp += 8;
        //     p0 += 4;
        //     p1 += 4;
        // }
        // for (; kk < max_kk; kk++)
        // {
        //     pp[0] = float32_to_float16(p0[0]);
        //     pp[1] = float32_to_float16(p1[0]);
        //     pp += 2;
        //     p0++;
        //     p1++;
        // }
    }
    for (; ii < max_ii; ii += 1)
    {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

        int kk = 0;
        int n = max_kk;

        while (n > 0) {
            vl = vsetvl_e32m2(n);
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vse16_v_f16m1(pp, _r0, vl);
            pp += vl;
            p0 += vl;
            n -= vl;
        }

        // for (; kk + 7 < max_kk; kk += 8)
        // {
        //     vfloat16m1_t v0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
        //     vse16_v_f16m1(pp, v0, vl);
        //     // uint16x8_t _r0 = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p0, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(p0 + 4, vl / 2)), vl / 2);
        //     // vst1q_u16(pp, _r0);
        //     pp += 8;
        //     p0 += 8;
        // }
        // for (; kk + 3 < max_kk; kk += 4)
        // {
        //     vfloat16mf2_t v0 = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p0, vl / 2), vl / 2);
        //     vse16_v_f16mf2(pp, v0, vl / 2);
        //     // uint16x4_t _r0 = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p0, vl / 2), vl / 2);
        //     // vse16_v_f16mf2(pp, _r0, vl / 2);
        //     pp += 4;
        //     p0 += 4;
        // }
        // for (; kk < max_kk; kk++)
        // {
        //     pp[0] = float32_to_float16(p0[0]);
        //     pp += 1;
        //     p0++;
        // }
    }
}

static void transpose_pack_A_tile_fp32_to_fp16(const Mat<>& A, Mat<__fp16>& AT, int i, int max_ii, int k, int max_kk)
{
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    __fp16* pp = AT;

    int ii = 0;
#if __riscv_vector
    int n = max_ii;
    while (n > 0) {
        vl = vsetvl_e32m2(n);
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vse16_v_f16m1(pp, _r0, vl);
            // uint16x8_t _r0 = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p0, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(p0 + 4, vl / 2)), vl / 2);
            // vst1q_u16(pp, _r0);
            pp += vl;
            p0 += A_hstep;
        }
        ii += vl;
        n -= vl;
    }


#endif


// #if __riscv_vector
//     for (; ii + 7 < max_ii; ii += 8)
//     {
//         const float* p0 = (const float*)A + k * A_hstep + (i + ii);

//         int kk = 0;
//         for (; kk < max_kk; kk++)
//         {
//             vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
//             vse16_v_f16m1(pp, _r0, vl);
//             // uint16x8_t _r0 = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p0, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(p0 + 4, vl / 2)), vl / 2);
//             // vst1q_u16(pp, _r0);
//             pp += 8;
//             p0 += A_hstep;
//         }
//     }
// #endif // __riscv_vector
//     for (; ii + 3 < max_ii; ii += 4)
//     {
//         const float* p0 = (const float*)A + k * A_hstep + (i + ii);

//         int kk = 0;
//         for (; kk < max_kk; kk++)
//         {
//             vfloat16mf2_t _r0 = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p0, vl / 2), vl / 2);
//             vse16_v_f16mf2(pp, _r0, vl / 2);
//             // uint16x4_t _r0 = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(p0, vl / 2), vl / 2);
//             // vse16_v_f16mf2(pp, _r0, vl / 2);
//             pp += 4;
//             p0 += A_hstep;
//         }
//     }
//     for (; ii + 1 < max_ii; ii += 2)
//     {
//         const float* p0 = (const float*)A + k * A_hstep + (i + ii);

//         int kk = 0;
//         for (; kk < max_kk; kk++)
//         {
//             pp[0] = float32_to_float16(p0[0]);
//             pp[1] = float32_to_float16(p0[1]);
//             pp += 2;
//             p0 += A_hstep;
//         }
//     }
//     for (; ii < max_ii; ii += 1)
//     {
//         const float* p0 = (const float*)A + k * A_hstep + (i + ii);

//         int kk = 0;
//         for (; kk < max_kk; kk++)
//         {
//             pp[0] = float32_to_float16(p0[0]);
//             pp += 1;
//             p0 += A_hstep;
//         }
//     }
}

int main() {
    printf("in main\n");
    int h = 15, w = 15;
    Mat<> in;
    Mat<__fp16> out;
    Mat<__fp16> out_check;
    in.w = w;
    in.h = h;
    out.w = w;
    out.h = h;
    out_check.w = w;
    out_check.h = h;
    float *data_in = new float[h*w];
    __fp16 *data_out = new __fp16[h*w];
    in.data = data_in;
    out.data = data_out;
    printf("before init mat\n");

    init_Mat(in);

    printf("after init mat\n");
    pack_A_tile_fp32_to_fp16(in, out, 0, h, 0, w);


#if PRINT_MAT
    printf("--------Origin Matrix--------\n");
    print_Mat(in);
    printf("--------Packed Matrix--------\n");
    print_Mat<__fp16>(out);
#endif

    transpose_pack_A_tile_fp32_to_fp16(in, out, 0, h, 0, w);

#if PRINT_MAT

    printf("--------Origin Matrix--------\n");
    print_Mat(in);
    printf("--------Transpose Packed Matrix--------\n");
    print_Mat<__fp16>(out);
#endif

}