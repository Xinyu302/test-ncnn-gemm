#include <riscv_vector.h>
#include <stdio.h>
#include "utils.h"
// #undef __aarch64__
#define PRINT_MAT 1
#define PACK_8 1

#if PACK_8

#else
#undef __aarch64__    
#endif

static void pack_B_tile_naive(const Mat<>& B, Mat<>& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    float* pp = BT;

    int jj = 0;
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;
            const float* p1 = (const float*)B + (j + jj + 4) * B_hstep + k * 4;
            const float* p2 = (const float*)B + (j + jj + 8) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p1[0];
                pp[5] = p1[1];
                pp[6] = p1[2];
                pp[7] = p1[3];
                pp[8] = p2[0];
                pp[9] = p2[1];
                pp[10] = p2[2];
                pp[11] = p2[3];
                pp += 12;
                p0 += 4;
                p1 += 4;
                p2 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
            const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
            const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;
            const float* p4 = (const float*)B + (j + jj + 4) * B_hstep + k;
            const float* p5 = (const float*)B + (j + jj + 5) * B_hstep + k;
            const float* p6 = (const float*)B + (j + jj + 6) * B_hstep + k;
            const float* p7 = (const float*)B + (j + jj + 7) * B_hstep + k;
            const float* p8 = (const float*)B + (j + jj + 8) * B_hstep + k;
            const float* p9 = (const float*)B + (j + jj + 9) * B_hstep + k;
            const float* pa = (const float*)B + (j + jj + 10) * B_hstep + k;
            const float* pb = (const float*)B + (j + jj + 11) * B_hstep + k;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp[4] = p4[0];
                pp[5] = p5[0];
                pp[6] = p6[0];
                pp[7] = p7[0];
                pp[8] = p8[0];
                pp[9] = p9[0];
                pp[10] = pa[0];
                pp[11] = pb[0];
                pp += 12;
                p0++;
                p1++;
                p2++;
                p3++;
                p4++;
                p5++;
                p6++;
                p7++;
                p8++;
                p9++;
                pa++;
                pb++;
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;
            const float* p1 = (const float*)B + (j + jj + 4) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p1[0];
                pp[5] = p1[1];
                pp[6] = p1[2];
                pp[7] = p1[3];
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
            const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
            const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;
            const float* p4 = (const float*)B + (j + jj + 4) * B_hstep + k;
            const float* p5 = (const float*)B + (j + jj + 5) * B_hstep + k;
            const float* p6 = (const float*)B + (j + jj + 6) * B_hstep + k;
            const float* p7 = (const float*)B + (j + jj + 7) * B_hstep + k;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp[4] = p4[0];
                pp[5] = p5[0];
                pp[6] = p6[0];
                pp[7] = p7[0];
                pp += 8;
                p0++;
                p1++;
                p2++;
                p3++;
                p4++;
                p5++;
                p6++;
                p7++;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];

                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
            const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
            const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp += 4;
                p0++;
                p1++;
                p2++;
                p3++;
            }
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp += 2;
                p0++;
                p1++;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}


static void pack_B_tile(const Mat<>& B, Mat<>& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    float* pp = BT;

    int jj = 0;
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;
            const float* p1 = (const float*)B + (j + jj + 4) * B_hstep + k * 4;
            const float* p2 = (const float*)B + (j + jj + 8) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p1, VL), VL);
                vse32_v_f32m1(pp + 8, vle32_v_f32m1(p2, VL), VL);
                pp += 12;
                p0 += 4;
                p1 += 4;
                p2 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
            const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
            const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;
            const float* p4 = (const float*)B + (j + jj + 4) * B_hstep + k;
            const float* p5 = (const float*)B + (j + jj + 5) * B_hstep + k;
            const float* p6 = (const float*)B + (j + jj + 6) * B_hstep + k;
            const float* p7 = (const float*)B + (j + jj + 7) * B_hstep + k;
            const float* p8 = (const float*)B + (j + jj + 8) * B_hstep + k;
            const float* p9 = (const float*)B + (j + jj + 9) * B_hstep + k;
            const float* pa = (const float*)B + (j + jj + 10) * B_hstep + k;
            const float* pb = (const float*)B + (j + jj + 11) * B_hstep + k;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1_t _r0 = vle32_v_f32m1(p0, VL);
                vfloat32m1_t _r1 = vle32_v_f32m1(p1, VL);
                vfloat32m1_t _r2 = vle32_v_f32m1(p2, VL);
                vfloat32m1_t _r3 = vle32_v_f32m1(p3, VL);
                vfloat32m1_t _r4 = vle32_v_f32m1(p4, VL);
                vfloat32m1_t _r5 = vle32_v_f32m1(p5, VL);
                vfloat32m1_t _r6 = vle32_v_f32m1(p6, VL);
                vfloat32m1_t _r7 = vle32_v_f32m1(p7, VL);
                vfloat32m1_t _r8 = vle32_v_f32m1(p8, VL);
                vfloat32m1_t _r9 = vle32_v_f32m1(p9, VL);
                vfloat32m1_t _ra = vle32_v_f32m1(pa, VL);
                vfloat32m1_t _rb = vle32_v_f32m1(pb, VL);

                transpose4x4_ps(_r0, _r1, _r2, _r3, VL);
                transpose4x4_ps(_r4, _r5, _r6, _r7, VL);
                transpose4x4_ps(_r8, _r9, _ra, _rb, VL);

                vse32_v_f32m1(pp, _r0, VL);
                vse32_v_f32m1(pp + 4, _r4, VL);
                vse32_v_f32m1(pp + 4 * 2, _r8, VL);
                vse32_v_f32m1(pp + 4 * 3, _r1, VL);
                vse32_v_f32m1(pp + 4 * 4, _r5, VL);
                vse32_v_f32m1(pp + 4 * 5, _r9, VL);
                vse32_v_f32m1(pp + 4 * 6, _r2, VL);
                vse32_v_f32m1(pp + 4 * 7, _r6, VL);
                vse32_v_f32m1(pp + 4 * 8, _ra, VL);
                vse32_v_f32m1(pp + 4 * 9, _r3, VL);
                vse32_v_f32m1(pp + 4 * 10, _r7, VL);
                vse32_v_f32m1(pp + 4 * 11, _rb, VL);
                pp += 48;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
                p4 += 4;
                p5 += 4;
                p6 += 4;
                p7 += 4;
                p8 += 4;
                p9 += 4;
                pa += 4;
                pb += 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp[4] = p4[0];
                pp[5] = p5[0];
                pp[6] = p6[0];
                pp[7] = p7[0];
                pp[8] = p8[0];
                pp[9] = p9[0];
                pp[10] = pa[0];
                pp[11] = pb[0];
                pp += 12;
                p0++;
                p1++;
                p2++;
                p3++;
                p4++;
                p5++;
                p6++;
                p7++;
                p8++;
                p9++;
                pa++;
                pb++;
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;
            const float* p1 = (const float*)B + (j + jj + 4) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p1, VL), VL);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
            const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
            const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;
            const float* p4 = (const float*)B + (j + jj + 4) * B_hstep + k;
            const float* p5 = (const float*)B + (j + jj + 5) * B_hstep + k;
            const float* p6 = (const float*)B + (j + jj + 6) * B_hstep + k;
            const float* p7 = (const float*)B + (j + jj + 7) * B_hstep + k;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1_t _r0 = vle32_v_f32m1(p0, VL);
                vfloat32m1_t _r1 = vle32_v_f32m1(p1, VL);
                vfloat32m1_t _r2 = vle32_v_f32m1(p2, VL);
                vfloat32m1_t _r3 = vle32_v_f32m1(p3, VL);
                vfloat32m1_t _r4 = vle32_v_f32m1(p4, VL);
                vfloat32m1_t _r5 = vle32_v_f32m1(p5, VL);
                vfloat32m1_t _r6 = vle32_v_f32m1(p6, VL);
                vfloat32m1_t _r7 = vle32_v_f32m1(p7, VL);

                transpose4x4_ps(_r0, _r1, _r2, _r3, VL);
                transpose4x4_ps(_r4, _r5, _r6, _r7, VL);

                vse32_v_f32m1(pp, _r0, VL);
                vse32_v_f32m1(pp + 4, _r4, VL);
                vse32_v_f32m1(pp + 4 * 2, _r1, VL);
                vse32_v_f32m1(pp + 4 * 3, _r5, VL);
                vse32_v_f32m1(pp + 4 * 4, _r2, VL);
                vse32_v_f32m1(pp + 4 * 5, _r6, VL);
                vse32_v_f32m1(pp + 4 * 6, _r3, VL);
                vse32_v_f32m1(pp + 4 * 7, _r7, VL);
                pp += 32;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
                p4 += 4;
                p5 += 4;
                p6 += 4;
                p7 += 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp[4] = p4[0];
                pp[5] = p5[0];
                pp[6] = p6[0];
                pp[7] = p7[0];
                pp += 8;
                p0++;
                p1++;
                p2++;
                p3++;
                p4++;
                p5++;
                p6++;
                p7++;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
            const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
            const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // vfloat32m1x4_t _r0123;
                // vget_f32m1x4_f32m1(_r0123, 0) = vle32_v_f32m1(p0, VL);
                // vget_f32m1x4_f32m1(_r0123, 1) = vle32_v_f32m1(p1, VL);
                // vget_f32m1x4_f32m1(_r0123, 2) = vle32_v_f32m1(p2, VL);
                // vget_f32m1x4_f32m1(_r0123, 3) = vle32_v_f32m1(p3, VL);
                // vst4q_f32(pp, _r0123);
                store_float_v4(vle32_v_f32m1(p0, VL), vle32_v_f32m1(p1, VL), vle32_v_f32m1(p2, VL), vle32_v_f32m1(p3, VL), pp);
                pp += 16;
                p0 += 4;
                p1 += 4;
                p2 += 4;
                p3 += 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp += 4;
                p0++;
                p1++;
                p2++;
                p3++;
            }
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // float32x4x2_t _r01;
                // vget_f32m1x4_f32m1(_r01, 0) = vle32_v_f32m1(p0, VL);
                // vget_f32m1x4_f32m1(_r01, 1) = vle32_v_f32m1(p1, VL);
                // vst2q_f32(pp, _r01);
                store_float32_v2(vle32_v_f32m1(p0, VL), vle32_v_f32m1(p1, VL), pp);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp += 2;
                p0++;
                p1++;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                pp += 4;
                p0 += 4;
            }
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_B_tile_naive(const Mat<>& B, Mat<>& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    float* pp = BT;

    int jj = 0;
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // vfloat32m1x4_t _r0123 = vlseg4e32_v_f32m1x4(p0, VL);
                // vfloat32m1x4_t _r4567 = vlseg4e32_v_f32m1x4(p0 + 16, VL);
                // vfloat32m1x4_t _r89ab = vlseg4e32_v_f32m1x4(p0 + 32, VL);
                // vse32_v_f32m1(pp, vget_f32m1x4_f32m1(_r0123, 0), VL);
                // vse32_v_f32m1(pp + 4, vget_f32m1x4_f32m1(_r4567, 0), VL);
                // vse32_v_f32m1(pp + 4 * 2, vget_f32m1x4_f32m1(_r89ab, 0), VL);
                // vse32_v_f32m1(pp + 4 * 3, vget_f32m1x4_f32m1(_r0123, 1), VL);
                // vse32_v_f32m1(pp + 4 * 4, vget_f32m1x4_f32m1(_r4567, 1), VL);
                // vse32_v_f32m1(pp + 4 * 5, vget_f32m1x4_f32m1(_r89ab, 1), VL);
                // vse32_v_f32m1(pp + 4 * 6, vget_f32m1x4_f32m1(_r0123, 2), VL);
                // vse32_v_f32m1(pp + 4 * 7, vget_f32m1x4_f32m1(_r4567, 2), VL);
                // vse32_v_f32m1(pp + 4 * 8, vget_f32m1x4_f32m1(_r89ab, 2), VL);
                // vse32_v_f32m1(pp + 4 * 9, vget_f32m1x4_f32m1(_r0123, 3), VL);
                // vse32_v_f32m1(pp + 4 * 10, vget_f32m1x4_f32m1(_r4567, 3), VL);
                // vse32_v_f32m1(pp + 4 * 11, vget_f32m1x4_f32m1(_r89ab, 3), VL);
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p0[16];
                pp[5] = p0[17];
                pp[6] = p0[18];
                pp[7] = p0[19];
                pp[8] = p0[32];
                pp[9] = p0[33];
                pp[10] = p0[34];
                pp[11] = p0[35];
                pp[12] = p0[4];
                pp[13] = p0[5];
                pp[14] = p0[6];
                pp[15] = p0[7];
                pp[16] = p0[20];
                pp[17] = p0[21];
                pp[18] = p0[22];
                pp[19] = p0[23];
                pp[20] = p0[36];
                pp[21] = p0[37];
                pp[22] = p0[38];
                pp[23] = p0[39];
                pp[24] = p0[8];
                pp[25] = p0[9];
                pp[26] = p0[10];
                pp[27] = p0[11];
                pp[28] = p0[24];
                pp[29] = p0[25];
                pp[30] = p0[26];
                pp[31] = p0[27];
                pp[32] = p0[40];
                pp[33] = p0[41];
                pp[34] = p0[42];
                pp[35] = p0[43];
                pp[36] = p0[12];
                pp[37] = p0[13];
                pp[38] = p0[14];
                pp[39] = p0[15];
                pp[40] = p0[28];
                pp[41] = p0[29];
                pp[42] = p0[30];
                pp[43] = p0[31];
                pp[44] = p0[44];
                pp[45] = p0[45];
                pp[46] = p0[46];
                pp[47] = p0[47];
                pp += 48;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p0[4];
                pp[5] = p0[5];
                pp[6] = p0[6];
                pp[7] = p0[7];
                pp[8] = p0[8];
                pp[9] = p0[9];
                pp[10] = p0[10];
                pp[11] = p0[11];
                pp[12] = p0[12];
                pp += 12;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p0[16];
                pp[5] = p0[17];
                pp[6] = p0[18];
                pp[7] = p0[19];
                pp[8] = p0[4];
                pp[9] = p0[5];
                pp[10] = p0[6];
                pp[11] = p0[7];
                pp[12] = p0[20];
                pp[13] = p0[21];
                pp[14] = p0[22];
                pp[15] = p0[23];
                pp[16] = p0[8];
                pp[17] = p0[9];
                pp[18] = p0[10];
                pp[19] = p0[11];
                pp[20] = p0[24];
                pp[21] = p0[25];
                pp[22] = p0[26];
                pp[23] = p0[27];
                pp[24] = p0[12];
                pp[25] = p0[13];
                pp[26] = p0[14];
                pp[27] = p0[15];
                pp[28] = p0[28];
                pp[29] = p0[29];
                pp[30] = p0[30];
                pp[31] = p0[31];
                pp += 32;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp[4] = p0[4];
                pp[5] = p0[5];
                pp[6] = p0[6];
                pp[7] = p0[7];
                pp += 8;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                for (int iter = 0; iter < 16; iter++) {
                    pp[iter] = p0[iter];
                }
                pp += 16;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                for (int iter = 0; iter < 8; iter++) {
                    pp[iter] = p0[iter];
                }
                pp += 8;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += B_hstep;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                for (int iter = 0; iter < 4; iter++) {
                    pp[iter] = p0[iter];
                }
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += B_hstep;
            }
        }
    }
}

static void transpose_pack_B_tile(const Mat<>& B, Mat<>& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    float* pp = BT;

    int jj = 0;
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1x4_t _r0123 = vlseg4e32_v_f32m1x4(p0, VL);
                vfloat32m1x4_t _r4567 = vlseg4e32_v_f32m1x4(p0 + 16, VL);
                vfloat32m1x4_t _r89ab = vlseg4e32_v_f32m1x4(p0 + 32, VL);
                vse32_v_f32m1(pp, vget_f32m1x4_f32m1(_r0123, 0), VL);
                vse32_v_f32m1(pp + 4, vget_f32m1x4_f32m1(_r4567, 0), VL);
                vse32_v_f32m1(pp + 4 * 2, vget_f32m1x4_f32m1(_r89ab, 0), VL);
                vse32_v_f32m1(pp + 4 * 3, vget_f32m1x4_f32m1(_r0123, 1), VL);
                vse32_v_f32m1(pp + 4 * 4, vget_f32m1x4_f32m1(_r4567, 1), VL);
                vse32_v_f32m1(pp + 4 * 5, vget_f32m1x4_f32m1(_r89ab, 1), VL);
                vse32_v_f32m1(pp + 4 * 6, vget_f32m1x4_f32m1(_r0123, 2), VL);
                vse32_v_f32m1(pp + 4 * 7, vget_f32m1x4_f32m1(_r4567, 2), VL);
                vse32_v_f32m1(pp + 4 * 8, vget_f32m1x4_f32m1(_r89ab, 2), VL);
                vse32_v_f32m1(pp + 4 * 9, vget_f32m1x4_f32m1(_r0123, 3), VL);
                vse32_v_f32m1(pp + 4 * 10, vget_f32m1x4_f32m1(_r4567, 3), VL);
                vse32_v_f32m1(pp + 4 * 11, vget_f32m1x4_f32m1(_r89ab, 3), VL);
                pp += 48;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p0 + 4, VL), VL);
                vse32_v_f32m1(pp + 8, vle32_v_f32m1(p0 + 8, VL), VL);
                pp += 12;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1x4_t _r0123 = vlseg4e32_v_f32m1x4(p0, VL);
                vfloat32m1x4_t _r4567 = vlseg4e32_v_f32m1x4(p0 + 16, VL);
                vse32_v_f32m1(pp, vget_f32m1x4_f32m1(_r0123, 0), VL);
                vse32_v_f32m1(pp + 4, vget_f32m1x4_f32m1(_r4567, 0), VL);
                vse32_v_f32m1(pp + 4 * 2, vget_f32m1x4_f32m1(_r0123, 1), VL);
                vse32_v_f32m1(pp + 4 * 3, vget_f32m1x4_f32m1(_r4567, 1), VL);
                vse32_v_f32m1(pp + 4 * 4, vget_f32m1x4_f32m1(_r0123, 2), VL);
                vse32_v_f32m1(pp + 4 * 5, vget_f32m1x4_f32m1(_r4567, 2), VL);
                vse32_v_f32m1(pp + 4 * 6, vget_f32m1x4_f32m1(_r0123, 3), VL);
                vse32_v_f32m1(pp + 4 * 7, vget_f32m1x4_f32m1(_r4567, 3), VL);
                pp += 32;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p0 + 4, VL), VL);
                pp += 8;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1x4_t _r0123 = vlseg4e32_v_f32m1x4(p0, VL);
                vse32_v_f32m1(pp, vget_f32m1x4_f32m1(_r0123, 0), VL);
                vse32_v_f32m1(pp + 4, vget_f32m1x4_f32m1(_r0123, 1), VL);
                vse32_v_f32m1(pp + 4 * 2, vget_f32m1x4_f32m1(_r0123, 2), VL);
                vse32_v_f32m1(pp + 4 * 3, vget_f32m1x4_f32m1(_r0123, 3), VL);
                pp += 16;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 1 < max_jj; jj += 2)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // float32x4x2_t _r01;
                // vget_f32m1x4_f32m1(_r01, 0) = vle32_v_f32m1(p0, VL);
                // vget_f32m1x4_f32m1(_r01, 1) = vle32_v_f32m1(p0 + 4, VL);
                // vst2q_f32(pp, _r01);
                store_float32_v2(vle32_v_f32m1(p0, VL), vle32_v_f32m1(p0 + 4, VL), pp);
                pp += 8;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += B_hstep;
            }
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += B_hstep;
            }
        }
    }
}

int main()
{
    int h = 27, w = 27;
    Mat<> in;
    Mat<> out;
    Mat<> out_check;
    in.w = w;
    in.h = h;
    out.w = w;
    out.h = h;
    out_check.w = w;
    out_check.h = h;

    float *data_in = new float[h*w];
    float *data_out = new float[h*w];
    float *data_check = new float[h*w];
    in.data = data_in;
    out.data = data_out;
    out_check.data = data_check;
    init_Mat(in);
    // check function pack_A_tile
    pack_B_tile(in, out, 0, h, 0, w);
    pack_B_tile_naive(in, out_check, 0, h, 0, w);

#if PRINT_MAT
    printf("-----------Origin Matrix------------\n");
    print_Mat(in);
    printf("------------------------\n");
    printf("-----------pack_B_tile------------\n");
    print_Mat(out);
    printf("------------------------\n");
    // print_Mat(out_check);
    // printf("------------------------\n");
#endif

    if (!check_Mat(out, out_check)) {
        printf("error\n");
    } else {
        printf("correct\n");
    }
    // check function transpose_pack_A_tile
    transpose_pack_B_tile(in, out, 0, h, 0, w);
    transpose_pack_B_tile_naive(in, out_check, 0, h, 0, w);
#if PRINT_MAT
    printf("------transpose_pack_B_tile------------\n");
    print_Mat(out);
#endif // PRINT_MAT
    if (!check_Mat(out, out_check)) {
        printf("error\n");
    } else {
        printf("correct\n");
    }
    delete [] data_in, data_out;

    return 0;
}