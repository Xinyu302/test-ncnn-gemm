#include <arm_neon.h>
#include <stdio.h>
#include "utils.h"
// #undef __aarch64__
#define PRINT_MAT 1
#define PACK_8 1

#if PACK_8

#else
#undef __aarch64__    
#endif

static void pack_B_tile_naive(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
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


static void pack_B_tile(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    float* pp = BT;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;
            const float* p1 = (const float*)B + (j + jj + 4) * B_hstep + k * 4;
            const float* p2 = (const float*)B + (j + jj + 8) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                vst1q_f32(pp + 4, vld1q_f32(p1));
                vst1q_f32(pp + 8, vld1q_f32(p2));
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
                float32x4_t _r0 = vld1q_f32(p0);
                float32x4_t _r1 = vld1q_f32(p1);
                float32x4_t _r2 = vld1q_f32(p2);
                float32x4_t _r3 = vld1q_f32(p3);
                float32x4_t _r4 = vld1q_f32(p4);
                float32x4_t _r5 = vld1q_f32(p5);
                float32x4_t _r6 = vld1q_f32(p6);
                float32x4_t _r7 = vld1q_f32(p7);
                float32x4_t _r8 = vld1q_f32(p8);
                float32x4_t _r9 = vld1q_f32(p9);
                float32x4_t _ra = vld1q_f32(pa);
                float32x4_t _rb = vld1q_f32(pb);

                transpose4x4_ps(_r0, _r1, _r2, _r3);
                transpose4x4_ps(_r4, _r5, _r6, _r7);
                transpose4x4_ps(_r8, _r9, _ra, _rb);

                vst1q_f32(pp, _r0);
                vst1q_f32(pp + 4, _r4);
                vst1q_f32(pp + 4 * 2, _r8);
                vst1q_f32(pp + 4 * 3, _r1);
                vst1q_f32(pp + 4 * 4, _r5);
                vst1q_f32(pp + 4 * 5, _r9);
                vst1q_f32(pp + 4 * 6, _r2);
                vst1q_f32(pp + 4 * 7, _r6);
                vst1q_f32(pp + 4 * 8, _ra);
                vst1q_f32(pp + 4 * 9, _r3);
                vst1q_f32(pp + 4 * 10, _r7);
                vst1q_f32(pp + 4 * 11, _rb);
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
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k * 4;
            const float* p1 = (const float*)B + (j + jj + 4) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                vst1q_f32(pp + 4, vld1q_f32(p1));
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
                float32x4_t _r0 = vld1q_f32(p0);
                float32x4_t _r1 = vld1q_f32(p1);
                float32x4_t _r2 = vld1q_f32(p2);
                float32x4_t _r3 = vld1q_f32(p3);
                float32x4_t _r4 = vld1q_f32(p4);
                float32x4_t _r5 = vld1q_f32(p5);
                float32x4_t _r6 = vld1q_f32(p6);
                float32x4_t _r7 = vld1q_f32(p7);

                transpose4x4_ps(_r0, _r1, _r2, _r3);
                transpose4x4_ps(_r4, _r5, _r6, _r7);

                vst1q_f32(pp, _r0);
                vst1q_f32(pp + 4, _r4);
                vst1q_f32(pp + 4 * 2, _r1);
                vst1q_f32(pp + 4 * 3, _r5);
                vst1q_f32(pp + 4 * 4, _r2);
                vst1q_f32(pp + 4 * 5, _r6);
                vst1q_f32(pp + 4 * 6, _r3);
                vst1q_f32(pp + 4 * 7, _r7);
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
                vst1q_f32(pp, vld1q_f32(p0));
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
                float32x4x4_t _r0123;
                _r0123.val[0] = vld1q_f32(p0);
                _r0123.val[1] = vld1q_f32(p1);
                _r0123.val[2] = vld1q_f32(p2);
                _r0123.val[3] = vld1q_f32(p3);
                vst4q_f32(pp, _r0123);
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
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
            const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

            int kk = 0;
#if __ARM_NEON
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x2_t _r01;
                _r01.val[0] = vld1q_f32(p0);
                _r01.val[1] = vld1q_f32(p1);
                vst2q_f32(pp, _r01);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
#endif // __ARM_NEON
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
#if __ARM_NEON
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                pp += 4;
                p0 += 4;
            }
#endif // __ARM_NEON
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
        }
    }
}

static void transpose_pack_B_tile_naive(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
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
                // float32x4x4_t _r0123 = vld4q_f32(p0);
                // float32x4x4_t _r4567 = vld4q_f32(p0 + 16);
                // float32x4x4_t _r89ab = vld4q_f32(p0 + 32);
                // vst1q_f32(pp, _r0123.val[0]);
                // vst1q_f32(pp + 4, _r4567.val[0]);
                // vst1q_f32(pp + 4 * 2, _r89ab.val[0]);
                // vst1q_f32(pp + 4 * 3, _r0123.val[1]);
                // vst1q_f32(pp + 4 * 4, _r4567.val[1]);
                // vst1q_f32(pp + 4 * 5, _r89ab.val[1]);
                // vst1q_f32(pp + 4 * 6, _r0123.val[2]);
                // vst1q_f32(pp + 4 * 7, _r4567.val[2]);
                // vst1q_f32(pp + 4 * 8, _r89ab.val[2]);
                // vst1q_f32(pp + 4 * 9, _r0123.val[3]);
                // vst1q_f32(pp + 4 * 10, _r4567.val[3]);
                // vst1q_f32(pp + 4 * 11, _r89ab.val[3]);
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
                float32x4x2_t _r01;
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

static void transpose_pack_B_tile(const Mat& B, Mat& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    float* pp = BT;

    int jj = 0;
#if __ARM_NEON
#if __aarch64__
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x4_t _r0123 = vld4q_f32(p0);
                float32x4x4_t _r4567 = vld4q_f32(p0 + 16);
                float32x4x4_t _r89ab = vld4q_f32(p0 + 32);
                vst1q_f32(pp, _r0123.val[0]);
                vst1q_f32(pp + 4, _r4567.val[0]);
                vst1q_f32(pp + 4 * 2, _r89ab.val[0]);
                vst1q_f32(pp + 4 * 3, _r0123.val[1]);
                vst1q_f32(pp + 4 * 4, _r4567.val[1]);
                vst1q_f32(pp + 4 * 5, _r89ab.val[1]);
                vst1q_f32(pp + 4 * 6, _r0123.val[2]);
                vst1q_f32(pp + 4 * 7, _r4567.val[2]);
                vst1q_f32(pp + 4 * 8, _r89ab.val[2]);
                vst1q_f32(pp + 4 * 9, _r0123.val[3]);
                vst1q_f32(pp + 4 * 10, _r4567.val[3]);
                vst1q_f32(pp + 4 * 11, _r89ab.val[3]);
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
                vst1q_f32(pp, vld1q_f32(p0));
                vst1q_f32(pp + 4, vld1q_f32(p0 + 4));
                vst1q_f32(pp + 8, vld1q_f32(p0 + 8));
                pp += 12;
                p0 += B_hstep;
            }
        }
    }
#endif // __aarch64__
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x4_t _r0123 = vld4q_f32(p0);
                float32x4x4_t _r4567 = vld4q_f32(p0 + 16);
                vst1q_f32(pp, _r0123.val[0]);
                vst1q_f32(pp + 4, _r4567.val[0]);
                vst1q_f32(pp + 4 * 2, _r0123.val[1]);
                vst1q_f32(pp + 4 * 3, _r4567.val[1]);
                vst1q_f32(pp + 4 * 4, _r0123.val[2]);
                vst1q_f32(pp + 4 * 5, _r4567.val[2]);
                vst1q_f32(pp + 4 * 6, _r0123.val[3]);
                vst1q_f32(pp + 4 * 7, _r4567.val[3]);
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
                vst1q_f32(pp, vld1q_f32(p0));
                vst1q_f32(pp + 4, vld1q_f32(p0 + 4));
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
                float32x4x4_t _r0123 = vld4q_f32(p0);
                vst1q_f32(pp, _r0123.val[0]);
                vst1q_f32(pp + 4, _r0123.val[1]);
                vst1q_f32(pp + 4 * 2, _r0123.val[2]);
                vst1q_f32(pp + 4 * 3, _r0123.val[3]);
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
                vst1q_f32(pp, vld1q_f32(p0));
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __ARM_NEON
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x2_t _r01;
                _r01.val[0] = vld1q_f32(p0);
                _r01.val[1] = vld1q_f32(p0 + 4);
                vst2q_f32(pp, _r01);
                pp += 8;
                p0 += B_hstep * 4;
            }
        }
#endif // __ARM_NEON
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
#if __ARM_NEON
        if (elempack == 4)
        {
            const float* p0 = (const float*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
#endif // __ARM_NEON
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
    Mat in;
    Mat out;
    Mat out_check;
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
    print_Mat(in);
    printf("------------------------\n");
    print_Mat(out);
    printf("------------------------\n");
    print_Mat(out_check);
    printf("------------------------\n");
#endif

    if (!check_Mat(out, out_check)) {
        printf("error\n");
    } else {
        printf("correct\n");
    }
    // check function transpose_pack_A_tile
    transpose_pack_B_tile(in, out, 0, h, 0, w);
    transpose_pack_B_tile_naive(in, out_check, 0, h, 0, w);
    // print_Mat(out);
    if (!check_Mat(out, out_check)) {
        printf("error\n");
    } else {
        printf("correct\n");
    }
    delete [] data_in, data_out;

    return 0;
}