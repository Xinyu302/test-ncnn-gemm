#include <arm_neon.h>
#include <stdio.h>
#include "utils.h"
// #undef __aarch64__
#define PRINT_MAT 0
#define PACK_8 1

#if PACK_8

#else
#undef __aarch64__    
#endif

// without arm neon
void pack_A_tile_naive(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    float* pp = AT;

    int ii = 0;
#if PACK_8
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;
            const float* p1 = (const float*)A + (i + ii + 4) * A_hstep + k * 4;

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

                p0 += 8;
                p1 += 4;
                pp += 4;
            }
        }
        else if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
            const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
            const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;
            const float* p4 = (const float*)A + (i + ii + 4) * A_hstep + k;
            const float* p5 = (const float*)A + (i + ii + 5) * A_hstep + k;
            const float* p6 = (const float*)A + (i + ii + 6) * A_hstep + k;
            const float* p7 = (const float*)A + (i + ii + 7) * A_hstep + k;

            for (int kk = 0; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];
                pp[4] = p4[0];
                pp[5] = p5[0];
                pp[6] = p6[0];
                pp[7] = p7[0];

                p0++;
                p1++;
                p2++;
                p3++;
                p4++;
                p5++;
                p6++;
                p7++;
                pp += 8;
            }
        }
    }
#endif // PACK_8
    for (; ii + 3 < max_ii; ii += 4) {
        if (elempack == 4) {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++) {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];

                p0 += 4;
                pp += 4;
            }
        } else if (elempack == 1) {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
            const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
            const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

            for (int kk = 0; kk < max_kk; kk++) {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp[2] = p2[0];
                pp[3] = p3[0];

                p0++;
                p1++;
                p2++;
                p3++;
                pp += 4;
            }
        }
    }
    for (; ii + 1 < max_ii; ii += 2) {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

        for (int kk = 0; kk < max_kk; kk++) {
            pp[0] = p0[0];
            pp[1] = p1[0];

            p0++;
            p1++;
            pp += 2;
        }
    }
    for (; ii < max_ii; ii += 1) {
        const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
        for (int kk = 0; kk < max_kk; kk++) {
            pp[0] = p0[0];
            p0++;
            pp++;
        }
    }

    
}


void pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    float* pp = AT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;
            const float* p1 = (const float*)A + (i + ii + 4) * A_hstep + k * 4;

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
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
            const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
            const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;
            const float* p4 = (const float*)A + (i + ii + 4) * A_hstep + k;
            const float* p5 = (const float*)A + (i + ii + 5) * A_hstep + k;
            const float* p6 = (const float*)A + (i + ii + 6) * A_hstep + k;
            const float* p7 = (const float*)A + (i + ii + 7) * A_hstep + k;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                float32x4_t _r0l = vld1q_f32(p0);
                float32x4_t _r0h = vld1q_f32(p0 + 4);
                float32x4_t _r1l = vld1q_f32(p1);
                float32x4_t _r1h = vld1q_f32(p1 + 4);
                float32x4_t _r2l = vld1q_f32(p2);
                float32x4_t _r2h = vld1q_f32(p2 + 4);
                float32x4_t _r3l = vld1q_f32(p3);
                float32x4_t _r3h = vld1q_f32(p3 + 4);
                float32x4_t _r4l = vld1q_f32(p4);
                float32x4_t _r4h = vld1q_f32(p4 + 4);
                float32x4_t _r5l = vld1q_f32(p5);
                float32x4_t _r5h = vld1q_f32(p5 + 4);
                float32x4_t _r6l = vld1q_f32(p6);
                float32x4_t _r6h = vld1q_f32(p6 + 4);
                float32x4_t _r7l = vld1q_f32(p7);
                float32x4_t _r7h = vld1q_f32(p7 + 4);
                transpose8x8_ps(_r0l, _r0h, _r1l, _r1h, _r2l, _r2h, _r3l, _r3h, _r4l, _r4h, _r5l, _r5h, _r6l, _r6h, _r7l, _r7h);
                vst1q_f32(pp, _r0l);
                vst1q_f32(pp + 4, _r0h);
                vst1q_f32(pp + 8, _r1l);
                vst1q_f32(pp + 12, _r1h);
                vst1q_f32(pp + 8 * 2, _r2l);
                vst1q_f32(pp + 8 * 2 + 4, _r2h);
                vst1q_f32(pp + 8 * 3, _r3l);
                vst1q_f32(pp + 8 * 3 + 4, _r3h);
                vst1q_f32(pp + 8 * 4, _r4l);
                vst1q_f32(pp + 8 * 4 + 4, _r4h);
                vst1q_f32(pp + 8 * 5, _r5l);
                vst1q_f32(pp + 8 * 5 + 4, _r5h);
                vst1q_f32(pp + 8 * 6, _r6l);
                vst1q_f32(pp + 8 * 6 + 4, _r6h);
                vst1q_f32(pp + 8 * 7, _r7l);
                vst1q_f32(pp + 8 * 7 + 4, _r7h);
                pp += 64;
                p0 += 8;
                p1 += 8;
                p2 += 8;
                p3 += 8;
                p4 += 8;
                p5 += 8;
                p6 += 8;
                p7 += 8;
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
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;
            const float* p2 = (const float*)A + (i + ii + 2) * A_hstep + k;
            const float* p3 = (const float*)A + (i + ii + 3) * A_hstep + k;

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
    for (; ii + 1 < max_ii; ii += 2)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

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
    for (; ii < max_ii; ii += 1)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;

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

static void transpose_pack_A_tile_naive(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk) {
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    float* pp = AT;

    int ii = 0;
#if PACK8
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // float32x4x4_t _r0123 = vld4q_f32(p0);
                // float32x4x4_t _r4567 = vld4q_f32(p0 + 16);
                // vst1q_f32(pp, _r0123.val[0]);
                // vst1q_f32(pp + 4, _r4567.val[0]);
                // vst1q_f32(pp + 4 * 2, _r0123.val[1]);
                // vst1q_f32(pp + 4 * 3, _r4567.val[1]);
                // vst1q_f32(pp + 4 * 4, _r0123.val[2]);
                // vst1q_f32(pp + 4 * 5, _r4567.val[2]);
                // vst1q_f32(pp + 4 * 6, _r0123.val[3]);
                // vst1q_f32(pp + 4 * 7, _r4567.val[3]);
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
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

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
                p0 += A_hstep;
            }
        }
    }
#endif // PACK8
    for (; ii + 3 < max_ii; ii += 4) {
        if (elempack == 4) {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4) {
                // float32x4x4_t _r0123 = vld4q_f32(p0);
                // float32x4x4_t _r4567 = vld4q_f32(p0 + 16);
                // vst1q_f32(pp, _r0123.val[0]);
                // vst1q_f32(pp + 4, _r4567.val[0]);
                // vst1q_f32(pp + 4 * 2, _r0123.val[1]);
                // vst1q_f32(pp + 4 * 3, _r4567.val[1]);
                for (int iter = 0; iter < 16; iter++) {
                    pp[iter] = p0[iter];
                }
                pp += 16;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1) {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++) {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp[2] = p0[2];
                pp[3] = p0[3];
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
    for (; ii + 1 < max_ii; ii += 2) {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x2_t _r01;
                for (int iter = 0; iter < 8; iter++) {
                    pp[iter] = p0[iter];
                }
                pp += 8;
                p0 += A_hstep * 4;
            }
        } 
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += A_hstep;
            }
        }
    }
    for (; ii < max_ii; ii++)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                for (int iter = 0; iter < 4; iter++) {
                    pp[iter] = p0[iter];
                }
                pp += 4;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += A_hstep;
            }
        }
    }
}

static void transpose_pack_A_tile(const Mat& A, Mat& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    float* pp = AT;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

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
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                vst1q_f32(pp + 4, vld1q_f32(p0 + 4));
                pp += 8;
                p0 += A_hstep;
            }
        }
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x4_t _r0123 = vld4q_f32(p0);
                vst1q_f32(pp, _r0123.val[0]);
                vst1q_f32(pp + 4, _r0123.val[1]);
                vst1q_f32(pp + 4 * 2, _r0123.val[2]);
                vst1q_f32(pp + 4 * 3, _r0123.val[3]);
                pp += 16;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __ARM_NEON
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4x2_t _r01;
                _r01.val[0] = vld1q_f32(p0);
                _r01.val[1] = vld1q_f32(p0 + 4);
                vst2q_f32(pp, _r01);
                pp += 8;
                p0 += A_hstep * 4;
            }
        }
#endif // __ARM_NEON
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p0[1];
                pp += 2;
                p0 += A_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
#if __ARM_NEON
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vst1q_f32(pp, vld1q_f32(p0));
                pp += 4;
                p0 += A_hstep * 4;
            }
        }
#endif // __ARM_NEON
        if (elempack == 1)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += A_hstep;
            }
        }
    }
}

int main() {
    int h = 15, w = 15;
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
    pack_A_tile(in, out, 0, h, 0, w);
    transpose_pack_A_tile(in, out, 0, h, 0, w);
    transpose_pack_A_tile_naive(in, out_check, 0, h, 0, w);

#if PRINT_MAT
    print_Mat(in);
    printf("------------------------\n");
    print_Mat(out);
    printf("------------------------\n");
    print_Mat(out);
    printf("------------------------\n");
#endif

    if (!check_Mat(out, out_check)) {
        printf("error\n");
    } else {
        printf("correct\n");
    }
    // check function transpose_pack_A_tile
    transpose_pack_A_tile(in, out, 0, h, 0, w);
    transpose_pack_A_tile_naive(in, out_check, 0, h, 0, w);
    // print_Mat(out);
    if (!check_Mat(out, out_check)) {
        printf("error\n");
    } else {
        printf("correct\n");
    }
    delete [] data_in, data_out;

    return 0;
}