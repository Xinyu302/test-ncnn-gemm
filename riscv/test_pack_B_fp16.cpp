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

static void pack_B_tile_bf16_fp16(const Mat<unsigned short>& B, Mat<unsigned short>& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __riscv_vector
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) / 8 * 8 * B_hstep + k * 8;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 8) / 8 * 8 * B_hstep + k * 8;

            if ((j + jj) % 8 == 0)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vl = 8;
                    vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                    vl = 4;
                    vse16_v_u16m1(pp + 8, vle16_v_u16m1(p1, vl), vl);
                    // vst1q_u16(pp, vle16_v_u16m1(p0), vl);
                    // vst1_u16(pp + 8, vld1_u16(p1));
                    pp += 12;
                    p0 += 8;
                    p1 += 8;
                }
            }
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vl = 4;
                    vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                    vl = 8;
                    vse16_v_u16m1(pp + 4, vle16_v_u16m1(p1, vl), vl);
                    // vst1_u16(pp, vld1_u16(p0 + 4));
                    // vst1q_u16(pp + 4, vle16_v_u16m1(p1), vl);
                    pp += 12;
                    p0 += 8;
                    p1 += 8;
                }
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k * 4;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 8) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vl = 4;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                vse16_v_u16m1(pp + 4, vle16_v_u16m1(p1, vl), vl);
                vse16_v_u16m1(pp + 8, vle16_v_u16m1(p2, vl), vl);
                // vst1_u16(pp, vld1_u16(p0));
                // vst1_u16(pp + 4, vld1_u16(p1));
                // vst1_u16(pp + 8, vld1_u16(p2));
                pp += 12;
                p0 += 4;
                p1 += 4;
                p2 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 2) * B_hstep + k;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 3) * B_hstep + k;
            const unsigned short* p4 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k;
            const unsigned short* p5 = (const unsigned short*)B + (j + jj + 5) * B_hstep + k;
            const unsigned short* p6 = (const unsigned short*)B + (j + jj + 6) * B_hstep + k;
            const unsigned short* p7 = (const unsigned short*)B + (j + jj + 7) * B_hstep + k;
            const unsigned short* p8 = (const unsigned short*)B + (j + jj + 8) * B_hstep + k;
            const unsigned short* p9 = (const unsigned short*)B + (j + jj + 9) * B_hstep + k;
            const unsigned short* pa = (const unsigned short*)B + (j + jj + 10) * B_hstep + k;
            const unsigned short* pb = (const unsigned short*)B + (j + jj + 11) * B_hstep + k;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p1, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(p2, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(p3, vl);
                vuint16m1_t _r4 = vle16_v_u16m1(p4, vl);
                vuint16m1_t _r5 = vle16_v_u16m1(p5, vl);
                vuint16m1_t _r6 = vle16_v_u16m1(p6, vl);
                vuint16m1_t _r7 = vle16_v_u16m1(p7, vl);
                vuint16m1_t _r8 = vle16_v_u16m1(p8, vl);
                vuint16m1_t _r9 = vle16_v_u16m1(p9, vl);
                vuint16m1_t _ra = vle16_v_u16m1(pa, vl);
                vuint16m1_t _rb = vle16_v_u16m1(pb, vl);

                transpose4x4_u16(_r0, _r1, _r2, _r3, vl);
                transpose4x4_u16(_r4, _r5, _r6, _r7, vl);
                transpose4x4_u16(_r8, _r9, _ra, _rb, vl);

                vse16_v_u16m1(pp, _r0, vl);
                vse16_v_u16m1(pp + 4, _r4, vl);
                vse16_v_u16m1(pp + 4 * 2, _r8, vl);
                vse16_v_u16m1(pp + 4 * 3, _r1, vl);
                vse16_v_u16m1(pp + 4 * 4, _r5, vl);
                vse16_v_u16m1(pp + 4 * 5, _r9, vl);
                vse16_v_u16m1(pp + 4 * 6, _r2, vl);
                vse16_v_u16m1(pp + 4 * 7, _r6, vl);
                vse16_v_u16m1(pp + 4 * 8, _ra, vl);
                vse16_v_u16m1(pp + 4 * 9, _r3, vl);
                vse16_v_u16m1(pp + 4 * 10, _r7, vl);
                vse16_v_u16m1(pp + 4 * 11, _rb, vl);

                // uint16x4_t _r0 = vld1_u16(p0);
                // uint16x4_t _r1 = vld1_u16(p1);
                // uint16x4_t _r2 = vld1_u16(p2);
                // uint16x4_t _r3 = vld1_u16(p3);
                // uint16x4_t _r4 = vld1_u16(p4);
                // uint16x4_t _r5 = vld1_u16(p5);
                // uint16x4_t _r6 = vld1_u16(p6);
                // uint16x4_t _r7 = vld1_u16(p7);
                // uint16x4_t _r8 = vld1_u16(p8);
                // uint16x4_t _r9 = vld1_u16(p9);
                // uint16x4_t _ra = vld1_u16(pa);
                // uint16x4_t _rb = vld1_u16(pb);

                // transpose4x4_u16(_r0, _r1, _r2, _r3);
                // transpose4x4_u16(_r4, _r5, _r6, _r7);
                // transpose4x4_u16(_r8, _r9, _ra, _rb);

                // vst1_u16(pp, _r0);
                // vst1_u16(pp + 4, _r4);
                // vst1_u16(pp + 4 * 2, _r8);
                // vst1_u16(pp + 4 * 3, _r1);
                // vst1_u16(pp + 4 * 4, _r5);
                // vst1_u16(pp + 4 * 5, _r9);
                // vst1_u16(pp + 4 * 6, _r2);
                // vst1_u16(pp + 4 * 7, _r6);
                // vst1_u16(pp + 4 * 8, _ra);
                // vst1_u16(pp + 4 * 9, _r3);
                // vst1_u16(pp + 4 * 10, _r7);
                // vst1_u16(pp + 4 * 11, _rb);
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
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) / 8 * 8 * B_hstep + k * 8;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 8) / 8 * 8 * B_hstep + k * 8;

            if ((j + jj) % 8 == 0)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vl = 8;
                    vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                    // vst1q_u16(pp, vle16_v_u16m1(p0), vl);
                    pp += 8;
                    p0 += 8;
                }
            }
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vl = 4;
                    vse16_v_u16m1(pp, vle16_v_u16m1(p0 + 4, vl), vl);
                    vse16_v_u16m1(pp + 4, vle16_v_u16m1(p1, vl), vl);
                    // vst1q_u16(pp, vcombine_u16(vld1_u16(p0 + 4), vld1_u16(p1)));
                    pp += 8;
                    p0 += 8;
                    p1 += 8;
                }
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vl = 4;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                vse16_v_u16m1(pp + 4, vle16_v_u16m1(p1, vl), vl);
                // vuint16m1_t _r0 = vcombine_u16(vld1_u16(p0), vld1_u16(p1));
                // vst1q_u16(pp, _r0);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 2) * B_hstep + k;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 3) * B_hstep + k;
            const unsigned short* p4 = (const unsigned short*)B + (j + jj + 4) * B_hstep + k;
            const unsigned short* p5 = (const unsigned short*)B + (j + jj + 5) * B_hstep + k;
            const unsigned short* p6 = (const unsigned short*)B + (j + jj + 6) * B_hstep + k;
            const unsigned short* p7 = (const unsigned short*)B + (j + jj + 7) * B_hstep + k;

            int kk = 0;

            int n = max_kk;
            while (n > 0) {
                vl = vsetvl_e16m1(n);
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p1, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(p2, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(p3, vl);
                vuint16m1_t _r4 = vle16_v_u16m1(p4, vl);
                vuint16m1_t _r5 = vle16_v_u16m1(p5, vl);
                vuint16m1_t _r6 = vle16_v_u16m1(p6, vl);
                vuint16m1_t _r7 = vle16_v_u16m1(p7, vl);
                vsseg8e16_v_u16m1(pp, _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);

                pp += 8 * vl;
                p0 += vl;
                p1 += vl;
                p2 += vl;
                p3 += vl;
                p4 += vl;
                p5 += vl;
                p6 += vl;
                p7 += vl;

                n -= vl;
            }
            // for (; kk + 7 < max_kk; kk += 8)
            // {
            //     vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
            //     vuint16m1_t _r1 = vle16_v_u16m1(p1, vl);
            //     vuint16m1_t _r2 = vle16_v_u16m1(p2, vl);
            //     vuint16m1_t _r3 = vle16_v_u16m1(p3, vl);
            //     vuint16m1_t _r4 = vle16_v_u16m1(p4, vl);
            //     vuint16m1_t _r5 = vle16_v_u16m1(p5, vl);
            //     vuint16m1_t _r6 = vle16_v_u16m1(p6, vl);
            //     vuint16m1_t _r7 = vle16_v_u16m1(p7, vl);
            //     transpose8x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7);
            //     vst1q_u16(pp, _r0);
            //     vst1q_u16(pp + 8, _r1);
            //     vst1q_u16(pp + 8 * 2, _r2);
            //     vst1q_u16(pp + 8 * 3, _r3);
            //     vst1q_u16(pp + 8 * 4, _r4);
            //     vst1q_u16(pp + 8 * 5, _r5);
            //     vst1q_u16(pp + 8 * 6, _r6);
            //     vst1q_u16(pp + 8 * 7, _r7);
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
            // for (; kk + 3 < max_kk; kk += 4)
            // {
            //     uint16x4_t _r0 = vld1_u16(p0);
            //     uint16x4_t _r1 = vld1_u16(p1);
            //     uint16x4_t _r2 = vld1_u16(p2);
            //     uint16x4_t _r3 = vld1_u16(p3);
            //     uint16x4_t _r4 = vld1_u16(p4);
            //     uint16x4_t _r5 = vld1_u16(p5);
            //     uint16x4_t _r6 = vld1_u16(p6);
            //     uint16x4_t _r7 = vld1_u16(p7);

            //     transpose4x4_u16(_r0, _r1, _r2, _r3);
            //     transpose4x4_u16(_r4, _r5, _r6, _r7);

            //     vst1_u16(pp, _r0);
            //     vst1_u16(pp + 4, _r4);
            //     vst1_u16(pp + 4 * 2, _r1);
            //     vst1_u16(pp + 4 * 3, _r5);
            //     vst1_u16(pp + 4 * 4, _r2);
            //     vst1_u16(pp + 4 * 5, _r6);
            //     vst1_u16(pp + 4 * 6, _r3);
            //     vst1_u16(pp + 4 * 7, _r7);
            //     pp += 32;
            //     p0 += 4;
            //     p1 += 4;
            //     p2 += 4;
            //     p3 += 4;
            //     p4 += 4;
            //     p5 += 4;
            //     p6 += 4;
            //     p7 += 4;
            // }
            // for (; kk < max_kk; kk++)
            // {
            //     pp[0] = p0[0];
            //     pp[1] = p1[0];
            //     pp[2] = p2[0];
            //     pp[3] = p3[0];
            //     pp[4] = p4[0];
            //     pp[5] = p5[0];
            //     pp[6] = p6[0];
            //     pp[7] = p7[0];
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
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) / 8 * 8 * B_hstep + k * 8;

            if ((j + jj) % 8 == 0)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vl = 8;
                    vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                    // vst1_u16(pp, vld1_u16(p0));
                    pp += 4;
                    p0 += 8;
                }
            }
            if ((j + jj) % 8 == 4)
            {
                for (int kk = 0; kk < max_kk; kk++)
                {
                    vl = 4;
                    vse16_v_u16m1(pp, vle16_v_u16m1(p0 + 4, vl), vl);
                    // vst1_u16(pp, vld1_u16(p0 + 4));
                    pp += 4;
                    p0 += 8;
                }
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k * 4;

            int kk = 0;
            for (; kk + 1 < max_kk; kk += 2)
            {
                vl = 8;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                // vst1q_u16(pp, vle16_v_u16m1(p0), vl);
                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                vl = 4;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                // vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;
            const unsigned short* p2 = (const unsigned short*)B + (j + jj + 2) * B_hstep + k;
            const unsigned short* p3 = (const unsigned short*)B + (j + jj + 3) * B_hstep + k;

            int kk = 0;
            
            int n = max_kk;
            while (n > 0) {
                vl = vsetvl_e16m1(n);
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p1, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(p2, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(p3, vl);
                vsseg4e16_v_u16m1(pp, _r0, _r1, _r2, _r3, vl);
                pp += 4 * vl;
                p0 += vl;
                p1 += vl;
                p2 += vl;
                p3 += vl;
                n -= vl;
            }
            // for (; kk + 7 < max_kk; kk += 8)
            // {
            //     uint16x8x4_t _r0123;
            //     _r0123.val[0] = vle16_v_u16m1(p0, vl);
            //     _r0123.val[1] = vle16_v_u16m1(p1, vl);
            //     _r0123.val[2] = vle16_v_u16m1(p2, vl);
            //     _r0123.val[3] = vle16_v_u16m1(p3, vl);
            //     vst4q_u16(pp, _r0123);
            //     pp += 32;
            //     p0 += 8;
            //     p1 += 8;
            //     p2 += 8;
            //     p3 += 8;
            // }
            // for (; kk + 3 < max_kk; kk += 4)
            // {
            //     uint16x4x4_t _r0123;
            //     _r0123.val[0] = vld1_u16(p0);
            //     _r0123.val[1] = vld1_u16(p1);
            //     _r0123.val[2] = vld1_u16(p2);
            //     _r0123.val[3] = vld1_u16(p3);
            //     vst4_u16(pp, _r0123);
            //     pp += 16;
            //     p0 += 4;
            //     p1 += 4;
            //     p2 += 4;
            //     p3 += 4;
            // }
            // for (; kk < max_kk; kk++)
            // {
            //     pp[0] = p0[0];
            //     pp[1] = p1[0];
            //     pp[2] = p2[0];
            //     pp[3] = p3[0];
            //     pp += 4;
            //     p0++;
            //     p1++;
            //     p2++;
            //     p3++;
            // }
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;
            const unsigned short* p1 = (const unsigned short*)B + (j + jj + 1) * B_hstep + k;

            int kk = 0;
#if __riscv_vector
            
            int n = max_kk;
            while (n > 0) {
                vl = vsetvl_e16m1(n);
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p1, vl);
                vsseg2e16_v_u16m1(pp, _r0, _r1, vl);
                pp += 2 * vl;
                p0 += vl;
                p1 += vl;
                n -= vl;
            }
            // for (; kk + 7 < max_kk; kk += 8)
            // {
            //     uint16x8x2_t _r01;
            //     _r01.val[0] = vle16_v_u16m1(p0, vl);
            //     _r01.val[1] = vle16_v_u16m1(p1, vl);
            //     vst2q_u16(pp, _r01);
            //     pp += 16;
            //     p0 += 8;
            //     p1 += 8;
            // }
            // for (; kk + 3 < max_kk; kk += 4)
            // {
            //     uint16x4x2_t _r01;
            //     _r01.val[0] = vld1_u16(p0);
            //     _r01.val[1] = vld1_u16(p1);
            //     vst2_u16(pp, _r01);
            //     pp += 8;
            //     p0 += 4;
            //     p1 += 4;
            // }
#else
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp[1] = p1[0];
                pp += 2;
                p0++;
                p1++;
            }
#endif // __riscv_vector
        }
    }
    for (; jj < max_jj; jj += 1)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + (j + jj) * B_hstep + k;

            int kk = 0;
#if __riscv_vector
            
            int n = max_kk;
            while (n > 0) {
                vl = vsetvl_e16m1(n);
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vse16_v_u16m1(pp, _r0, vl);
                pp += vl;
                p0 += vl;
                n -= vl;
            }
            // for (; kk + 7 < max_kk; kk += 8)
            // {
            //     vst1q_u16(pp, vle16_v_u16m1(p0), vl);
            //     pp += 8;
            //     p0 += 8;
            // }
            // for (; kk + 3 < max_kk; kk += 4)
            // {
            //     vst1_u16(pp, vld1_u16(p0));
            //     pp += 4;
            //     p0 += 4;
            // }
#else
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0++;
            }
#endif // __riscv_vector
        }
    }
}

static void transpose_pack_B_tile_bf16_fp16(const Mat<unsigned short>& B, Mat<unsigned short>& BT, int j, int max_jj, int k, int max_kk)
{
    const int elempack = B.elempack;
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    unsigned short* pp = BT;

    int jj = 0;
#if __riscv_vector
    for (; jj + 11 < max_jj; jj += 12)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;

            for (; kk + 7 < max_kk; kk += 8)
            {

                vl = 8;
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p0 + 8, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(p0 + 16, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(p0 + 24, vl);
                vuint16m1_t _r4 = vle16_v_u16m1(p0 + 32, vl);
                vuint16m1_t _r5 = vle16_v_u16m1(p0 + 40, vl);
                vuint16m1_t _r6 = vle16_v_u16m1(p0 + 48, vl);
                vuint16m1_t _r7 = vle16_v_u16m1(p0 + 56, vl);
                vuint16m1_t _r8 = vle16_v_u16m1(p0 + 64, vl);
                vuint16m1_t _r9 = vle16_v_u16m1(p0 + 72, vl);
                vuint16m1_t _ra = vle16_v_u16m1(p0 + 80, vl);
                vuint16m1_t _rb = vle16_v_u16m1(p0 + 88, vl);

                vsse16_v_u16m1(pp, 12 * sizeof(unsigned short), _r0, vl);
                vsse16_v_u16m1(pp + 1, 12 * sizeof(unsigned short), _r1, vl);
                vsse16_v_u16m1(pp + 2, 12 * sizeof(unsigned short), _r2, vl);
                vsse16_v_u16m1(pp + 3, 12 * sizeof(unsigned short), _r3, vl);
                vsse16_v_u16m1(pp + 4, 12 * sizeof(unsigned short), _r4, vl);
                vsse16_v_u16m1(pp + 5, 12 * sizeof(unsigned short), _r5, vl);
                vsse16_v_u16m1(pp + 6, 12 * sizeof(unsigned short), _r6, vl);
                vsse16_v_u16m1(pp + 7, 12 * sizeof(unsigned short), _r7, vl);
                vsse16_v_u16m1(pp + 8, 12 * sizeof(unsigned short), _r8, vl);
                vsse16_v_u16m1(pp + 9, 12 * sizeof(unsigned short), _r9, vl);
                vsse16_v_u16m1(pp + 10, 12 * sizeof(unsigned short), _ra, vl);
                vsse16_v_u16m1(pp + 11, 12 * sizeof(unsigned short), _rb, vl); 


                // vl = 4;
                // vuint16m1_t _r80 = vle16_v_u16m1(p0 + 64, vl);
                // vuint16m1_t _r81 = vle16_v_u16m1(p0 + 68, vl);
                // vuint16m1_t _r90 = vle16_v_u16m1(p0 + 72, vl);
                // vuint16m1_t _r91 = vle16_v_u16m1(p0 + 76, vl);
                // vuint16m1_t _ra0 = vle16_v_u16m1(p0 + 80, vl);
                // vuint16m1_t _ra1 = vle16_v_u16m1(p0 + 84, vl);
                // vuint16m1_t _rb0 = vle16_v_u16m1(p0 + 88, vl);
                // vuint16m1_t _rb1 = vle16_v_u16m1(p0 + 92, vl);
                // // vuint16m1_t _r9 = vle16_v_u16m1(p0 + 72, vl);
                // // vuint16m1_t _ra = vle16_v_u16m1(p0 + 80, vl);
                // // vuint16m1_t _rb = vle16_v_u16m1(p0 + 88, vl);
                // vl = 8;
                // transpose8x8_u16(_r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);
                // vl = 4;
                // transpose4x4_u16(_r80, _r90, _ra0, _rb0, vl);
                // transpose4x4_u16(_r81, _r91, _ra1, _rb1, vl);

                // vl = 8;
                // vse16_v_u16m1(pp, _r0, vl);
                // vl = 4;
                // vse16_v_u16m1(pp + 8, _r80, vl);
                // vl = 8;
                // vse16_v_u16m1(pp + 12, _r1, vl);
                // vl = 4;
                // vse16_v_u16m1(pp + 20, _r81, vl);
                // vl = 8;
                // vse16_v_u16m1(pp + 24, _r2, vl);
                // vl = 4;
                // vse16_v_u16m1(pp + 32, _r90, vl);
                // vl = 8;
                // vse16_v_u16m1(pp + 36, _r3, vl);
                // vl = 4;
                // vse16_v_u16m1(pp + 44, _r91, vl);
                // vl = 8;
                // vse16_v_u16m1(pp + 48, _r4, vl);
                // vl = 4;
                // vse16_v_u16m1(pp + 56, _ra0, vl);
                // vl = 8;
                // vse16_v_u16m1(pp + 60, _r5, vl);
                // vl = 4;
                // vse16_v_u16m1(pp + 68, _ra1, vl);
                // vl = 8;
                // vse16_v_u16m1(pp + 72, _r6, vl);
                // vl = 4;
                // vse16_v_u16m1(pp + 80, _rb0, vl);
                // vl = 8;
                // vse16_v_u16m1(pp + 84, _r7, vl);
                // vl = 4;
                // vse16_v_u16m1(pp + 92, _rb1, vl);

                // vuint16m1_t _r0;
                // vuint16m1_t _r1;
                // vuint16m1_t _r2;
                // vuint16m1_t _r3;
                // vuint16m1_t _r4;
                // vuint16m1_t _r5;
                // vuint16m1_t _r6;
                // vuint16m1_t _r7;
                // vuint16m1_t _r8;
                // vuint16m1_t _r9;
                // vuint16m1_t _ra;
                // vuint16m1_t _rb;

                // vlseg4e16_v_f16m1(&_r0, &_r1, &_r2, &_r3, p0, vl);
                // vlseg4e16_v_f16m1(&_r4, &_r5, &_r6, &_r7, p0 + 32, vl);
                // vlseg4e16_v_f16m1(&_r8, &_r9, &_ra, &_rb, p0 + 64, vl);



                // uint16x8x4_t _r0123 = vld4q_u16(p0);
                // uint16x8x4_t _r4567 = vld4q_u16(p0 + 32);
                // uint16x8x4_t _r89ab = vld4q_u16(p0 + 64);
                // uint16x8x2_t _r04 = vuzpq_u16(_r0123.val[0], _r4567.val[0]);
                // uint16x8x2_t _r15 = vuzpq_u16(_r0123.val[1], _r4567.val[1]);
                // uint16x8x2_t _r26 = vuzpq_u16(_r0123.val[2], _r4567.val[2]);
                // uint16x8x2_t _r37 = vuzpq_u16(_r0123.val[3], _r4567.val[3]);
                // uint16x4x2_t _r04_1 = vuzp_u16(vget_low_u16(_r89ab.val[0]), vget_high_u16(_r89ab.val[0]));
                // uint16x4x2_t _r15_1 = vuzp_u16(vget_low_u16(_r89ab.val[1]), vget_high_u16(_r89ab.val[1]));
                // uint16x4x2_t _r26_1 = vuzp_u16(vget_low_u16(_r89ab.val[2]), vget_high_u16(_r89ab.val[2]));
                // uint16x4x2_t _r37_1 = vuzp_u16(vget_low_u16(_r89ab.val[3]), vget_high_u16(_r89ab.val[3]));
                // vst1q_u16(pp, _r04.val[0]);
                // vst1_u16(pp + 8, _r04_1.val[0]);
                // vst1q_u16(pp + 12, _r15.val[0]);
                // vst1_u16(pp + 20, _r15_1.val[0]);
                // vst1q_u16(pp + 24, _r26.val[0]);
                // vst1_u16(pp + 32, _r26_1.val[0]);
                // vst1q_u16(pp + 36, _r37.val[0]);
                // vst1_u16(pp + 44, _r37_1.val[0]);
                // vst1q_u16(pp + 48, _r04.val[1]);
                // vst1_u16(pp + 56, _r04_1.val[1]);
                // vst1q_u16(pp + 60, _r15.val[1]);
                // vst1_u16(pp + 68, _r15_1.val[1]);
                // vst1q_u16(pp + 72, _r26.val[1]);
                // vst1_u16(pp + 80, _r26_1.val[1]);
                // vst1q_u16(pp + 84, _r37.val[1]);
                // vst1_u16(pp + 92, _r37_1.val[1]);
                pp += 96;
                p0 += B_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // vl = 8;
                // vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                // vuint16m1_t _r1 = vle16_v_u16m1(p0 + 8, vl);
                // vuint16m1_t _r2 = vle16_v_u16m1(p0 + 16, vl);
                // vuint16m1_t _r3 = vle16_v_u16m1(p0 + 24, vl);
                // vl = 4;
                // vuint16m1_t _r8 = vle16_v_u16m1(p0 + 32, vl);
                // vuint16m1_t _r9 = vle16_v_u16m1(p0 + 36, vl);
                // vuint16m1_t _ra = vle16_v_u16m1(p0 + 40, vl);
                // vuint16m1_t _rb = vle16_v_u16m1(p0 + 44, vl);
                
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p0 + 8, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(p0 + 16, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(p0 + 24, vl);
                vuint16m1_t _r4 = vle16_v_u16m1(p0 + 32, vl);
                vuint16m1_t _r5 = vle16_v_u16m1(p0 + 40, vl);
                vuint16m1_t _r6 = vle16_v_u16m1(p0 + 48, vl);
                vuint16m1_t _r7 = vle16_v_u16m1(p0 + 56, vl);
                vuint16m1_t _r8 = vle16_v_u16m1(p0 + 64, vl);
                vuint16m1_t _r9 = vle16_v_u16m1(p0 + 72, vl);
                vuint16m1_t _ra = vle16_v_u16m1(p0 + 80, vl);
                vuint16m1_t _rb = vle16_v_u16m1(p0 + 88, vl);

                vsse16_v_u16m1(pp, 12 * sizeof(unsigned short), _r0, vl);
                vsse16_v_u16m1(pp + 1, 12 * sizeof(unsigned short), _r1, vl);
                vsse16_v_u16m1(pp + 2, 12 * sizeof(unsigned short), _r2, vl);
                vsse16_v_u16m1(pp + 3, 12 * sizeof(unsigned short), _r3, vl);
                vsse16_v_u16m1(pp + 4, 12 * sizeof(unsigned short), _r4, vl);
                vsse16_v_u16m1(pp + 5, 12 * sizeof(unsigned short), _r5, vl);
                vsse16_v_u16m1(pp + 6, 12 * sizeof(unsigned short), _r6, vl);
                vsse16_v_u16m1(pp + 7, 12 * sizeof(unsigned short), _r7, vl);
                vsse16_v_u16m1(pp + 8, 12 * sizeof(unsigned short), _r8, vl);
                vsse16_v_u16m1(pp + 9, 12 * sizeof(unsigned short), _r9, vl);
                vsse16_v_u16m1(pp + 10, 12 * sizeof(unsigned short), _ra, vl);
                vsse16_v_u16m1(pp + 11, 12 * sizeof(unsigned short), _rb, vl); 

                // uint16x8x4_t _r0123 = vld4q_u16(p0);
                // uint16x4x4_t _r89ab = vld4_u16(p0 + 32);
                // vst1q_u16(pp, _r0123.val[0]);
                // vst1_u16(pp + 8, _r89ab.val[0]);
                // vst1q_u16(pp + 12, _r0123.val[1]);
                // vst1_u16(pp + 20, _r89ab.val[1]);
                // vst1q_u16(pp + 24, _r0123.val[2]);
                // vst1_u16(pp + 32, _r89ab.val[2]);
                // vst1q_u16(pp + 36, _r0123.val[3]);
                // vst1_u16(pp + 44, _r89ab.val[3]);
                pp += 48;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vl = 12;
                vse16_v_u16m2(pp, vle16_v_u16m2(p0, vl), vl);
                // vst1q_u16(pp, vle16_v_u16m1(p0), vl);
                // vst1_u16(pp + 8, vld1_u16(p0 + 8));
                pp += 12;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 7 < max_jj; jj += 8)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p0 + 8, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(p0 + 16, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(p0 + 24, vl);
                vuint16m1_t _r4 = vle16_v_u16m1(p0 + 32, vl);
                vuint16m1_t _r5 = vle16_v_u16m1(p0 + 40, vl);
                vuint16m1_t _r6 = vle16_v_u16m1(p0 + 48, vl);
                vuint16m1_t _r7 = vle16_v_u16m1(p0 + 56, vl);
                vsseg8e16_v_u16m1(pp, _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);
                // uint16x8x4_t _r0123 = vld4q_u16(p0);
                // uint16x8x4_t _r4567 = vld4q_u16(p0 + 32);
                // uint16x8x2_t _r04 = vuzpq_u16(_r0123.val[0], _r4567.val[0]);
                // uint16x8x2_t _r15 = vuzpq_u16(_r0123.val[1], _r4567.val[1]);
                // uint16x8x2_t _r26 = vuzpq_u16(_r0123.val[2], _r4567.val[2]);
                // uint16x8x2_t _r37 = vuzpq_u16(_r0123.val[3], _r4567.val[3]);
                // vst1q_u16(pp, _r04.val[0]);
                // vst1q_u16(pp + 8, _r15.val[0]);
                // vst1q_u16(pp + 16, _r26.val[0]);
                // vst1q_u16(pp + 24, _r37.val[0]);
                // vst1q_u16(pp + 32, _r04.val[1]);
                // vst1q_u16(pp + 40, _r15.val[1]);
                // vst1q_u16(pp + 48, _r26.val[1]);
                // vst1q_u16(pp + 56, _r37.val[1]);
                pp += 64;
                p0 += B_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vl = 8;
                vuint16m1_t _r0;
                vuint16m1_t _r1;
                vuint16m1_t _r2;
                vuint16m1_t _r3;

                vlseg4e16_v_u16m1(&_r0, &_r1, &_r2, &_r3, p0, vl);

                vse16_v_u16m1(pp, _r0, vl);
                vse16_v_u16m1(pp + 8, _r1, vl);
                vse16_v_u16m1(pp + 16, _r2, vl);
                vse16_v_u16m1(pp + 24, _r3, vl);

                // uint16x8x4_t _r0123 = vld4q_u16(p0);
                // vst1q_u16(pp, _r0123.val[0]);
                // vst1q_u16(pp + 8, _r0123.val[1]);
                // vst1q_u16(pp + 16, _r0123.val[2]);
                // vst1q_u16(pp + 24, _r0123.val[3]);
                pp += 32;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                // vst1q_u16(pp, vle16_v_u16m1(p0), vl);
                pp += 8;
                p0 += B_hstep;
            }
        }
    }
    for (; jj + 3 < max_jj; jj += 4)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p0 + 8, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(p0 + 16, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(p0 + 24, vl);

                vsseg4e16_v_u16m1(pp, _r0, _r1, _r2, _r3, vl);
                // uint16x8x4_t _r0123;
                // _r0123.val[0] = vle16_v_u16m1(p0, vl);
                // _r0123.val[1] = vle16_v_u16m1(p0 + 8, vl);
                // _r0123.val[2] = vle16_v_u16m1(p0 + 16, vl);
                // _r0123.val[3] = vle16_v_u16m1(p0 + 24, vl);
                // vst4q_u16(pp, _r0123);
                pp += 32;
                p0 += B_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vl = 4;
                vuint16m1_t _r0;
                vuint16m1_t _r1;
                vuint16m1_t _r2;
                vuint16m1_t _r3;

                vlseg4e16_v_u16m1(&_r0, &_r1, &_r2, &_r3, p0, vl);

                vse16_v_u16m1(pp, _r0, vl);
                vse16_v_u16m1(pp + 4, _r1, vl);
                vse16_v_u16m1(pp + 8, _r2, vl);
                vse16_v_u16m1(pp + 12, _r3, vl);
                // uint16x4x4_t _r0123 = vld4_u16(p0);
                // vst1q_u16(pp, vcombine_u16(_r0123.val[0], _r0123.val[1]));
                // vst1q_u16(pp + 8, vcombine_u16(_r0123.val[2], _r0123.val[3]));
                pp += 16;
                p0 += B_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vl = 4;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                // vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += B_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; jj + 1 < max_jj; jj += 2)
    {
#if __riscv_vector
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                vl = 8;
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p0 + 8, vl);
                vsseg2e16_v_u16m1(pp, _r0, _r1, vl);
                // uint16x8x2_t _r01;
                // _r01.val[0] = vle16_v_u16m1(p0, vl);
                // _r01.val[1] = vle16_v_u16m1(p0 + 8, vl);
                // vst2q_u16(pp, _r01);
                pp += 16;
                p0 += B_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(p0 + 8, vl);
                vsseg2e16_v_u16m1(pp, _r0, _r1, vl); 
                // uint16x4x2_t _r01;
                // _r01.val[0] = vld1_u16(p0);
                // _r01.val[1] = vld1_u16(p0 + 4);
                // vst2_u16(pp, _r01);
                pp += 8;
                p0 += B_hstep * 4;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

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
#if __riscv_vector
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                vl = 8;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                // vst1q_u16(pp, vle16_v_u16m1(p0), vl);
                pp += 8;
                p0 += B_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vl = 8;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                // vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += B_hstep * 4;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)B + k * B_hstep + (j + jj);

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
    Mat<unsigned short> in;
    Mat<unsigned short> out;
    Mat<unsigned short> out_check;
    in.w = w;
    in.h = h;
    out.w = w;
    out.h = h;
    out_check.w = w;
    out_check.h = h;

    unsigned short *data_in = new unsigned short[h*w];
    unsigned short *data_out = new unsigned short[h*w];
    in.data = data_in;
    out.data = data_out;
    // out_check.data = data_check;
    init_Mat(in);
    pack_B_tile_bf16_fp16(in, out, 0, h, 0, w);

#if PRINT_MAT
    printf("-----------Origin Matrix------------\n");
    print_Mat(in);
    printf("------------------------\n");
    printf("-----------pack_B_tile_fp32_to_fp16------------\n");
    print_Mat(out);
    printf("------------------------\n");
    // print_Mat(out_check);
    // printf("------------------------\n");
#endif


    // check function transpose_pack_A_tile
    transpose_pack_B_tile_bf16_fp16(in, out, 0, h, 0, w);
#if PRINT_MAT
    printf("------transpose_pack_B_tile_fp32_to_fp16------------\n");
    print_Mat(out);
#endif // PRINT_MAT

    delete [] data_in, data_out;

    return 0;
}