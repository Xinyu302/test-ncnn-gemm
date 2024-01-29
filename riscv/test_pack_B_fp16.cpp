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
                pp += 8;
                p0 += 8;
            }
            for (; kk < max_kk; kk++)
            {
                vl = 4;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
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