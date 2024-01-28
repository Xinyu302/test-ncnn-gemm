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



static void pack_A_tile_bf16_fp16(const Mat<unsigned short>& A, Mat<unsigned short>& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __riscv_vector
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 8;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vl = 8;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                // vst1q_u16(pp, vle16_v_u16m1(p0), vl);
                pp += 8;
                p0 += 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 4) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vl = 4;
                vuint16m1_t _r00 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r01 = vle16_v_u16m1(p1, vl);
                vsseg2e16_v_u16m1(pp, _r00, _r01, vl);
                // vuint16m1_t _r0 = vcombine_u16(vld1_u16(p0), vld1_u16(p1));
                // vst1q_u16(pp, _r0);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;
            const unsigned short* p2 = (const unsigned short*)A + (i + ii + 2) * A_hstep + k;
            const unsigned short* p3 = (const unsigned short*)A + (i + ii + 3) * A_hstep + k;
            const unsigned short* p4 = (const unsigned short*)A + (i + ii + 4) * A_hstep + k;
            const unsigned short* p5 = (const unsigned short*)A + (i + ii + 5) * A_hstep + k;
            const unsigned short* p6 = (const unsigned short*)A + (i + ii + 6) * A_hstep + k;
            const unsigned short* p7 = (const unsigned short*)A + (i + ii + 7) * A_hstep + k;

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
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k * 4;

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
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;
            const unsigned short* p2 = (const unsigned short*)A + (i + ii + 2) * A_hstep + k;
            const unsigned short* p3 = (const unsigned short*)A + (i + ii + 3) * A_hstep + k;

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
    for (; ii + 1 < max_ii; ii += 2)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;
            const unsigned short* p1 = (const unsigned short*)A + (i + ii + 1) * A_hstep + k;

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
                p0 += 1;
                p1 += 1;
            }
#endif // __riscv_vector
        }
    }
    for (; ii < max_ii; ii += 1)
    {
        // if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + (i + ii) * A_hstep + k;

            int kk = 0;
#if __riscv_vector
            int n = max_kk;
            while (n > 0) {
                vl = vsetvl_e16m1(n);
                vuint16m1_t _r0 = vle16_v_u16m1(p0, vl);
                vse16_v_u16m1(pp, _r0, vl);
                pp += 1 * vl;
                p0 += vl;
                n -= vl;
            }
#else
            for (; kk < max_kk; kk++)
            {
                pp[0] = p0[0];
                pp += 1;
                p0 += 1;
            }
#endif 
        }
    }
}

static void transpose_pack_A_tile_bf16_fp16(const Mat<unsigned short>& A, Mat<unsigned short>& AT, int i, int max_ii, int k, int max_kk)
{
    const int elempack = A.elempack;
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    unsigned short* pp = AT;

    int ii = 0;
#if __riscv_vector
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

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
                p0 += A_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

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
                vse16_v_u16m1(pp + 4, _r1, vl);
                vse16_v_u16m1(pp + 8, _r2, vl);
                vse16_v_u16m1(pp + 12, _r3, vl);
                // uint16x8x4_t _r0123 = vld4q_u16(p0);
                // vst1q_u16(pp, _r0123.val[0]);
                // vst1q_u16(pp + 8, _r0123.val[1]);
                // vst1q_u16(pp + 16, _r0123.val[2]);
                // vst1q_u16(pp + 24, _r0123.val[3]);
                pp += 32;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vl = 8;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                pp += 8;
                p0 += A_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

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
                p0 += A_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

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


                // vse16_v_u16m4(pp, _r0123, vl);
                // uint16x4x4_t _r0123 = vld4_u16(p0);
                // vst1q_u16(pp, vcombine_u16(_r0123.val[0], _r0123.val[1]));
                // vst1q_u16(pp + 8, vcombine_u16(_r0123.val[2], _r0123.val[3]));
                pp += 16;
                p0 += A_hstep * 4;
            }
        }
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vl = 4;
                // vst1_u16(pp, vld1_u16(p0));
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __riscv_vector
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

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
                p0 += A_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vl = 4;
                vuint16m1_t _r010 = vle16_v_u16m1(p0, vl);
                vuint16m1_t _r011 = vle16_v_u16m1(p0 + 4, vl);
                vsseg2e16_v_u16m1(pp, _r010, _r011, vl);
        
                // uint16x4x2_t _r01;
                // _r01.val[0] = vld1_u16(p0);
                // _r01.val[1] = vld1_u16(p0 + 4);
                // vst2_u16(pp, _r01);
                pp += 8;
                p0 += A_hstep * 4;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

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
#if __riscv_vector
        if (elempack == 8)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 8;

            int kk = 0;
            for (; kk + 7 < max_kk; kk += 8)
            {
                vl = 8;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                pp += 8;
                p0 += A_hstep * 8;
            }
        }
        if (elempack == 4)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vl = 4;
                vse16_v_u16m1(pp, vle16_v_u16m1(p0, vl), vl);
                // vst1_u16(pp, vld1_u16(p0));
                pp += 4;
                p0 += A_hstep * 4;
            }
        }
#endif // __riscv_vector
        if (elempack == 1)
        {
            const unsigned short* p0 = (const unsigned short*)A + k * A_hstep + (i + ii);

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
    printf("in main\n");
    int h = 15, w = 15;
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
    printf("before init mat\n");

    init_Mat(in);

    printf("after init mat\n");
    pack_A_tile_bf16_fp16(in, out, 0, h, 0, w);


#if PRINT_MAT
    printf("--------Origin Matrix--------\n");
    print_Mat(in);
    printf("--------Packed Matrix--------\n");
    print_Mat<unsigned short>(out);
#endif

    transpose_pack_A_tile_bf16_fp16(in, out, 0, h, 0, w);

#if PRINT_MAT

    printf("--------Origin Matrix--------\n");
    print_Mat(in);
    printf("--------Transpose Packed Matrix--------\n");
    print_Mat<unsigned short>(out);
#endif

}
