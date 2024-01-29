#include <riscv_vector.h>
#include <stdio.h>
#include "utils.h"

int vl = 8;

static void transpose_unpack_output_tile_bf16_fp16(const Mat& topT, Mat& top_blob, int i, int max_ii, int j, int max_jj)
{
    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const unsigned short* pp = topT;

    int ii = 0;
#if __riscv_vector
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (out_elempack == 8)
        {
            unsigned short* p0 = (unsigned short*)top_blob + (j / 8 * 8) * out_hstep + (i + ii) * 8;

            int jj = 0;
            if (j % 8 == 4)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(pp + 4, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(pp + 8, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(pp + 12, vl);
                vuint16m1_t _r4 = vle16_v_u16m1(pp + 16, vl);
                vuint16m1_t _r5 = vle16_v_u16m1(pp + 20, vl);
                vuint16m1_t _r6 = vle16_v_u16m1(pp + 24, vl);
                vuint16m1_t _r7 = vle16_v_u16m1(pp + 28, vl);

                vsseg8e16_v_u16m1(p0 + 4, _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);

                pp += 32;
                p0 += out_hstep * 8;
                jj += 4;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                vl = 8;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(pp + 8, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(pp + 8 * 2, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(pp + 8 * 3, vl);
                vuint16m1_t _r4 = vle16_v_u16m1(pp + 8 * 4, vl);
                vuint16m1_t _r5 = vle16_v_u16m1(pp + 8 * 5, vl);
                vuint16m1_t _r6 = vle16_v_u16m1(pp + 8 * 6, vl);
                vuint16m1_t _r7 = vle16_v_u16m1(pp + 8 * 7, vl);

                vsseg8e16_v_u16m1(p0, _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);

                pp += 64;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(pp + 4, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(pp + 8, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(pp + 12, vl);
                vuint16m1_t _r4 = vle16_v_u16m1(pp + 16, vl);
                vuint16m1_t _r5 = vle16_v_u16m1(pp + 20, vl);
                vuint16m1_t _r6 = vle16_v_u16m1(pp + 24, vl);
                vuint16m1_t _r7 = vle16_v_u16m1(pp + 28, vl);

                vsseg8e16_v_u16m1(p0, _r0, _r1, _r2, _r3, _r4, _r5, _r6, _r7, vl);

                pp += 32;
                p0 += out_hstep * 8;
            }
        }
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                vl = 8;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(pp + 8, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(pp + 8 * 2, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(pp + 8 * 3, vl);

                vsseg4e16_v_u16m1(p0, _r0, _r1, _r2, _r3, vl);

                pp += 32;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                vl = 8;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vse16_v_u16(p0, _r0, vl);
                pp += 8;
                p0 += out_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (out_elempack == 8)
        {
            unsigned short* p0 = (unsigned short*)top_blob + (j / 8 * 8) * out_hstep + (i + ii) * 8;

            int jj = 0;
            if (j % 8 == 4)
            {
                vl = 4;
                vuint16m1_t _r0;
                vuint16m1_t _r1;
                vuint16m1_t _r2;
                vuint16m1_t _r3;

                vlseg4e16_v_u16m1(&_r0, &_r1, &_r2, &_r3, pp, vl);
                vse16_v_u16m1(p0 + 4, _r0, vl);
                vse16_v_u16m1(p0 + 8 + 4, _r1, vl);
                vse16_v_u16m1(p0 + 16 + 4, _r2, vl);
                vse16_v_u16m1(p0 + 24 + 4, _r3, vl);

                pp += 16;
                p0 += out_hstep * 8;
                jj += 4;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                vl = 8;
                vuint16m1_t _r0;
                vuint16m1_t _r1;
                vuint16m1_t _r2;
                vuint16m1_t _r3;

                vlseg4e16_v_u16m1(&_r0, &_r1, &_r2, &_r3, pp, vl);
                vse16_v_u16m1(p0, _r0, vl);
                vse16_v_u16m1(p0 + 8, _r1, vl);
                vse16_v_u16m1(p0 + 8 * 2, _r2, vl);
                vse16_v_u16m1(p0 + 8 * 3, _r3, vl);

                pp += 32;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                vl = 4;
                vuint16m1_t _r0;
                vuint16m1_t _r1;
                vuint16m1_t _r2;
                vuint16m1_t _r3;

                vlseg4e16_v_u16m1(&_r0, &_r1, &_r2, &_r3, pp, vl);
                vse16_v_u16m1(p0, _r0, vl);
                vse16_v_u16m1(p0 + 8, _r1, vl);
                vse16_v_u16m1(p0 + 16, _r2, vl);
                vse16_v_u16m1(p0 + 24, _r3, vl);

                pp += 16;
                p0 += out_hstep * 8;
            }
        }
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vuint16m1_t _r1 = vle16_v_u16m1(pp + 4, vl);
                vuint16m1_t _r2 = vle16_v_u16m1(pp + 8, vl);
                vuint16m1_t _r3 = vle16_v_u16m1(pp + 12, vl);

                vsseg4e16_v_u16m1(p0, _r0, _r1, _r2, _r3, vl);

                pp += 16;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vse16_v_u16m1(p0, _r0, vl);

                pp += 4;
                p0 += out_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __riscv_vector
        if (out_elempack == 8)
        {
            unsigned short* p0 = (unsigned short*)top_blob + (j / 8 * 8) * out_hstep + (i + ii) * 8;

            int jj = 0;
            if (j % 8 == 4)
            {
                p0[0 + 4] = pp[0];
                p0[1 + 4] = pp[2];
                p0[2 + 4] = pp[4];
                p0[3 + 4] = pp[6];
                p0[8 + 4] = pp[1];
                p0[9 + 4] = pp[3];
                p0[10 + 4] = pp[5];
                p0[11 + 4] = pp[7];
                pp += 8;
                p0 += out_hstep * 8;
                jj += 4;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                p0[0] = pp[0];
                p0[1] = pp[2];
                p0[2] = pp[4];
                p0[3] = pp[6];
                p0[4] = pp[8];
                p0[5] = pp[10];
                p0[6] = pp[12];
                p0[7] = pp[14];
                p0[8] = pp[1];
                p0[9] = pp[3];
                p0[10] = pp[5];
                p0[11] = pp[7];
                p0[12] = pp[9];
                p0[13] = pp[11];
                p0[14] = pp[13];
                p0[15] = pp[15];
                pp += 16;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                p0[0] = pp[0];
                p0[1] = pp[2];
                p0[2] = pp[4];
                p0[3] = pp[6];
                p0[8] = pp[1];
                p0[9] = pp[3];
                p0[10] = pp[5];
                p0[11] = pp[7];
                pp += 8;
                p0 += out_hstep * 8;
            }
        }
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                p0[0] = pp[0];
                p0[1] = pp[2];
                p0[2] = pp[4];
                p0[3] = pp[6];
                p0[4] = pp[1];
                p0[5] = pp[3];
                p0[6] = pp[5];
                p0[7] = pp[7];
                pp += 8;
                p0 += out_hstep * 4;
            }
        }
#endif // __riscv_vector
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = pp[0];
                p0[1] = pp[1];
                pp += 2;
                p0 += out_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
#if __riscv_vector
        if (out_elempack == 8)
        {
            unsigned short* p0 = (unsigned short*)top_blob + (j / 8 * 8) * out_hstep + (i + ii) * 8;

            int jj = 0;
            if (j % 8 == 4)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vse16_v_u16m1(p0 + 4, _r0, vl);

                pp += 4;
                p0 += out_hstep * 8;
                jj += 4;
            }
            for (; jj + 7 < max_jj; jj += 8)
            {
                vl = 8;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vse16_v_u16m1(p0, _r0, vl);
                pp += 8;
                p0 += out_hstep * 8;
            }
            for (; jj + 3 < max_jj; jj += 4)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vse16_v_u16m1(p0, _r0, vl);

                pp += 4;
                p0 += out_hstep * 8;
            }
        }
        if (out_elempack == 4)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                vl = 4;
                vuint16m1_t _r0 = vle16_v_u16m1(pp, vl);
                vse16_v_u16m1(p0, _r0, vl);

                pp += 4;
                p0 += out_hstep * 4;
            }
        }
#endif // __riscv_vector
        if (out_elempack == 1)
        {
            unsigned short* p0 = (unsigned short*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = pp[0];
                pp += 1;
                p0 += out_hstep;
            }
        }
    }
}