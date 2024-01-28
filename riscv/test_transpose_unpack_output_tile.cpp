#include <riscv_vector.h>
#include <stdio.h>
#include "utils.h"

static void transpose_unpack_output_tile(const Mat<>& topT, Mat<>& top_blob, int i, int max_ii, int j, int max_jj)
{
    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const float* pp = topT;

    int ii = 0;
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                // vfloat32m1x4_t _r0;
                // vfloat32m1x4_t _r1;
                // vget_f32m1x4_f32m1(_r0, 0) = vle32_v_f32m1(pp, VL);
                // vget_f32m1x4_f32m1(_r1, 0) = vle32_v_f32m1(pp + 4, VL);
                // vget_f32m1x4_f32m1(_r0, 1) = vle32_v_f32m1(pp + 8, VL);
                // vget_f32m1x4_f32m1(_r1, 1) = vle32_v_f32m1(pp + 12, VL);
                // vget_f32m1x4_f32m1(_r0, 2) = vle32_v_f32m1(pp + 16, VL);
                // vget_f32m1x4_f32m1(_r1, 2) = vle32_v_f32m1(pp + 20, VL);
                // vget_f32m1x4_f32m1(_r0, 3) = vle32_v_f32m1(pp + 24, VL);
                // vget_f32m1x4_f32m1(_r1, 3) = vle32_v_f32m1(pp + 28, VL);
                store_float_v4(vle32_v_f32m1(pp, VL), vle32_v_f32m1(pp + 8, VL), vle32_v_f32m1(pp + 16, VL), vle32_v_f32m1(pp + 24, VL), p0);
                store_float_v4(vle32_v_f32m1(pp + 4, VL), vle32_v_f32m1(pp + 12, VL), vle32_v_f32m1(pp + 20, VL), vle32_v_f32m1(pp + 28, VL), p0 + 4);
                // vst4q_f32(p0, _r0);
                // vst4q_f32(p0 + 16, _r1);
                pp += 32;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                vfloat32m1_t _r0 = vle32_v_f32m1(pp, VL);
                vfloat32m1_t _r1 = vle32_v_f32m1(pp + 4, VL);
                vse32_v_f32m1(p0, _r0, VL);
                vse32_v_f32m1(p0 + 4, _r1, VL);
                pp += 8;
                p0 += out_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                // vfloat32m1x4_t _r0123;
                // vget_f32m1x4_f32m1(_r0123, 0) = vle32_v_f32m1(pp, VL);
                // vget_f32m1x4_f32m1(_r0123, 1) = vle32_v_f32m1(pp + 4, VL);
                // vget_f32m1x4_f32m1(_r0123, 2) = vle32_v_f32m1(pp + 8, VL);
                // vget_f32m1x4_f32m1(_r0123, 3) = vle32_v_f32m1(pp + 12, VL);
                // vst4q_f32(p0, _r0123);
                store_float_v4(vle32_v_f32m1(pp, VL), vle32_v_f32m1(pp + 4, VL), vle32_v_f32m1(pp + 8, VL), vle32_v_f32m1(pp + 12, VL), p0);
                pp += 16;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                vfloat32m1_t _r0 = vle32_v_f32m1(pp, VL);
                vse32_v_f32m1(p0, _r0, VL);
                pp += 4;
                p0 += out_hstep;
            }
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

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
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

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
        if (out_elempack == 4)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                vfloat32m1_t _r0 = vle32_v_f32m1(pp, VL);
                vse32_v_f32m1(p0, _r0, VL);
                pp += 4;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            float* p0 = (float*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = pp[0];
                pp += 1;
                p0 += out_hstep;
            }
        }
    }
}