#include <riscv_vector.h>
#include <stdio.h>
#include "utils.h"

int vl = 8;


static void transpose_unpack_output_tile_fp32_to_fp16(const Mat<>& topT, Mat<__fp16>& top_blob, int i, int max_ii, int j, int max_jj)
{
    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const float* pp = topT;

    int ii = 0;
#if __riscv_vector
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (out_elempack == 4)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp, vl), vl);
                vfloat16m1_t _r1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp + 8, vl), vl);
                vfloat16m1_t _r2 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp + 16, vl), vl);
                vfloat16m1_t _r3 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp + 24, vl), vl);

                vsseg4e16_v_f16m1(p0, _r0, _r1, _r2, _r3, vl);
                // uint16x8x4_t _r0;
                // _r0.val[0] = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(pp, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(pp + 4, vl / 2)), vl / 2);
                // _r0.val[1] = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(pp + 8, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(pp + 12, vl / 2)), vl / 2);
                // _r0.val[2] = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(pp + 16, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(pp + 20, vl / 2)), vl / 2);
                // _r0.val[3] = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(pp + 24, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(pp + 28, vl / 2)), vl / 2);
                // vst4q_u16(p0, _r0);
                pp += 32;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp, vl), vl);
                vse16_v_f16m1(p0, _r0, vl);
                // uint16x8_t _r0 = vcombine_u16(vfncvt_f_f_w_f16mf2(vle32_v_f32m1(pp, vl / 2)), vfncvt_f_f_w_f16mf2(vld1q_f32(pp + 4, vl / 2)), vl / 2);
                // vst1q_u16(p0, _r0);
                pp += 8;
                p0 += out_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (out_elempack == 4)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp, vl / 2), vl / 2);
                vfloat16m1_t _r1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp + 4, vl / 2), vl / 2);
                vfloat16m1_t _r2 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp + 8, vl / 2), vl / 2);
                vfloat16m1_t _r3 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp + 12, vl / 2), vl / 2);

                vsseg4e16_v_f16m1(p0, _r0, _r1, _r2, _r3, vl / 2);
                // uint16x4x4_t _r0123;
                // _r0123.val[0] = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(pp, vl / 2), vl / 2);
                // _r0123.val[1] = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(pp + 4, vl / 2), vl / 2);
                // _r0123.val[2] = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(pp + 8, vl / 2), vl / 2);
                // _r0123.val[3] = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(pp + 12, vl / 2), vl / 2);
                // vst4_u16(p0, _r0123);
                pp += 16;
                p0 += out_hstep * 4;
            }
        }
        if (out_elempack == 1)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp, vl / 2), vl / 2);
                vse16_v_f16m1(p0, _r0, vl / 2);
                // uint16x4_t _r0 = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(pp, vl / 2), vl / 2);
                // vse16_v_f16mf2(p0, _r0, vl / 2);
                pp += 4;
                p0 += out_hstep;
            }
        }
    }
#endif // __riscv_vector
    for (; ii + 1 < max_ii; ii += 2)
    {
#if __riscv_vector
        if (out_elempack == 4)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                p0[0] = float32_to_float16(pp[0]);
                p0[1] = float32_to_float16(pp[2]);
                p0[2] = float32_to_float16(pp[4]);
                p0[3] = float32_to_float16(pp[6]);
                p0[4] = float32_to_float16(pp[1]);
                p0[5] = float32_to_float16(pp[3]);
                p0[6] = float32_to_float16(pp[5]);
                p0[7] = float32_to_float16(pp[7]);
                pp += 8;
                p0 += out_hstep * 4;
            }
        }
#endif // __riscv_vector
        if (out_elempack == 1)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = float32_to_float16(pp[0]);
                p0[1] = float32_to_float16(pp[1]);
                pp += 2;
                p0 += out_hstep;
            }
        }
    }
    for (; ii < max_ii; ii += 1)
    {
#if __riscv_vector
        if (out_elempack == 4)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii) * 4;

            for (int jj = 0; jj + 3 < max_jj; jj += 4)
            {
                vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pp, vl / 2), vl / 2);
                vse16_v_f16m1(p0, _r0, vl / 2);
                // uint16x4_t _r0 = vfncvt_f_f_w_f16mf2(vle32_v_f32m1(pp, vl / 2), vl / 2);
                // vse16_v_f16mf2(p0, _r0, vl / 2);
                pp += 4;
                p0 += out_hstep * 4;
            }
        }
#endif // __riscv_vector
        if (out_elempack == 1)
        {
            __fp16* p0 = (__fp16*)top_blob + j * out_hstep + (i + ii);

            for (int jj = 0; jj < max_jj; jj += 1)
            {
                p0[0] = float32_to_float16(pp[0]);
                pp += 1;
                p0 += out_hstep;
            }
        }
    }
}