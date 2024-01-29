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
            p4 += vl;
            p5 += vl;
            p6 += vl;
            p7 += vl;
            n -= vl;
        }
    }
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
    }
#endif // __riscv_vector
}

static void transpose_pack_A_tile_fp32_to_fp16(const Mat<>& A, Mat<__fp16>& AT, int i, int max_ii, int k, int max_kk)
{
    const int A_hstep = A.dims == 3 ? (int)A.cstep : A.w;

    __fp16* pp = AT;

    int ii = 0;
#if __riscv_vector
    // vl = 8;
    for (; ii + 7 < max_ii; ii += 8) {
        vl = 8;
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vse16_v_f16m1(pp, _r0, vl);
            pp += vl;
            p0 += A_hstep;
        }
    }
    for (; ii + 3 < max_ii; ii += 4) {
        vl = 4;
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vse16_v_f16m1(pp, _r0, vl);
            pp += vl;
            p0 += A_hstep;
        }
    }

    for (; ii + 1 < max_ii; ii += 2) {
        vl = 2;
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vse16_v_f16m1(pp, _r0, vl);
            pp += vl;
            p0 += A_hstep;
        }
    }

    for (; ii < max_ii; ii += 1) {
        vl = 1;
        const float* p0 = (const float*)A + k * A_hstep + (i + ii);

        int kk = 0;
        for (; kk < max_kk; kk++)
        {
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vse16_v_f16m1(pp, _r0, vl);
            pp += vl;
            p0 += A_hstep;
        }
    }
#endif // __riscv_vector
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