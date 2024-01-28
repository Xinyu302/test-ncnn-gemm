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


static void pack_B_tile_fp32_to_fp16(const Mat<>& B, Mat<__fp16>& BT, int j, int max_jj, int k, int max_kk)
{
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    __fp16* pp = BT;

    int jj = 0;
#if __riscv_vector
    for (; jj + 11 < max_jj; jj += 12)
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
            vl = 4;
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);
            vfloat16m1_t _r1 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p1, vl), vl);
            vfloat16m1_t _r2 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p2, vl), vl);
            vfloat16m1_t _r3 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p3, vl), vl);
            vfloat16m1_t _r4 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p4, vl), vl);
            vfloat16m1_t _r5 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p5, vl), vl);
            vfloat16m1_t _r6 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p6, vl), vl);
            vfloat16m1_t _r7 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p7, vl), vl);
            vfloat16m1_t _r8 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p8, vl), vl);
            vfloat16m1_t _r9 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p9, vl), vl);
            vfloat16m1_t _ra = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pa, vl), vl);
            vfloat16m1_t _rb = vfncvt_f_f_w_f16m1(vle32_v_f32m2(pb, vl), vl);
            
            transpose4x4_f16(_r0, _r1, _r2, _r3, vl);
            transpose4x4_f16(_r4, _r5, _r6, _r7, vl);
            transpose4x4_f16(_r8, _r9, _ra, _rb, vl);


            vse16_v_f16m1(pp, _r0, vl);
            vse16_v_f16m1(pp + 4, _r4, vl);
            vse16_v_f16m1(pp + 4 * 2, _r8, vl);
            vse16_v_f16m1(pp + 4 * 3, _r1, vl);
            vse16_v_f16m1(pp + 4 * 4, _r5, vl);
            vse16_v_f16m1(pp + 4 * 5, _r9, vl);
            vse16_v_f16m1(pp + 4 * 6, _r2, vl);
            vse16_v_f16m1(pp + 4 * 7, _r6, vl);
            vse16_v_f16m1(pp + 4 * 8, _ra, vl);
            vse16_v_f16m1(pp + 4 * 9, _r3, vl);
            vse16_v_f16m1(pp + 4 * 10, _r7, vl);
            vse16_v_f16m1(pp + 4 * 11, _rb, vl);

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
            pp[0] = float32_to_float16(p0[0]);
            pp[1] = float32_to_float16(p1[0]);
            pp[2] = float32_to_float16(p2[0]);
            pp[3] = float32_to_float16(p3[0]);
            pp[4] = float32_to_float16(p4[0]);
            pp[5] = float32_to_float16(p5[0]);
            pp[6] = float32_to_float16(p6[0]);
            pp[7] = float32_to_float16(p7[0]);
            pp[8] = float32_to_float16(p8[0]);
            pp[9] = float32_to_float16(p9[0]);
            pp[10] = float32_to_float16(pa[0]);
            pp[11] = float32_to_float16(pb[0]);
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
#endif // __riscv_vector
    for (; jj + 7 < max_jj; jj += 8)
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

        int n = max_kk;
        while (n > 0)
        {
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
    for (; jj + 3 < max_jj; jj += 4)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;
        const float* p2 = (const float*)B + (j + jj + 2) * B_hstep + k;
        const float* p3 = (const float*)B + (j + jj + 3) * B_hstep + k;

        int kk = 0;

        int n = max_kk;
        
        while (n > 0)
        {
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
    for (; jj + 1 < max_jj; jj += 2)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;
        const float* p1 = (const float*)B + (j + jj + 1) * B_hstep + k;

        int kk = 0;

        int n = max_kk;
        while (n > 0)
        {
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
    for (; jj < max_jj; jj += 1)
    {
        const float* p0 = (const float*)B + (j + jj) * B_hstep + k;

        int kk = 0;
        
        int n = max_kk;
        while (n > 0)
        {
            vl = vsetvl_e32m2(n);
            vfloat16m1_t _r0 = vfncvt_f_f_w_f16m1(vle32_v_f32m2(p0, vl), vl);

            vse16_v_f16m1(pp, _r0, vl);

            pp += 1 * vl;
            p0 += vl;
            n -= vl;
        }
    }
}

static void transpose_pack_B_tile_fp32_to_fp16(const Mat<>& B, Mat<__fp16>& BT, int j, int max_jj, int k, int max_kk)
{
    const int B_hstep = B.dims == 3 ? (int)B.cstep : B.w;

    __fp16* pp = BT;

    int jj = 0;
    
    int len = 12;
    // len should be 12, 8, 4, 2, 1
    while (len > 0) {
        for (; jj + len - 1 < max_jj; jj += len)
        {
            vl = vsetvl_e32m4(len);
            const float* p0 = (const float*)B + k * B_hstep + (j + jj);

            int kk = 0;
            for (; kk < max_kk; kk++)
            {
                vfloat16m2_t _r0 = vfncvt_f_f_w_f16m2(vle32_v_f32m4(p0, vl), vl);
                vse16_v_f16m2(pp, _r0, vl);
                pp += vl;
                p0 += B_hstep;
            }
        }
        if (len <= 4) {
            len /= 2;
        } else {
            len -= 4;
        }
    }
}

int main()
{
    int h = 27, w = 27;
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
    __fp16 *data_check = new __fp16[h*w];
    in.data = data_in;
    out.data = data_out;
    out_check.data = data_check;
    init_Mat(in);
    pack_B_tile_fp32_to_fp16(in, out, 0, h, 0, w);

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
    transpose_pack_B_tile_fp32_to_fp16(in, out, 0, h, 0, w);
#if PRINT_MAT
    printf("------transpose_pack_B_tile_fp32_to_fp16------------\n");
    print_Mat(out);
#endif // PRINT_MAT

    delete [] data_in, data_out;

    return 0;
}