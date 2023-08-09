#include <riscv_vector.h>
#include <stdio.h>
#include "utils.h"
#define PRINT_MAT 1
#define PACK_8 1

#if PACK_8

#else
#undef __aarch64__    
#endif

#define VL 4

vfloat32m1x2_t vzip_f32(vfloat32m1_t vector1, vfloat32m1_t vector2) {
    vfloat32m2_t d = vundefined_f32m2();
    d = vset_v_f32m1_f32m2(d, 0, vector1);
    d = vset_v_f32m1_f32m2(d, 1, vector2);
    vuint32m2_t index;
    uint32_t index_128[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    index = vle32_v_u32m2(index_128, 8);
    vfloat32m2_t g_d = vrgather_vv_f32m2(d, index, 8);
    vfloat32m1_t v0 = vget_v_f32m2_f32m1(g_d, 0);
    vfloat32m1_t v1 = vget_v_f32m2_f32m1(g_d, 1);
    return vcreate_f32m1x2(v0, v1);
}

vfloat32m1x4_t vzip_f32_v4(vfloat32m1_t vector1, vfloat32m1_t vector2, vfloat32m1_t vector3, vfloat32m1_t vector4) {
    vfloat32m4_t d = vundefined_f32m4();
    d = vset_v_f32m1_f32m4(d, 0, vector1);
    d = vset_v_f32m1_f32m4(d, 1, vector2);
    d = vset_v_f32m1_f32m4(d, 2, vector3);
    d = vset_v_f32m1_f32m4(d, 3, vector4);
    
    vuint32m4_t index;
    uint32_t index_128[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    index = vle32_v_u32m4(index_128, 16);
    vfloat32m4_t g_d = vrgather_vv_f32m4(d, index, 16);
    vfloat32m1_t v0 = vget_v_f32m4_f32m1(g_d, 0);
    vfloat32m1_t v1 = vget_v_f32m4_f32m1(g_d, 1);
    vfloat32m1_t v2 = vget_v_f32m4_f32m1(g_d, 2);
    vfloat32m1_t v3 = vget_v_f32m4_f32m1(g_d, 3);
    return vcreate_f32m1x4(v0, v1, v2, v3);
}

vfloat32m1_t vget_low_f32(vfloat32m1_t a) {
    return vmv_v_v_f32m1(a, 2);
}

vfloat32m1_t vget_high_f32(vfloat32m1_t a) 
{
    float t[4];
    vse32_v_f32m1(t, a, 4);
    return vle32_v_f32m1(t + 2, 2);
}

vfloat32m1_t vcombine_f32(vfloat32m1_t a, vfloat32m1_t b) {
    float t[4];
    vse32_v_f32m1(t, a, 2);
    vse32_v_f32m1(t + 2, b, 2);
    return vle32_v_f32m1(t, 4);
}

static inline void transpose8x8_ps(vfloat32m1_t& _r0l, vfloat32m1_t& _r0h,
                                   vfloat32m1_t& _r1l, vfloat32m1_t& _r1h,
                                   vfloat32m1_t& _r2l, vfloat32m1_t& _r2h,
                                   vfloat32m1_t& _r3l, vfloat32m1_t& _r3h,
                                   vfloat32m1_t& _r4l, vfloat32m1_t& _r4h,
                                   vfloat32m1_t& _r5l, vfloat32m1_t& _r5h,
                                   vfloat32m1_t& _r6l, vfloat32m1_t& _r6h,
                                   vfloat32m1_t& _r7l, vfloat32m1_t& _r7h)
{
    vfloat32m1x2_t _r01lz = vzip_f32(_r0l, _r1l);
    vfloat32m1x2_t _r23lz = vzip_f32(_r2l, _r3l);
    vfloat32m1x2_t _r01hz = vzip_f32(_r0h, _r1h);
    vfloat32m1x2_t _r23hz = vzip_f32(_r2h, _r3h);
    vfloat32m1x2_t _r45lz = vzip_f32(_r4l, _r5l);
    vfloat32m1x2_t _r67lz = vzip_f32(_r6l, _r7l);
    vfloat32m1x2_t _r45hz = vzip_f32(_r4h, _r5h);
    vfloat32m1x2_t _r67hz = vzip_f32(_r6h, _r7h); 
    _r0l = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r01lz, 0)), vget_low_f32(vget_f32m1x2_f32m1(_r23lz, 0)));
    _r0h = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r45lz, 0)), vget_low_f32(vget_f32m1x2_f32m1(_r67lz, 0)));
    _r1l = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r01lz, 0)), vget_high_f32(vget_f32m1x2_f32m1(_r23lz, 0)));
    _r1h = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r45lz, 0)), vget_high_f32(vget_f32m1x2_f32m1(_r67lz, 0)));
    _r2l = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r01lz, 1)), vget_low_f32(vget_f32m1x2_f32m1(_r23lz, 1)));
    _r2h = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r45lz, 1)), vget_low_f32(vget_f32m1x2_f32m1(_r67lz, 1)));
    _r3l = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r01lz, 1)), vget_high_f32(vget_f32m1x2_f32m1(_r23lz, 1)));
    _r3h = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r45lz, 1)), vget_high_f32(vget_f32m1x2_f32m1(_r67lz, 1)));
    _r4l = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r01hz, 0)), vget_low_f32(vget_f32m1x2_f32m1(_r23hz, 0)));
    _r4h = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r45hz, 0)), vget_low_f32(vget_f32m1x2_f32m1(_r67hz, 0)));
    _r5l = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r01hz, 0)), vget_high_f32(vget_f32m1x2_f32m1(_r23hz, 0)));
    _r5h = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r45hz, 0)), vget_high_f32(vget_f32m1x2_f32m1(_r67hz, 0)));
    _r6l = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r01hz, 1)), vget_low_f32(vget_f32m1x2_f32m1(_r23hz, 1)));
    _r6h = vcombine_f32(vget_low_f32(vget_f32m1x2_f32m1(_r45hz, 1)), vget_low_f32(vget_f32m1x2_f32m1(_r67hz, 1)));
    _r7l = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r01hz, 1)), vget_high_f32(vget_f32m1x2_f32m1(_r23hz, 1)));
    _r7h = vcombine_f32(vget_high_f32(vget_f32m1x2_f32m1(_r45hz, 1)), vget_high_f32(vget_f32m1x2_f32m1(_r67hz, 1)));

}

static inline void store_float32_v2(vfloat32m1_t vector1, vfloat32m1_t vector2, float *buf) 
{
    vfloat32m1x2_t zip = vzip_f32(vector1, vector2);
    vse32_v_f32m1(buf, vget_f32m1x2_f32m1(zip, 0), VL);
    vse32_v_f32m1(buf + 4, vget_f32m1x2_f32m1(zip, 1), VL);
}

static inline void store_float_v4(vfloat32m1_t vector1, vfloat32m1_t vector2, vfloat32m1_t vector3, vfloat32m1_t vector4, float *buf) 
{
    vfloat32m1x4_t zip = vzip_f32_v4(vector1, vector2, vector3, vector4);
    vse32_v_f32m1(buf, vget_f32m1x4_f32m1(zip, 0), VL);
    vse32_v_f32m1(buf + 4, vget_f32m1x4_f32m1(zip, 1), VL);
    vse32_v_f32m1(buf + 8, vget_f32m1x4_f32m1(zip, 2), VL);
    vse32_v_f32m1(buf + 12, vget_f32m1x4_f32m1(zip, 3), VL);
}

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
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;
            const float* p1 = (const float*)A + (i + ii + 4) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p1, VL), VL);
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
                vfloat32m1_t _r0l = vle32_v_f32m1(p0, VL);
                vfloat32m1_t _r0h = vle32_v_f32m1(p0 + 4, VL);
                vfloat32m1_t _r1l = vle32_v_f32m1(p1, VL);
                vfloat32m1_t _r1h = vle32_v_f32m1(p1 + 4, VL);
                vfloat32m1_t _r2l = vle32_v_f32m1(p2, VL);
                vfloat32m1_t _r2h = vle32_v_f32m1(p2 + 4, VL);
                vfloat32m1_t _r3l = vle32_v_f32m1(p3, VL);
                vfloat32m1_t _r3h = vle32_v_f32m1(p3 + 4, VL);
                vfloat32m1_t _r4l = vle32_v_f32m1(p4, VL);
                vfloat32m1_t _r4h = vle32_v_f32m1(p4 + 4, VL);
                vfloat32m1_t _r5l = vle32_v_f32m1(p5, VL);
                vfloat32m1_t _r5h = vle32_v_f32m1(p5 + 4, VL);
                vfloat32m1_t _r6l = vle32_v_f32m1(p6, VL);
                vfloat32m1_t _r6h = vle32_v_f32m1(p6 + 4, VL);
                vfloat32m1_t _r7l = vle32_v_f32m1(p7, VL);
                vfloat32m1_t _r7h = vle32_v_f32m1(p7 + 4, VL);
                transpose8x8_ps(_r0l, _r0h, _r1l, _r1h, _r2l, _r2h, _r3l, _r3h, _r4l, _r4h, _r5l, _r5h, _r6l, _r6h, _r7l, _r7h);
                vse32_v_f32m1(pp, _r0l, VL);
                vse32_v_f32m1(pp + 4, _r0h, VL);
                vse32_v_f32m1(pp + 8, _r1l, VL);
                vse32_v_f32m1(pp + 12, _r1h, VL);
                vse32_v_f32m1(pp + 8 * 2, _r2l, VL);
                vse32_v_f32m1(pp + 8 * 2 + 4, _r2h, VL);
                vse32_v_f32m1(pp + 8 * 3, _r3l, VL);
                vse32_v_f32m1(pp + 8 * 3 + 4, _r3h, VL);
                vse32_v_f32m1(pp + 8 * 4, _r4l, VL);
                vse32_v_f32m1(pp + 8 * 4 + 4, _r4h, VL);
                vse32_v_f32m1(pp + 8 * 5, _r5l, VL);
                vse32_v_f32m1(pp + 8 * 5 + 4, _r5h, VL);
                vse32_v_f32m1(pp + 8 * 6, _r6l, VL);
                vse32_v_f32m1(pp + 8 * 6 + 4, _r6h, VL);
                vse32_v_f32m1(pp + 8 * 7, _r7l, VL);
                vse32_v_f32m1(pp + 8 * 7 + 4, _r7h, VL);
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
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k * 4;

            for (int kk = 0; kk < max_kk; kk++)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
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
                // vfloat32m4_t _r0123;
                // vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                // vse32_v_f32m1(pp + 4, vle32_v_f32m1(p1, VL), VL);
                // vse32_v_f32m1(pp + 8, vle32_v_f32m1(p2, VL), VL);
                // vse32_v_f32m1(pp + 12, vle32_v_f32m1(p3, VL), VL);
                store_float_v4(vle32_v_f32m1(p0, VL), vle32_v_f32m1(p1, VL), vle32_v_f32m1(p2, VL), vle32_v_f32m1(p3, VL), pp);
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
    for (; ii + 1 < max_ii; ii += 2)
    {
        // if (elempack == 1)
        {
            const float* p0 = (const float*)A + (i + ii) * A_hstep + k;
            const float* p1 = (const float*)A + (i + ii + 1) * A_hstep + k;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // vfloat32m1x2_t _r01;
                // _r01.val[0] = vle32_v_f32m1(p0, VL);
                // _r01.val[1] = vle32_v_f32m1(p1, VL);
                store_float32_v2(vle32_v_f32m1(p0, VL), vle32_v_f32m1(p1, VL), pp);
                // vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL); 
                // vse32_v_f32m1(pp + 4, vle32_v_f32m1(p1, VL), VL); 
                // vsseg2e32_v_f32m1x2(pp, _r01, VL);
                pp += 8;
                p0 += 4;
                p1 += 4;
            }
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
            for (; kk + 3 < max_kk; kk += 4)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                pp += 4;
                p0 += 4;
            }
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
#if PACK_8
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // vfloat32m1x4_t _r0123 = vlseg4e32_v_f32m1x4(p0, VL);
                // vfloat32m1x4_t _r4567 = vlseg4e32_v_f32m1x4(p0 + 16, VL);
                // vse32_v_f32m1(pp, _r0123.val[0], VL);
                // vse32_v_f32m1(pp + 4, _r4567.val[0], VL);
                // vse32_v_f32m1(pp + 4 * 2, _r0123.val[1], VL);
                // vse32_v_f32m1(pp + 4 * 3, _r4567.val[1], VL);
                // vse32_v_f32m1(pp + 4 * 4, _r0123.val[2], VL);
                // vse32_v_f32m1(pp + 4 * 5, _r4567.val[2], VL);
                // vse32_v_f32m1(pp + 4 * 6, _r0123.val[3], VL);
                // vse32_v_f32m1(pp + 4 * 7, _r4567.val[3], VL);
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
#endif // PACK_8
    for (; ii + 3 < max_ii; ii += 4) {
        if (elempack == 4) {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4) {
                // vfloat32m1x4_t _r0123 = vlseg4e32_v_f32m1x4(p0, VL);
                // vfloat32m1x4_t _r4567 = vlseg4e32_v_f32m1x4(p0 + 16, VL);
                // vse32_v_f32m1(pp, _r0123.val[0], VL);
                // vse32_v_f32m1(pp + 4, _r4567.val[0], VL);
                // vse32_v_f32m1(pp + 4 * 2, _r0123.val[1], VL);
                // vse32_v_f32m1(pp + 4 * 3, _r4567.val[1], VL);
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
                vfloat32m1x2_t _r01;
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
    for (; ii + 7 < max_ii; ii += 8)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1x4_t _r0123 = vlseg4e32_v_f32m1x4(p0, VL);
                vfloat32m1x4_t _r4567 = vlseg4e32_v_f32m1x4(p0 + 16, VL);
                vse32_v_f32m1(pp, vget_f32m1x4_f32m1(_r0123, 0), VL);
                vse32_v_f32m1(pp + 4, vget_f32m1x4_f32m1(_r4567, 0), VL);
                vse32_v_f32m1(pp + 4 * 2, vget_f32m1x4_f32m1(_r0123, 1), VL);
                vse32_v_f32m1(pp + 4 * 3, vget_f32m1x4_f32m1(_r4567, 1), VL);
                vse32_v_f32m1(pp + 4 * 4, vget_f32m1x4_f32m1(_r0123, 2), VL);
                vse32_v_f32m1(pp + 4 * 5, vget_f32m1x4_f32m1(_r4567, 2), VL);
                vse32_v_f32m1(pp + 4 * 6, vget_f32m1x4_f32m1(_r0123, 3), VL);
                vse32_v_f32m1(pp + 4 * 7, vget_f32m1x4_f32m1(_r4567, 3), VL);
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
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                vse32_v_f32m1(pp + 4, vle32_v_f32m1(p0 + 4, VL), VL);
                pp += 8;
                p0 += A_hstep;
            }
        }
    }
    for (; ii + 3 < max_ii; ii += 4)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1x4_t _r0123 = vlseg4e32_v_f32m1x4(p0, VL);
                vse32_v_f32m1(pp, vget_f32m1x4_f32m1(_r0123, 0), VL);
                vse32_v_f32m1(pp + 4, vget_f32m1x4_f32m1(_r0123, 1), VL);
                vse32_v_f32m1(pp + 4 * 2, vget_f32m1x4_f32m1(_r0123, 2), VL);
                vse32_v_f32m1(pp + 4 * 3, vget_f32m1x4_f32m1(_r0123, 3), VL);
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
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
                pp += 4;
                p0 += A_hstep;
            }
        }
    }
    for (; ii + 1 < max_ii; ii += 2)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                // vfloat32m1x2_t _r01;
                // vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL); 
                // vse32_v_f32m1(pp + 4, vle32_v_f32m1(p0 + 4, VL), VL);
                store_float32_v2(vle32_v_f32m1(p0, VL), vle32_v_f32m1(p0 + 4, VL), pp);
                // vsseg2e32_v_f32m1x2(pp, _r01, VL);
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
    for (; ii < max_ii; ii += 1)
    {
        if (elempack == 4)
        {
            const float* p0 = (const float*)A + k * A_hstep + (i + ii) * 4;

            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vse32_v_f32m1(pp, vle32_v_f32m1(p0, VL), VL);
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
    pack_A_tile_naive(in, out_check, 0, h, 0, w);

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
    transpose_pack_A_tile(in, out, 0, h, 0, w);
    transpose_pack_A_tile_naive(in, out_check, 0, h, 0, w);
#if PRINT_MAT
    print_Mat(out);
#endif
    if (!check_Mat(out, out_check)) {
        printf("error\n");
    } else {
        printf("correct\n");
    }
    delete [] data_in, data_out;

    return 0;
}