#include <riscv_vector.h>

template<typename T = float>
struct Mat
{
    int elempack = 1;
    int dims = 1;
    int h = 32;
    int w = 32;
    int cstep = 1;

    T* data;

    operator T *() const
    {
        return data;
    }

    T* operator[](int i) const
    {
        return data + i * w;
    }

    void init_Mat() {
        data = new T[h * w];
        for (int i = 0; i < h * w; i++) {
            data[i] = i;
        }
    }

    void print_Mat() {
        for (int i = 0; i < h; i++) {
            printf("%d: ", i);
            for (int j = 0; j < w; j++) {
                printf("%3.0f ", data[i * w + j]);
            }
            printf("\n");
        } 
    }

    bool check_Mat(const Mat& correct) {
        T* data_correct = correct;
        int w = correct.w;
        int h = correct.h;
        int cstep = correct.cstep;
        int wstep = w * cstep;
        int hstep = h * wstep;
        for (int i = 0; i < hstep; i++) {
            if (data_correct[i] != data[i]) {
                printf("error at %d, correct: %f, test: %f\n", i, data_correct[i], data[i]);
                return false;
            }
        }
        return true;
    }
    
};

template<typename T = float>
void init_Mat(Mat<T>& mat) {
    T* data = mat;
    int w = mat.w;
    int h = mat.h;
    printf("in init_Mat, w: %d, h: %d\n", w, h);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            data[i * w + j] = i * w + j;
        }
    }
}

template<typename T = float>
void print_Mat(Mat<T>& mat) {
    T* data = mat;
    int w = mat.w;
    int h = mat.h;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%3.0f ", data[i * w + j]);
        }
        printf("\n");
    } 
}

template<typename T = float>
bool check_Mat(Mat<T>& correct, Mat<T>& test) {
    float* data_correct = correct;
    float* data_test = test;
    int w = correct.w;
    int h = correct.h;
    int cstep = correct.cstep;
    int wstep = w * cstep;
    int hstep = h * wstep;
    for (int i = 0; i < hstep; i++) {
        if (data_correct[i] != data_test[i]) {
            printf("error at %d, correct: %f, test: %f\n", i, data_correct[i], data_test[i]);
            return false;
        }
    }
    return true;
}

vfloat32m1_t vfmaq_laneq_f32_riscv(vfloat32m1_t sum, vfloat32m1_t a, vfloat32m1_t b, int lane) {
    float t[4];
    vse32_v_f32m1(t, b, 4);
    vfloat32m1_t ret = vfmadd_vf_f32m1(
            a, t[lane], sum, 4);
    return ret;     
}

vfloat32m1_t vdupq_n_f32_riscv(float32_t f) {
    return vfmv_v_f_f32m1(f, 4);
}

__fp16 float32_to_float16(float a) {
    return (__fp16)a;
}

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

static inline int csrr_vl()
{
    int a = 0;
    asm volatile("csrr %0, vl"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}

static inline int csrr_vtype()
{
    int a = 0;
    asm volatile("csrr %0, vtype"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}

static inline int csrr_vlenb()
{
    int a = 0;
    asm volatile("csrr %0, vlenb"
                 : "=r"(a)
                 :
                 : "memory");
    return a;
}

static inline void transpose4x4_f16(vfloat16mf2_t& _r0, vfloat16mf2_t& _r1, vfloat16mf2_t& _r2, vfloat16mf2_t& _r3, size_t vl)
{
    __fp16 tmp[4][4];
    vsse16_v_f16m1(&tmp[0][0], sizeof(__fp16) * 4, _r0, vl);
    vsse16_v_f16m1(&tmp[0][1], sizeof(__fp16) * 4, _r1, vl);
    vsse16_v_f16m1(&tmp[0][2], sizeof(__fp16) * 4, _r2, vl);
    vsse16_v_f16m1(&tmp[0][3], sizeof(__fp16) * 4, _r3, vl);
    __fp16* ptr = (__fp16*)tmp;
    _r0 = vle16_v_f16m1(ptr + 0 * 4, vl);
    _r1 = vle16_v_f16m1(ptr + 1 * 4, vl);
    _r2 = vle16_v_f16m1(ptr + 2 * 4, vl);
    _r3 = vle16_v_f16m1(ptr + 3 * 4, vl);
}

static inline void transpose8x8_f16(vfloat16m1_t& _r0, vfloat16m1_t _r1, vfloat16m1_t& _r2, vfloat16m1_t& _r3, vfloat16m1_t& _r4, vfloat16m1_t& _r5, vfloat16m1_t& _r6, vfloat16m1_t& _r7, size_t vl)
{
    __fp16 tmp[8][8];
    vsse16_v_f16m1(&tmp[0][0], sizeof(__fp16) * 8, _r0, vl);
    vsse16_v_f16m1(&tmp[0][1], sizeof(__fp16) * 8, _r1, vl);
    vsse16_v_f16m1(&tmp[0][2], sizeof(__fp16) * 8, _r2, vl);
    vsse16_v_f16m1(&tmp[0][3], sizeof(__fp16) * 8, _r3, vl);
    vsse16_v_f16m1(&tmp[0][4], sizeof(__fp16) * 8, _r4, vl);
    vsse16_v_f16m1(&tmp[0][5], sizeof(__fp16) * 8, _r5, vl);
    vsse16_v_f16m1(&tmp[0][6], sizeof(__fp16) * 8, _r6, vl);
    vsse16_v_f16m1(&tmp[0][7], sizeof(__fp16) * 8, _r7, vl);
    __fp16* ptr = (__fp16*)tmp;
    _r0 = vle16_v_f16m1(ptr + 0 * 4, vl);
    _r1 = vle16_v_f16m1(ptr + 1 * 4, vl);
    _r2 = vle16_v_f16m1(ptr + 2 * 4, vl);
    _r3 = vle16_v_f16m1(ptr + 3 * 4, vl);
    _r4 = vle16_v_f16m1(ptr + 4 * 4, vl);
    _r5 = vle16_v_f16m1(ptr + 5 * 4, vl);
    _r6 = vle16_v_f16m1(ptr + 6 * 4, vl);
    _r7 = vle16_v_f16m1(ptr + 7 * 4, vl);
}

static inline void transpose8x8_ps(vfloat32m1_t& _r0l, vfloat32m1_t& _r0h,
                                   vfloat32m1_t& _r1l, vfloat32m1_t& _r1h,
                                   vfloat32m1_t& _r2l, vfloat32m1_t& _r2h,
                                   vfloat32m1_t& _r3l, vfloat32m1_t& _r3h,
                                   vfloat32m1_t& _r4l, vfloat32m1_t& _r4h,
                                   vfloat32m1_t& _r5l, vfloat32m1_t& _r5h,
                                   vfloat32m1_t& _r6l, vfloat32m1_t& _r6h,
                                   vfloat32m1_t& _r7l, vfloat32m1_t& _r7h, size_t vl)
{
    float tmp[8][8];
    vsse32_v_f32m1(&tmp[0][0], sizeof(float) * 8, _r0l, vl);
    vsse32_v_f32m1(&tmp[4][0], sizeof(float) * 8, _r0h, vl);
    vsse32_v_f32m1(&tmp[0][1], sizeof(float) * 8, _r1l, vl);
    vsse32_v_f32m1(&tmp[4][1], sizeof(float) * 8, _r1h, vl);
    vsse32_v_f32m1(&tmp[0][2], sizeof(float) * 8, _r2l, vl);
    vsse32_v_f32m1(&tmp[4][2], sizeof(float) * 8, _r2h, vl);
    vsse32_v_f32m1(&tmp[0][3], sizeof(float) * 8, _r3l, vl);
    vsse32_v_f32m1(&tmp[4][3], sizeof(float) * 8, _r3h, vl);
    vsse32_v_f32m1(&tmp[0][4], sizeof(float) * 8, _r4l, vl);
    vsse32_v_f32m1(&tmp[4][4], sizeof(float) * 8, _r4h, vl);
    vsse32_v_f32m1(&tmp[0][5], sizeof(float) * 8, _r5l, vl);
    vsse32_v_f32m1(&tmp[4][5], sizeof(float) * 8, _r5h, vl);
    vsse32_v_f32m1(&tmp[0][6], sizeof(float) * 8, _r6l, vl);
    vsse32_v_f32m1(&tmp[4][6], sizeof(float) * 8, _r6h, vl);
    vsse32_v_f32m1(&tmp[0][7], sizeof(float) * 8, _r7l, vl);
    vsse32_v_f32m1(&tmp[4][7], sizeof(float) * 8, _r7h, vl);
    float* ptr = (float*)tmp;
    _r0l = vle32_v_f32m1(ptr + 0 * 4, vl);
    _r0h = vle32_v_f32m1(ptr + 1 * 4, vl);
    _r1l = vle32_v_f32m1(ptr + 2 * 4, vl);
    _r1h = vle32_v_f32m1(ptr + 3 * 4, vl);
    _r2l = vle32_v_f32m1(ptr + 4 * 4, vl);
    _r2h = vle32_v_f32m1(ptr + 5 * 4, vl);
    _r3l = vle32_v_f32m1(ptr + 6 * 4, vl);
    _r3h = vle32_v_f32m1(ptr + 7 * 4, vl);
    _r4l = vle32_v_f32m1(ptr + 8 * 4, vl);
    _r4h = vle32_v_f32m1(ptr + 9 * 4, vl);
    _r5l = vle32_v_f32m1(ptr + 10 * 4, vl);
    _r5h = vle32_v_f32m1(ptr + 11 * 4, vl);
    _r6l = vle32_v_f32m1(ptr + 12 * 4, vl);
    _r6h = vle32_v_f32m1(ptr + 13 * 4, vl);
    _r7l = vle32_v_f32m1(ptr + 14 * 4, vl);
    _r7h = vle32_v_f32m1(ptr + 15 * 4, vl);
}

static inline void transpose4x4_ps(vfloat32m1_t& _r0, vfloat32m1_t& _r1, vfloat32m1_t& _r2, vfloat32m1_t& _r3, size_t vl)
{
    float tmp[4][4];
    vsse32_v_f32m1(&tmp[0][0], sizeof(float) * 4, _r0, vl);
    vsse32_v_f32m1(&tmp[0][1], sizeof(float) * 4, _r1, vl);
    vsse32_v_f32m1(&tmp[0][2], sizeof(float) * 4, _r2, vl);
    vsse32_v_f32m1(&tmp[0][3], sizeof(float) * 4, _r3, vl);
    float* ptr = (float*)tmp;
    _r0 = vle32_v_f32m1(ptr + 0 * 4, vl);
    _r1 = vle32_v_f32m1(ptr + 1 * 4, vl);
    _r2 = vle32_v_f32m1(ptr + 2 * 4, vl);
    _r3 = vle32_v_f32m1(ptr + 3 * 4, vl);
}

static inline void transpose8x12_ps(vfloat32m1_t& _r0l, vfloat32m1_t& _r0h,
                                    vfloat32m1_t& _r1l, vfloat32m1_t& _r1h,
                                    vfloat32m1_t& _r2l, vfloat32m1_t& _r2h,
                                    vfloat32m1_t& _r3l, vfloat32m1_t& _r3h,
                                    vfloat32m1_t& _r4l, vfloat32m1_t& _r4h,
                                    vfloat32m1_t& _r5l, vfloat32m1_t& _r5h,
                                    vfloat32m1_t& _r6l, vfloat32m1_t& _r6h,
                                    vfloat32m1_t& _r7l, vfloat32m1_t& _r7h,
                                    vfloat32m1_t& _r8l, vfloat32m1_t& _r8h,
                                    vfloat32m1_t& _r9l, vfloat32m1_t& _r9h,
                                    vfloat32m1_t& _ral, vfloat32m1_t& _rah,
                                    vfloat32m1_t& _rbl, vfloat32m1_t& _rbh, size_t vl)
{
    float tmp[8][12];
    vsse32_v_f32m1(&tmp[0][0], sizeof(float) * 12, _r0l, vl);
    vsse32_v_f32m1(&tmp[4][0], sizeof(float) * 12, _r0h, vl);
    vsse32_v_f32m1(&tmp[0][1], sizeof(float) * 12, _r1l, vl);
    vsse32_v_f32m1(&tmp[4][1], sizeof(float) * 12, _r1h, vl);
    vsse32_v_f32m1(&tmp[0][2], sizeof(float) * 12, _r2l, vl);
    vsse32_v_f32m1(&tmp[4][2], sizeof(float) * 12, _r2h, vl);
    vsse32_v_f32m1(&tmp[0][3], sizeof(float) * 12, _r3l, vl);
    vsse32_v_f32m1(&tmp[4][3], sizeof(float) * 12, _r3h, vl);
    vsse32_v_f32m1(&tmp[0][4], sizeof(float) * 12, _r4l, vl);
    vsse32_v_f32m1(&tmp[4][4], sizeof(float) * 12, _r4h, vl);
    vsse32_v_f32m1(&tmp[0][5], sizeof(float) * 12, _r5l, vl);
    vsse32_v_f32m1(&tmp[4][5], sizeof(float) * 12, _r5h, vl);
    vsse32_v_f32m1(&tmp[0][6], sizeof(float) * 12, _r6l, vl);
    vsse32_v_f32m1(&tmp[4][6], sizeof(float) * 12, _r6h, vl);
    vsse32_v_f32m1(&tmp[0][7], sizeof(float) * 12, _r7l, vl);
    vsse32_v_f32m1(&tmp[4][7], sizeof(float) * 12, _r7h, vl);
    vsse32_v_f32m1(&tmp[0][8], sizeof(float) * 12, _r8l, vl);
    vsse32_v_f32m1(&tmp[4][8], sizeof(float) * 12, _r8h, vl);
    vsse32_v_f32m1(&tmp[0][9], sizeof(float) * 12, _r9l, vl);
    vsse32_v_f32m1(&tmp[4][9], sizeof(float) * 12, _r9h, vl);
    vsse32_v_f32m1(&tmp[0][10], sizeof(float) * 12, _ral, vl);
    vsse32_v_f32m1(&tmp[4][10], sizeof(float) * 12, _rah, vl);
    vsse32_v_f32m1(&tmp[0][11], sizeof(float) * 12, _rbl, vl);
    vsse32_v_f32m1(&tmp[4][11], sizeof(float) * 12, _rbh, vl);
    float* ptr = (float*)tmp;
    _r0l = vle32_v_f32m1(ptr + 0 * 4, vl);
    _r0h = vle32_v_f32m1(ptr + 1 * 4, vl);
    _r1l = vle32_v_f32m1(ptr + 2 * 4, vl);
    _r1h = vle32_v_f32m1(ptr + 3 * 4, vl);
    _r2l = vle32_v_f32m1(ptr + 4 * 4, vl);
    _r2h = vle32_v_f32m1(ptr + 5 * 4, vl);
    _r3l = vle32_v_f32m1(ptr + 6 * 4, vl);
    _r3h = vle32_v_f32m1(ptr + 7 * 4, vl);
    _r4l = vle32_v_f32m1(ptr + 8 * 4, vl);
    _r4h = vle32_v_f32m1(ptr + 9 * 4, vl);
    _r5l = vle32_v_f32m1(ptr + 10 * 4, vl);
    _r5h = vle32_v_f32m1(ptr + 11 * 4, vl);
    _r6l = vle32_v_f32m1(ptr + 12 * 4, vl);
    _r6h = vle32_v_f32m1(ptr + 13 * 4, vl);
    _r7l = vle32_v_f32m1(ptr + 14 * 4, vl);
    _r7h = vle32_v_f32m1(ptr + 15 * 4, vl);
    _r8l = vle32_v_f32m1(ptr + 16 * 4, vl);
    _r8h = vle32_v_f32m1(ptr + 17 * 4, vl);
    _r9l = vle32_v_f32m1(ptr + 18 * 4, vl);
    _r9h = vle32_v_f32m1(ptr + 19 * 4, vl);
    _ral = vle32_v_f32m1(ptr + 20 * 4, vl);
    _rah = vle32_v_f32m1(ptr + 21 * 4, vl);
    _rbl = vle32_v_f32m1(ptr + 22 * 4, vl);
    _rbh = vle32_v_f32m1(ptr + 23 * 4, vl);
}

static inline void transpose12x8_ps(vfloat32m1_t& _r0l, vfloat32m1_t& _r0m, vfloat32m1_t& _r0h,
                                    vfloat32m1_t& _r1l, vfloat32m1_t& _r1m, vfloat32m1_t& _r1h,
                                    vfloat32m1_t& _r2l, vfloat32m1_t& _r2m, vfloat32m1_t& _r2h,
                                    vfloat32m1_t& _r3l, vfloat32m1_t& _r3m, vfloat32m1_t& _r3h,
                                    vfloat32m1_t& _r4l, vfloat32m1_t& _r4m, vfloat32m1_t& _r4h,
                                    vfloat32m1_t& _r5l, vfloat32m1_t& _r5m, vfloat32m1_t& _r5h,
                                    vfloat32m1_t& _r6l, vfloat32m1_t& _r6m, vfloat32m1_t& _r6h,
                                    vfloat32m1_t& _r7l, vfloat32m1_t& _r7m, vfloat32m1_t& _r7h, size_t vl)
{
    float tmp[12][8];
    vsse32_v_f32m1(&tmp[0][0], sizeof(float) * 8, _r0l, vl);
    vsse32_v_f32m1(&tmp[4][0], sizeof(float) * 8, _r0m, vl);
    vsse32_v_f32m1(&tmp[8][0], sizeof(float) * 8, _r0h, vl);
    vsse32_v_f32m1(&tmp[0][1], sizeof(float) * 8, _r1l, vl);
    vsse32_v_f32m1(&tmp[4][1], sizeof(float) * 8, _r1m, vl);
    vsse32_v_f32m1(&tmp[8][0], sizeof(float) * 8, _r1h, vl);
    vsse32_v_f32m1(&tmp[0][2], sizeof(float) * 8, _r2l, vl);
    vsse32_v_f32m1(&tmp[4][2], sizeof(float) * 8, _r2m, vl);
    vsse32_v_f32m1(&tmp[8][2], sizeof(float) * 8, _r2h, vl);
    vsse32_v_f32m1(&tmp[0][3], sizeof(float) * 8, _r3l, vl);
    vsse32_v_f32m1(&tmp[4][3], sizeof(float) * 8, _r3m, vl);
    vsse32_v_f32m1(&tmp[8][3], sizeof(float) * 8, _r3h, vl);
    vsse32_v_f32m1(&tmp[0][4], sizeof(float) * 8, _r4l, vl);
    vsse32_v_f32m1(&tmp[4][4], sizeof(float) * 8, _r4m, vl);
    vsse32_v_f32m1(&tmp[8][4], sizeof(float) * 8, _r4h, vl);
    vsse32_v_f32m1(&tmp[0][5], sizeof(float) * 8, _r5l, vl);
    vsse32_v_f32m1(&tmp[4][5], sizeof(float) * 8, _r5m, vl);
    vsse32_v_f32m1(&tmp[8][5], sizeof(float) * 8, _r5h, vl);
    vsse32_v_f32m1(&tmp[0][6], sizeof(float) * 8, _r6l, vl);
    vsse32_v_f32m1(&tmp[4][6], sizeof(float) * 8, _r6m, vl);
    vsse32_v_f32m1(&tmp[8][6], sizeof(float) * 8, _r6h, vl);
    vsse32_v_f32m1(&tmp[0][7], sizeof(float) * 8, _r7l, vl);
    vsse32_v_f32m1(&tmp[4][7], sizeof(float) * 8, _r7m, vl);
    vsse32_v_f32m1(&tmp[8][7], sizeof(float) * 8, _r7h, vl);
    float* ptr = (float*)tmp;
    _r0l = vle32_v_f32m1(ptr + 0 * 4, vl);
    _r0m = vle32_v_f32m1(ptr + 1 * 4, vl);
    _r0h = vle32_v_f32m1(ptr + 2 * 4, vl);
    _r1l = vle32_v_f32m1(ptr + 3 * 4, vl);
    _r1m = vle32_v_f32m1(ptr + 4 * 4, vl);
    _r1h = vle32_v_f32m1(ptr + 5 * 4, vl);
    _r2l = vle32_v_f32m1(ptr + 6 * 4, vl);
    _r2m = vle32_v_f32m1(ptr + 7 * 4, vl);
    _r2h = vle32_v_f32m1(ptr + 8 * 4, vl);
    _r3l = vle32_v_f32m1(ptr + 9 * 4, vl);
    _r3m = vle32_v_f32m1(ptr + 10 * 4, vl);
    _r3h = vle32_v_f32m1(ptr + 11 * 4, vl);
    _r4l = vle32_v_f32m1(ptr + 12 * 4, vl);
    _r4m = vle32_v_f32m1(ptr + 13 * 4, vl);
    _r4h = vle32_v_f32m1(ptr + 14 * 4, vl);
    _r5l = vle32_v_f32m1(ptr + 15 * 4, vl);
    _r5m = vle32_v_f32m1(ptr + 16 * 4, vl);
    _r5h = vle32_v_f32m1(ptr + 17 * 4, vl);
    _r6l = vle32_v_f32m1(ptr + 18 * 4, vl);
    _r6m = vle32_v_f32m1(ptr + 19 * 4, vl);
    _r6h = vle32_v_f32m1(ptr + 20 * 4, vl);
    _r7l = vle32_v_f32m1(ptr + 21 * 4, vl);
    _r7m = vle32_v_f32m1(ptr + 22 * 4, vl);
    _r7h = vle32_v_f32m1(ptr + 23 * 4, vl);
}

static inline void transpose4x8_ps(vfloat32m1_t& _r0, vfloat32m1_t& _r1, vfloat32m1_t& _r2, vfloat32m1_t& _r3, vfloat32m1_t& _r4, vfloat32m1_t& _r5, vfloat32m1_t& _r6, vfloat32m1_t& _r7, size_t vl)
{
    float tmp[4][8];
    vsse32_v_f32m1(&tmp[0][0], sizeof(float) * 8, _r0, vl);
    vsse32_v_f32m1(&tmp[0][1], sizeof(float) * 8, _r1, vl);
    vsse32_v_f32m1(&tmp[0][2], sizeof(float) * 8, _r2, vl);
    vsse32_v_f32m1(&tmp[0][3], sizeof(float) * 8, _r3, vl);
    vsse32_v_f32m1(&tmp[0][4], sizeof(float) * 8, _r4, vl);
    vsse32_v_f32m1(&tmp[0][5], sizeof(float) * 8, _r5, vl);
    vsse32_v_f32m1(&tmp[0][6], sizeof(float) * 8, _r6, vl);
    vsse32_v_f32m1(&tmp[0][7], sizeof(float) * 8, _r7, vl);
    float* ptr = (float*)tmp;
    _r0 = vle32_v_f32m1(ptr + 0 * 4, vl);
    _r1 = vle32_v_f32m1(ptr + 1 * 4, vl);
    _r2 = vle32_v_f32m1(ptr + 2 * 4, vl);
    _r3 = vle32_v_f32m1(ptr + 3 * 4, vl);
    _r4 = vle32_v_f32m1(ptr + 4 * 4, vl);
    _r5 = vle32_v_f32m1(ptr + 5 * 4, vl);
    _r6 = vle32_v_f32m1(ptr + 6 * 4, vl);
    _r7 = vle32_v_f32m1(ptr + 7 * 4, vl);
}

static inline void transpose4x12_ps(vfloat32m1_t& _r0, vfloat32m1_t& _r1, vfloat32m1_t& _r2, vfloat32m1_t& _r3, vfloat32m1_t& _r4, vfloat32m1_t& _r5, vfloat32m1_t& _r6, vfloat32m1_t& _r7, vfloat32m1_t& _r8, vfloat32m1_t& _r9, vfloat32m1_t& _ra, vfloat32m1_t& _rb, size_t vl)
{
    float tmp[4][12];
    vsse32_v_f32m1(&tmp[0][0], sizeof(float) * 12, _r0, vl);
    vsse32_v_f32m1(&tmp[0][1], sizeof(float) * 12, _r1, vl);
    vsse32_v_f32m1(&tmp[0][2], sizeof(float) * 12, _r2, vl);
    vsse32_v_f32m1(&tmp[0][3], sizeof(float) * 12, _r3, vl);
    vsse32_v_f32m1(&tmp[0][4], sizeof(float) * 12, _r4, vl);
    vsse32_v_f32m1(&tmp[0][5], sizeof(float) * 12, _r5, vl);
    vsse32_v_f32m1(&tmp[0][6], sizeof(float) * 12, _r6, vl);
    vsse32_v_f32m1(&tmp[0][7], sizeof(float) * 12, _r7, vl);
    vsse32_v_f32m1(&tmp[0][8], sizeof(float) * 12, _r8, vl);
    vsse32_v_f32m1(&tmp[0][9], sizeof(float) * 12, _r9, vl);
    vsse32_v_f32m1(&tmp[0][10], sizeof(float) * 12, _ra, vl);
    vsse32_v_f32m1(&tmp[0][11], sizeof(float) * 12, _rb, vl);
    float* ptr = (float*)tmp;
    _r0 = vle32_v_f32m1(ptr + 0 * 4, vl);
    _r1 = vle32_v_f32m1(ptr + 1 * 4, vl);
    _r2 = vle32_v_f32m1(ptr + 2 * 4, vl);
    _r3 = vle32_v_f32m1(ptr + 3 * 4, vl);
    _r4 = vle32_v_f32m1(ptr + 4 * 4, vl);
    _r5 = vle32_v_f32m1(ptr + 5 * 4, vl);
    _r6 = vle32_v_f32m1(ptr + 6 * 4, vl);
    _r7 = vle32_v_f32m1(ptr + 7 * 4, vl);
    _r8 = vle32_v_f32m1(ptr + 8 * 4, vl);
    _r9 = vle32_v_f32m1(ptr + 9 * 4, vl);
    _ra = vle32_v_f32m1(ptr + 10 * 4, vl);
    _rb = vle32_v_f32m1(ptr + 11 * 4, vl);
}

static inline void transpose8x4_ps(vfloat32m1_t& _r0l, vfloat32m1_t& _r0h,
                                   vfloat32m1_t& _r1l, vfloat32m1_t& _r1h,
                                   vfloat32m1_t& _r2l, vfloat32m1_t& _r2h,
                                   vfloat32m1_t& _r3l, vfloat32m1_t& _r3h, size_t vl)
{
    float tmp[8][4];
    vsse32_v_f32m1(&tmp[0][0], sizeof(float) * 4, _r0l, vl);
    vsse32_v_f32m1(&tmp[4][0], sizeof(float) * 4, _r0h, vl);
    vsse32_v_f32m1(&tmp[0][1], sizeof(float) * 4, _r1l, vl);
    vsse32_v_f32m1(&tmp[4][1], sizeof(float) * 4, _r1h, vl);
    vsse32_v_f32m1(&tmp[0][2], sizeof(float) * 4, _r2l, vl);
    vsse32_v_f32m1(&tmp[4][2], sizeof(float) * 4, _r2h, vl);
    vsse32_v_f32m1(&tmp[0][3], sizeof(float) * 4, _r3l, vl);
    vsse32_v_f32m1(&tmp[4][3], sizeof(float) * 4, _r3h, vl);
    float* ptr = (float*)tmp;
    _r0l = vle32_v_f32m1(ptr + 0 * 4, vl);
    _r0h = vle32_v_f32m1(ptr + 1 * 4, vl);
    _r1l = vle32_v_f32m1(ptr + 2 * 4, vl);
    _r1h = vle32_v_f32m1(ptr + 3 * 4, vl);
    _r2l = vle32_v_f32m1(ptr + 4 * 4, vl);
    _r2h = vle32_v_f32m1(ptr + 5 * 4, vl);
    _r3l = vle32_v_f32m1(ptr + 6 * 4, vl);
    _r3h = vle32_v_f32m1(ptr + 7 * 4, vl);
}

static inline void store_float_v2(vfloat32m1_t& vector1, vfloat32m1_t& vector2, float* buf, size_t vl)
{
    vsse32_v_f32m1(buf + 0, sizeof(float) * 2, vector1, vl);
    vsse32_v_f32m1(buf + 1, sizeof(float) * 2, vector2, vl);
}

static inline void store_float_v4(vfloat32m1_t& vector1, vfloat32m1_t& vector2, vfloat32m1_t& vector3, vfloat32m1_t& vector4, float* buf, size_t vl)
{
    vsse32_v_f32m1(buf + 0, sizeof(float) * 4, vector1, vl);
    vsse32_v_f32m1(buf + 1, sizeof(float) * 4, vector2, vl);
    vsse32_v_f32m1(buf + 2, sizeof(float) * 4, vector3, vl);
    vsse32_v_f32m1(buf + 3, sizeof(float) * 4, vector4, vl);
}

#if __riscv_zfh
static inline vfloat16m8_t vle16_v_f16m8_f16m1(const __fp16* ptr)
{
    const int packn = csrr_vlenb() / 2;
    const size_t vl = vsetvl_e16m8(packn * 8);

    // NOTE vloxei8_v_f16m8 gets illegal instruction on d1  --- nihui

    // 128bit
    static const uint16_t index_128bit[64] = {
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14,
        0, 2, 4, 6, 8, 10, 12, 14
    };

    // 256bit
    static const uint16_t index_256bit[128] = {
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
    };

    const uint16_t* index = packn == 8 ? index_128bit : index_256bit;
    vuint16m8_t bindex = vle16_v_u16m8(index, vl);
    return vloxei16_v_f16m8(ptr, bindex, vl);
}
#endif // __riscv_zfh

#if __riscv_zfh && __rvv_tuple
// f32m1, vsseg.v
static inline void vsseg8e32_v_f32m1(float32_t* base, vfloat32m1_t v0, vfloat32m1_t v1, vfloat32m1_t v2, vfloat32m1_t v3, vfloat32m1_t v4, vfloat32m1_t v5, vfloat32m1_t v6, vfloat32m1_t v7, size_t vl)
{
    vfloat32m1x8_t _tmp = vcreate_f32m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
    vsseg8e32_v_f32m1x8(base, _tmp, vl);
}

static inline void vsseg4e32_v_f32m1(float32_t* base, vfloat32m1_t v0, vfloat32m1_t v1, vfloat32m1_t v2, vfloat32m1_t v3, size_t vl)
{
    vfloat32m1x4_t _tmp = vcreate_f32m1x4(v0, v1, v2, v3);
    vsseg4e32_v_f32m1x4(base, _tmp, vl);
}

static inline void vsseg2e32_v_f32m1(float32_t* base, vfloat32m1_t v0, vfloat32m1_t v1, size_t vl)
{
    vfloat32m1x2_t _tmp = vcreate_f32m1x2(v0, v1);
    vsseg2e32_v_f32m1x2(base, _tmp, vl);
}

// f32m1, vssseg.v, 8/4/2
static inline void vssseg8e32_v_f32m1(float32_t* base, ptrdiff_t bstride, vfloat32m1_t v0, vfloat32m1_t v1, vfloat32m1_t v2, vfloat32m1_t v3, vfloat32m1_t v4, vfloat32m1_t v5, vfloat32m1_t v6, vfloat32m1_t v7, size_t vl)
{
    vfloat32m1x8_t _tmp = vcreate_f32m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
    vssseg8e32_v_f32m1x8(base, bstride, _tmp, vl);
}

static inline void vssseg4e32_v_f32m1(float32_t* base, ptrdiff_t bstride, vfloat32m1_t v0, vfloat32m1_t v1, vfloat32m1_t v2, vfloat32m1_t v3, size_t vl)
{
    vfloat32m1x4_t _tmp = vcreate_f32m1x4(v0, v1, v2, v3);
    vssseg4e32_v_f32m1x4(base, bstride, _tmp, vl);
}

static inline void vssseg2e32_v_f32m1(float32_t* base, ptrdiff_t bstride, vfloat32m1_t v0, vfloat32m1_t v1, size_t vl)
{
    vfloat32m1x2_t _tmp = vcreate_f32m1x2(v0, v1);
    vssseg2e32_v_f32m1x2(base, bstride, _tmp, vl);
}

// f32m2, vsseg.v, 4/2
static inline void vsseg4e32_v_f32m2(float32_t* base, vfloat32m2_t v0, vfloat32m2_t v1, vfloat32m2_t v2, vfloat32m2_t v3, size_t vl)
{
    vfloat32m2x4_t _tmp = vcreate_f32m2x4(v0, v1, v2, v3);
    vsseg4e32_v_f32m2x4(base, _tmp, vl);
}

static inline void vsseg2e32_v_f32m2(float32_t* base, vfloat32m2_t v0, vfloat32m2_t v1, size_t vl)
{
    vfloat32m2x2_t _tmp = vcreate_f32m2x2(v0, v1);
    vsseg2e32_v_f32m2x2(base, _tmp, vl);
}

// u16m1, vsseg.v, 8/4
static inline void vsseg8e16_v_u16m1(uint16_t* base, vuint16m1_t v0, vuint16m1_t v1, vuint16m1_t v2, vuint16m1_t v3, vuint16m1_t v4, vuint16m1_t v5, vuint16m1_t v6, vuint16m1_t v7, size_t vl)
{
    vuint16m1x8_t _tmp = vcreate_u16m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
    vsseg8e16_v_u16m1x8(base, _tmp, vl);
}

static inline void vsseg4e16_v_u16m1(uint16_t* base, vuint16m1_t v0, vuint16m1_t v1, vuint16m1_t v2, vuint16m1_t v3, size_t vl)
{
    vuint16m1x4_t _tmp = vcreate_u16m1x4(v0, v1, v2, v3);
    vsseg4e16_v_u16m1x4(base, _tmp, vl);
}

// u16m2, vsseg.v, 4/2
static inline void vsseg4e16_v_u16m2(uint16_t* base, vuint16m2_t v0, vuint16m2_t v1, vuint16m2_t v2, vuint16m2_t v3, size_t vl)
{
    vuint16m2x4_t _tmp = vcreate_u16m2x4(v0, v1, v2, v3);
    vsseg4e16_v_u16m2x4(base, _tmp, vl);
}

static inline void vsseg2e16_v_u16m2(uint16_t* base, vuint16m2_t v0, vuint16m2_t v1, size_t vl)
{
    vuint16m2x2_t _tmp = vcreate_u16m2x2(v0, v1);
    vsseg2e16_v_u16m2x2(base, _tmp, vl);
}

// f32m1, vlseg.v 8/4/2
static inline void vlseg8e32_v_f32m1(vfloat32m1_t* v0, vfloat32m1_t* v1, vfloat32m1_t* v2, vfloat32m1_t* v3, vfloat32m1_t* v4, vfloat32m1_t* v5, vfloat32m1_t* v6, vfloat32m1_t* v7, const float32_t* base, size_t vl)
{
    vfloat32m1x8_t _tmp = vlseg8e32_v_f32m1x8(base, vl);
    *v0 = vget_f32m1x8_f32m1(_tmp, 0);
    *v1 = vget_f32m1x8_f32m1(_tmp, 1);
    *v2 = vget_f32m1x8_f32m1(_tmp, 2);
    *v3 = vget_f32m1x8_f32m1(_tmp, 3);
    *v4 = vget_f32m1x8_f32m1(_tmp, 4);
    *v5 = vget_f32m1x8_f32m1(_tmp, 5);
    *v6 = vget_f32m1x8_f32m1(_tmp, 6);
    *v7 = vget_f32m1x8_f32m1(_tmp, 7);
}

static inline void vlseg4e32_v_f32m1(vfloat32m1_t* v0, vfloat32m1_t* v1, vfloat32m1_t* v2, vfloat32m1_t* v3, const float32_t* base, size_t vl)
{
    vfloat32m1x4_t _tmp = vlseg4e32_v_f32m1x4(base, vl);
    *v0 = vget_f32m1x4_f32m1(_tmp, 0);
    *v1 = vget_f32m1x4_f32m1(_tmp, 1);
    *v2 = vget_f32m1x4_f32m1(_tmp, 2);
    *v3 = vget_f32m1x4_f32m1(_tmp, 3);
}

static inline void vlseg2e32_v_f32m1(vfloat32m1_t* v0, vfloat32m1_t* v1, const float32_t* base, size_t vl)
{
    vfloat32m1x2_t _tmp = vlseg2e32_v_f32m1x2(base, vl);
    *v0 = vget_f32m1x2_f32m1(_tmp, 0);
    *v1 = vget_f32m1x2_f32m1(_tmp, 1);
}

// f32m2, vlseg.v, 4
static inline void vlseg4e32_v_f32m2(vfloat32m2_t* v0, vfloat32m2_t* v1, vfloat32m2_t* v2, vfloat32m2_t* v3, const float32_t* base, size_t vl)
{
    vfloat32m2x4_t _tmp = vlseg4e32_v_f32m2x4(base, vl);
    *v0 = vget_f32m2x4_f32m2(_tmp, 0);
    *v1 = vget_f32m2x4_f32m2(_tmp, 1);
    *v2 = vget_f32m2x4_f32m2(_tmp, 2);
    *v3 = vget_f32m2x4_f32m2(_tmp, 3);
}

// f32m4, vlseg.v, 2
static inline void vlseg2e32_v_f32m4(vfloat32m4_t* v0, vfloat32m4_t* v1, const float32_t* base, size_t vl)
{
    vfloat32m4x2_t _tmp = vlseg2e32_v_f32m4x2(base, vl);
    *v0 = vget_f32m4x2_f32m4(_tmp, 0);
    *v1 = vget_f32m4x2_f32m4(_tmp, 1);
}

// f32m4, vloxseg.v
static inline void vloxseg2ei32_v_f32m4(vfloat32m4_t* v0, vfloat32m4_t* v1, const float32_t* base, vuint32m4_t bindex, size_t vl)
{
    vfloat32m4x2_t _tmp = vloxseg2ei32_v_f32m4x2(base, bindex, vl);
    *v0 = vget_f32m4x2_f32m4(_tmp, 0);
    *v1 = vget_f32m4x2_f32m4(_tmp, 1);
}

// u16m1, vlseg.v 8/4/2
static inline void vlseg8e16_v_u16m1(vuint16m1_t* v0, vuint16m1_t* v1, vuint16m1_t* v2, vuint16m1_t* v3, vuint16m1_t* v4, vuint16m1_t* v5, vuint16m1_t* v6, vuint16m1_t* v7, const uint16_t* base, size_t vl)
{
    vuint16m1x8_t _tmp = vlseg8e16_v_u16m1x8(base, vl);
    *v0 = vget_u16m1x8_u16m1(_tmp, 0);
    *v1 = vget_u16m1x8_u16m1(_tmp, 1);
    *v2 = vget_u16m1x8_u16m1(_tmp, 2);
    *v3 = vget_u16m1x8_u16m1(_tmp, 3);
    *v4 = vget_u16m1x8_u16m1(_tmp, 4);
    *v5 = vget_u16m1x8_u16m1(_tmp, 5);
    *v6 = vget_u16m1x8_u16m1(_tmp, 6);
    *v7 = vget_u16m1x8_u16m1(_tmp, 7);
}

static inline void vlseg4e16_v_u16m1(vuint16m1_t* v0, vuint16m1_t* v1, vuint16m1_t* v2, vuint16m1_t* v3, const uint16_t* base, size_t vl)
{
    vuint16m1x4_t _tmp = vlseg4e16_v_u16m1x4(base, vl);
    *v0 = vget_u16m1x4_u16m1(_tmp, 0);
    *v1 = vget_u16m1x4_u16m1(_tmp, 1);
    *v2 = vget_u16m1x4_u16m1(_tmp, 2);
    *v3 = vget_u16m1x4_u16m1(_tmp, 3);
}

static inline void vlseg2e16_v_u16m1(vuint16m1_t* v0, vuint16m1_t* v1, const uint16_t* base, size_t vl)
{
    vuint16m1x2_t _tmp = vlseg2e16_v_u16m1x2(base, vl);
    *v0 = vget_u16m1x2_u16m1(_tmp, 0);
    *v1 = vget_u16m1x2_u16m1(_tmp, 1);
}

// u16m2, vlseg.v, 4
static inline void vlseg4e16_v_u16m2(vuint16m2_t* v0, vuint16m2_t* v1, vuint16m2_t* v2, vuint16m2_t* v3, const uint16_t* base, size_t vl)
{
    vuint16m2x4_t _tmp = vlseg4e16_v_u16m2x4(base, vl);
    *v0 = vget_u16m2x4_u16m2(_tmp, 0);
    *v1 = vget_u16m2x4_u16m2(_tmp, 1);
    *v2 = vget_u16m2x4_u16m2(_tmp, 2);
    *v3 = vget_u16m2x4_u16m2(_tmp, 3);
}

// u16m4, vlseg.v, 2
static inline void vlseg2e16_v_u16m4(vuint16m4_t* v0, vuint16m4_t* v1, const uint16_t* base, size_t vl)
{
    vuint16m4x2_t _tmp = vlseg2e16_v_u16m4x2(base, vl);
    *v0 = vget_u16m4x2_u16m4(_tmp, 0);
    *v1 = vget_u16m4x2_u16m4(_tmp, 1);
}

#if __riscv_zfh

// f16m1, vsseg.v, 8/4/2
static inline void vsseg4e16_v_f16mf2(float16_t* base, vfloat16mf2_t v0, vfloat16mf2_t v1, vfloat16mf2_t v2, vfloat16mf2_t v3, size_t vl)
{
    vfloat16mf2x4_t _tmp = vcreate_f16mf2x4(v0, v1, v2, v3);
    vsseg4e16_v_f16mf2x4(base, _tmp, vl);
}

static inline void vsseg2e16_v_f16mf2(float16_t* base, vfloat16mf2_t v0, vfloat16mf2_t v1, size_t vl)
{
    vfloat16mf2x2_t _tmp = vcreate_f16mf2x2(v0, v1);
    vsseg2e16_v_f16mf2x2(base, _tmp, vl);
}

static inline void vsseg8e16_v_f16m1(float16_t* base, vfloat16m1_t v0, vfloat16m1_t v1, vfloat16m1_t v2, vfloat16m1_t v3, vfloat16m1_t v4, vfloat16m1_t v5, vfloat16m1_t v6, vfloat16m1_t v7, size_t vl)
{
    vfloat16m1x8_t _tmp = vcreate_f16m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
    vsseg8e16_v_f16m1x8(base, _tmp, vl);
}

static inline void vsseg4e16_v_f16m1(float16_t* base, vfloat16m1_t v0, vfloat16m1_t v1, vfloat16m1_t v2, vfloat16m1_t v3, size_t vl)
{
    vfloat16m1x4_t _tmp = vcreate_f16m1x4(v0, v1, v2, v3);
    vsseg4e16_v_f16m1x4(base, _tmp, vl);
}

static inline void vsseg2e16_v_f16m1(float16_t* base, vfloat16m1_t v0, vfloat16m1_t v1, size_t vl)
{
    vfloat16m1x2_t _tmp = vcreate_f16m1x2(v0, v1);
    vsseg2e16_v_f16m1x2(base, _tmp, vl);
}

// f16m1, vssseg.v, 8/4/2
static inline void vssseg8e16_v_f16m1(float16_t* base, ptrdiff_t bstride, vfloat16m1_t v0, vfloat16m1_t v1, vfloat16m1_t v2, vfloat16m1_t v3, vfloat16m1_t v4, vfloat16m1_t v5, vfloat16m1_t v6, vfloat16m1_t v7, size_t vl)
{
    vfloat16m1x8_t _tmp = vcreate_f16m1x8(v0, v1, v2, v3, v4, v5, v6, v7);
    vssseg8e16_v_f16m1x8(base, bstride, _tmp, vl);
}

static inline void vssseg4e16_v_f16m1(float16_t* base, ptrdiff_t bstride, vfloat16m1_t v0, vfloat16m1_t v1, vfloat16m1_t v2, vfloat16m1_t v3, size_t vl)
{
    vfloat16m1x4_t _tmp = vcreate_f16m1x4(v0, v1, v2, v3);
    vssseg4e16_v_f16m1x4(base, bstride, _tmp, vl);
}

static inline void vssseg2e16_v_f16m1(float16_t* base, ptrdiff_t bstride, vfloat16m1_t v0, vfloat16m1_t v1, size_t vl)
{
    vfloat16m1x2_t _tmp = vcreate_f16m1x2(v0, v1);
    vssseg2e16_v_f16m1x2(base, bstride, _tmp, vl);
}

// f16m1, vlseg.v 8/4/2
static inline void vlseg8e16_v_f16m1(vfloat16m1_t* v0, vfloat16m1_t* v1, vfloat16m1_t* v2, vfloat16m1_t* v3, vfloat16m1_t* v4, vfloat16m1_t* v5, vfloat16m1_t* v6, vfloat16m1_t* v7, const float16_t* base, size_t vl)
{
    vfloat16m1x8_t _tmp = vlseg8e16_v_f16m1x8(base, vl);
    *v0 = vget_f16m1x8_f16m1(_tmp, 0);
    *v1 = vget_f16m1x8_f16m1(_tmp, 1);
    *v2 = vget_f16m1x8_f16m1(_tmp, 2);
    *v3 = vget_f16m1x8_f16m1(_tmp, 3);
    *v4 = vget_f16m1x8_f16m1(_tmp, 4);
    *v5 = vget_f16m1x8_f16m1(_tmp, 5);
    *v6 = vget_f16m1x8_f16m1(_tmp, 6);
    *v7 = vget_f16m1x8_f16m1(_tmp, 7);
}

static inline void vlseg4e16_v_f16m1(vfloat16m1_t* v0, vfloat16m1_t* v1, vfloat16m1_t* v2, vfloat16m1_t* v3, const float16_t* base, size_t vl)
{
    vfloat16m1x4_t _tmp = vlseg4e16_v_f16m1x4(base, vl);
    *v0 = vget_f16m1x4_f16m1(_tmp, 0);
    *v1 = vget_f16m1x4_f16m1(_tmp, 1);
    *v2 = vget_f16m1x4_f16m1(_tmp, 2);
    *v3 = vget_f16m1x4_f16m1(_tmp, 3);
}

static inline void vlseg2e16_v_f16m1(vfloat16m1_t* v0, vfloat16m1_t* v1, const float16_t* base, size_t vl)
{
    vfloat16m1x2_t _tmp = vlseg2e16_v_f16m1x2(base, vl);
    *v0 = vget_f16m1x2_f16m1(_tmp, 0);
    *v1 = vget_f16m1x2_f16m1(_tmp, 1);
}

// f16m2, vlseg.v, 4
static inline void vlseg4e16_v_f16m2(vfloat16m2_t* v0, vfloat16m2_t* v1, vfloat16m2_t* v2, vfloat16m2_t* v3, const float16_t* base, size_t vl)
{
    vfloat16m2x4_t _tmp = vlseg4e16_v_f16m2x4(base, vl);
    *v0 = vget_f16m2x4_f16m2(_tmp, 0);
    *v1 = vget_f16m2x4_f16m2(_tmp, 1);
    *v2 = vget_f16m2x4_f16m2(_tmp, 2);
    *v3 = vget_f16m2x4_f16m2(_tmp, 3);
}


// f16m4, vlseg.v, 2
static inline void vlseg2e16_v_f16m4(vfloat16m4_t* v0, vfloat16m4_t* v1, const float16_t* base, size_t vl)
{
    vfloat16m4x2_t _tmp = vlseg2e16_v_f16m4x2(base, vl);
    *v0 = vget_f16m4x2_f16m4(_tmp, 0);
    *v1 = vget_f16m4x2_f16m4(_tmp, 1);
}



#endif // __riscv_zfh

#endif // __riscv_zfh && __rvv_tuple

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
