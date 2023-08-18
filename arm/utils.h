
struct Mat
{
    int elempack = 1;
    int dims = 1;
    int h = 32;
    int w = 32;
    int cstep = 1;

    float* data;

    operator float *() const
    {
        return data;
    }
    
};

void init_Mat(Mat& mat) {
    float* data = mat;
    int w = mat.w;
    int h = mat.h;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            data[i * w + j] = i * w + j;
        }
    }
}

void print_Mat(Mat& mat) {
    float* data = mat;
    int w = mat.w;
    int h = mat.h;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            printf("%f ", data[i * w + j]);
        }
        printf("\n");
    } 
}

bool check_Mat(Mat& correct, Mat& test) {
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

static inline void transpose8x8_ps(float32x4_t& _r0l, float32x4_t& _r0h,
                                   float32x4_t& _r1l, float32x4_t& _r1h,
                                   float32x4_t& _r2l, float32x4_t& _r2h,
                                   float32x4_t& _r3l, float32x4_t& _r3h,
                                   float32x4_t& _r4l, float32x4_t& _r4h,
                                   float32x4_t& _r5l, float32x4_t& _r5h,
                                   float32x4_t& _r6l, float32x4_t& _r6h,
                                   float32x4_t& _r7l, float32x4_t& _r7h)
{
    float32x4x2_t _r01lz = vzipq_f32(_r0l, _r1l);
    float32x4x2_t _r23lz = vzipq_f32(_r2l, _r3l);
    float32x4x2_t _r01hz = vzipq_f32(_r0h, _r1h);
    float32x4x2_t _r23hz = vzipq_f32(_r2h, _r3h);
    float32x4x2_t _r45lz = vzipq_f32(_r4l, _r5l);
    float32x4x2_t _r67lz = vzipq_f32(_r6l, _r7l);
    float32x4x2_t _r45hz = vzipq_f32(_r4h, _r5h);
    float32x4x2_t _r67hz = vzipq_f32(_r6h, _r7h);
    _r0l = vcombine_f32(vget_low_f32(_r01lz.val[0]), vget_low_f32(_r23lz.val[0]));
    _r0h = vcombine_f32(vget_low_f32(_r45lz.val[0]), vget_low_f32(_r67lz.val[0]));
    _r1l = vcombine_f32(vget_high_f32(_r01lz.val[0]), vget_high_f32(_r23lz.val[0]));
    _r1h = vcombine_f32(vget_high_f32(_r45lz.val[0]), vget_high_f32(_r67lz.val[0]));
    _r2l = vcombine_f32(vget_low_f32(_r01lz.val[1]), vget_low_f32(_r23lz.val[1]));
    _r2h = vcombine_f32(vget_low_f32(_r45lz.val[1]), vget_low_f32(_r67lz.val[1]));
    _r3l = vcombine_f32(vget_high_f32(_r01lz.val[1]), vget_high_f32(_r23lz.val[1]));
    _r3h = vcombine_f32(vget_high_f32(_r45lz.val[1]), vget_high_f32(_r67lz.val[1]));
    _r4l = vcombine_f32(vget_low_f32(_r01hz.val[0]), vget_low_f32(_r23hz.val[0]));
    _r4h = vcombine_f32(vget_low_f32(_r45hz.val[0]), vget_low_f32(_r67hz.val[0]));
    _r5l = vcombine_f32(vget_high_f32(_r01hz.val[0]), vget_high_f32(_r23hz.val[0]));
    _r5h = vcombine_f32(vget_high_f32(_r45hz.val[0]), vget_high_f32(_r67hz.val[0]));
    _r6l = vcombine_f32(vget_low_f32(_r01hz.val[1]), vget_low_f32(_r23hz.val[1]));
    _r6h = vcombine_f32(vget_low_f32(_r45hz.val[1]), vget_low_f32(_r67hz.val[1]));
    _r7l = vcombine_f32(vget_high_f32(_r01hz.val[1]), vget_high_f32(_r23hz.val[1]));
    _r7h = vcombine_f32(vget_high_f32(_r45hz.val[1]), vget_high_f32(_r67hz.val[1]));
}

static inline void transpose4x4_ps(float32x4_t& _r0, float32x4_t& _r1, float32x4_t& _r2, float32x4_t& _r3)
{
    float32x4x2_t _r01z = vzipq_f32(_r0, _r1);
    float32x4x2_t _r23z = vzipq_f32(_r2, _r3);
    _r0 = vcombine_f32(vget_low_f32(_r01z.val[0]), vget_low_f32(_r23z.val[0]));
    _r1 = vcombine_f32(vget_high_f32(_r01z.val[0]), vget_high_f32(_r23z.val[0]));
    _r2 = vcombine_f32(vget_low_f32(_r01z.val[1]), vget_low_f32(_r23z.val[1]));
    _r3 = vcombine_f32(vget_high_f32(_r01z.val[1]), vget_high_f32(_r23z.val[1]));
}


