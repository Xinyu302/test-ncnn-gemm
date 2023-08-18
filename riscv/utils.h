
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

vfloat32m1 vfmaq_laneq_f32(vfloat32m1 sum, vfloat32m1 a, vfloat32m1 b, int lane) {
    float t[4];
    vse32_v_f32m1(t, b, 4);
    vfloat32m1 ret = vfmadd_vf_f32m1(
            a, t[lane], sum, 4);
    return ret;     
}
