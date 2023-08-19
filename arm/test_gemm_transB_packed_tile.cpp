#include <arm_neon.h>
#include <stdio.h>
#include "utils.h"

static void gemm_transB_packed_tile(const Mat& AT_tile, const Mat& BT_tile, const Mat& CT_tile, Mat& topT_tile, Mat& top_blob, int broadcast_type_C, int i, int max_ii, int j, int max_jj, int k, int max_kk, bool k_end)
{
    const int out_elempack = top_blob.elempack;
    const int out_hstep = top_blob.dims == 3 ? (int)top_blob.cstep : top_blob.w;

    const float* pAT = AT_tile;
    const float* pBT = BT_tile;
    const float* pC = CT_tile;

    float* outptr = topT_tile;

    int ii = 0;
#if __ARM_NEON
#if __aarch64__
    for (; ii + 7 < max_ii; ii += 8)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum20;
            float32x4_t _sum21;
            float32x4_t _sum30;
            float32x4_t _sum31;
            float32x4_t _sum40;
            float32x4_t _sum41;
            float32x4_t _sum50;
            float32x4_t _sum51;
            float32x4_t _sum60;
            float32x4_t _sum61;
            float32x4_t _sum70;
            float32x4_t _sum71;
            float32x4_t _sum80;
            float32x4_t _sum81;
            float32x4_t _sum90;
            float32x4_t _sum91;
            float32x4_t _suma0;
            float32x4_t _suma1;
            float32x4_t _sumb0;
            float32x4_t _sumb1;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);
                _sum20 = vdupq_n_f32(0.f);
                _sum21 = vdupq_n_f32(0.f);
                _sum30 = vdupq_n_f32(0.f);
                _sum31 = vdupq_n_f32(0.f);
                _sum40 = vdupq_n_f32(0.f);
                _sum41 = vdupq_n_f32(0.f);
                _sum50 = vdupq_n_f32(0.f);
                _sum51 = vdupq_n_f32(0.f);
                _sum60 = vdupq_n_f32(0.f);
                _sum61 = vdupq_n_f32(0.f);
                _sum70 = vdupq_n_f32(0.f);
                _sum71 = vdupq_n_f32(0.f);
                _sum80 = vdupq_n_f32(0.f);
                _sum81 = vdupq_n_f32(0.f);
                _sum90 = vdupq_n_f32(0.f);
                _sum91 = vdupq_n_f32(0.f);
                _suma0 = vdupq_n_f32(0.f);
                _suma1 = vdupq_n_f32(0.f);
                _sumb0 = vdupq_n_f32(0.f);
                _sumb1 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[0]);
                        _sum11 = vdupq_n_f32(pC[0]);
                        _sum20 = vdupq_n_f32(pC[0]);
                        _sum21 = vdupq_n_f32(pC[0]);
                        _sum30 = vdupq_n_f32(pC[0]);
                        _sum31 = vdupq_n_f32(pC[0]);
                        _sum40 = vdupq_n_f32(pC[0]);
                        _sum41 = vdupq_n_f32(pC[0]);
                        _sum50 = vdupq_n_f32(pC[0]);
                        _sum51 = vdupq_n_f32(pC[0]);
                        _sum60 = vdupq_n_f32(pC[0]);
                        _sum61 = vdupq_n_f32(pC[0]);
                        _sum70 = vdupq_n_f32(pC[0]);
                        _sum71 = vdupq_n_f32(pC[0]);
                        _sum80 = vdupq_n_f32(pC[0]);
                        _sum81 = vdupq_n_f32(pC[0]);
                        _sum90 = vdupq_n_f32(pC[0]);
                        _sum91 = vdupq_n_f32(pC[0]);
                        _suma0 = vdupq_n_f32(pC[0]);
                        _suma1 = vdupq_n_f32(pC[0]);
                        _sumb0 = vdupq_n_f32(pC[0]);
                        _sumb1 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        _sum20 = _sum00;
                        _sum21 = _sum01;
                        _sum30 = _sum00;
                        _sum31 = _sum01;
                        _sum40 = _sum00;
                        _sum41 = _sum01;
                        _sum50 = _sum00;
                        _sum51 = _sum01;
                        _sum60 = _sum00;
                        _sum61 = _sum01;
                        _sum70 = _sum00;
                        _sum71 = _sum01;
                        _sum80 = _sum00;
                        _sum81 = _sum01;
                        _sum90 = _sum00;
                        _sum91 = _sum01;
                        _suma0 = _sum00;
                        _suma1 = _sum01;
                        _sumb0 = _sum00;
                        _sumb1 = _sum01;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4 * 1);
                        _sum10 = vld1q_f32(pC + 4 * 2);
                        _sum11 = vld1q_f32(pC + 4 * 3);
                        _sum20 = vld1q_f32(pC + 4 * 4);
                        _sum21 = vld1q_f32(pC + 4 * 5);
                        _sum30 = vld1q_f32(pC + 4 * 6);
                        _sum31 = vld1q_f32(pC + 4 * 7);
                        _sum40 = vld1q_f32(pC + 4 * 8);
                        _sum41 = vld1q_f32(pC + 4 * 9);
                        _sum50 = vld1q_f32(pC + 4 * 10);
                        _sum51 = vld1q_f32(pC + 4 * 11);
                        _sum60 = vld1q_f32(pC + 4 * 12);
                        _sum61 = vld1q_f32(pC + 4 * 13);
                        _sum70 = vld1q_f32(pC + 4 * 14);
                        _sum71 = vld1q_f32(pC + 4 * 15);
                        _sum80 = vld1q_f32(pC + 4 * 16);
                        _sum81 = vld1q_f32(pC + 4 * 17);
                        _sum90 = vld1q_f32(pC + 4 * 18);
                        _sum91 = vld1q_f32(pC + 4 * 19);
                        _suma0 = vld1q_f32(pC + 4 * 20);
                        _suma1 = vld1q_f32(pC + 4 * 21);
                        _sumb0 = vld1q_f32(pC + 4 * 22);
                        _sumb1 = vld1q_f32(pC + 4 * 23);
                        pC += 96;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum20 = vdupq_n_f32(pC[2]);
                        _sum30 = vdupq_n_f32(pC[3]);
                        _sum40 = vdupq_n_f32(pC[4]);
                        _sum50 = vdupq_n_f32(pC[5]);
                        _sum60 = vdupq_n_f32(pC[6]);
                        _sum70 = vdupq_n_f32(pC[7]);
                        _sum80 = vdupq_n_f32(pC[8]);
                        _sum90 = vdupq_n_f32(pC[9]);
                        _suma0 = vdupq_n_f32(pC[10]);
                        _sumb0 = vdupq_n_f32(pC[11]);
                        _sum01 = _sum00;
                        _sum11 = _sum10;
                        _sum21 = _sum20;
                        _sum31 = _sum30;
                        _sum41 = _sum40;
                        _sum51 = _sum50;
                        _sum61 = _sum60;
                        _sum71 = _sum70;
                        _sum81 = _sum80;
                        _sum91 = _sum90;
                        _suma1 = _suma0;
                        _sumb1 = _sumb0;
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
                _sum20 = vld1q_f32(outptr + 4 * 4);
                _sum21 = vld1q_f32(outptr + 4 * 5);
                _sum30 = vld1q_f32(outptr + 4 * 6);
                _sum31 = vld1q_f32(outptr + 4 * 7);
                _sum40 = vld1q_f32(outptr + 4 * 8);
                _sum41 = vld1q_f32(outptr + 4 * 9);
                _sum50 = vld1q_f32(outptr + 4 * 10);
                _sum51 = vld1q_f32(outptr + 4 * 11);
                _sum60 = vld1q_f32(outptr + 4 * 12);
                _sum61 = vld1q_f32(outptr + 4 * 13);
                _sum70 = vld1q_f32(outptr + 4 * 14);
                _sum71 = vld1q_f32(outptr + 4 * 15);
                _sum80 = vld1q_f32(outptr + 4 * 16);
                _sum81 = vld1q_f32(outptr + 4 * 17);
                _sum90 = vld1q_f32(outptr + 4 * 18);
                _sum91 = vld1q_f32(outptr + 4 * 19);
                _suma0 = vld1q_f32(outptr + 4 * 20);
                _suma1 = vld1q_f32(outptr + 4 * 21);
                _sumb0 = vld1q_f32(outptr + 4 * 22);
                _sumb1 = vld1q_f32(outptr + 4 * 23);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                float32x4_t _pA0 = vld1q_f32(pA);
                float32x4_t _pA1 = vld1q_f32(pA + 4);

                float32x4_t _pB0 = vld1q_f32(pB);
                float32x4_t _pB1 = vld1q_f32(pB + 4);
                float32x4_t _pB2 = vld1q_f32(pB + 8);

                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32(_sum71, _pA1, _pB1, 3);
                _sum80 = vfmaq_laneq_f32(_sum80, _pA0, _pB2, 0);
                _sum81 = vfmaq_laneq_f32(_sum81, _pA1, _pB2, 0);
                _sum90 = vfmaq_laneq_f32(_sum90, _pA0, _pB2, 1);
                _sum91 = vfmaq_laneq_f32(_sum91, _pA1, _pB2, 1);
                _suma0 = vfmaq_laneq_f32(_suma0, _pA0, _pB2, 2);
                _suma1 = vfmaq_laneq_f32(_suma1, _pA1, _pB2, 2);
                _sumb0 = vfmaq_laneq_f32(_sumb0, _pA0, _pB2, 3);
                _sumb1 = vfmaq_laneq_f32(_sumb1, _pA1, _pB2, 3);

                pA += 8;
                pB += 12;

                _pA0 = vld1q_f32(pA);
                _pA1 = vld1q_f32(pA + 4);

                _pB0 = vld1q_f32(pB);
                _pB1 = vld1q_f32(pB + 4);
                _pB2 = vld1q_f32(pB + 8);

                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32(_sum71, _pA1, _pB1, 3);
                _sum80 = vfmaq_laneq_f32(_sum80, _pA0, _pB2, 0);
                _sum81 = vfmaq_laneq_f32(_sum81, _pA1, _pB2, 0);
                _sum90 = vfmaq_laneq_f32(_sum90, _pA0, _pB2, 1);
                _sum91 = vfmaq_laneq_f32(_sum91, _pA1, _pB2, 1);
                _suma0 = vfmaq_laneq_f32(_suma0, _pA0, _pB2, 2);
                _suma1 = vfmaq_laneq_f32(_suma1, _pA1, _pB2, 2);
                _sumb0 = vfmaq_laneq_f32(_sumb0, _pA0, _pB2, 3);
                _sumb1 = vfmaq_laneq_f32(_sumb1, _pA1, _pB2, 3);

                pA += 8;
                pB += 12;

                _pA0 = vld1q_f32(pA);
                _pA1 = vld1q_f32(pA + 4);

                _pB0 = vld1q_f32(pB);
                _pB1 = vld1q_f32(pB + 4);
                _pB2 = vld1q_f32(pB + 8);

                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32(_sum71, _pA1, _pB1, 3);
                _sum80 = vfmaq_laneq_f32(_sum80, _pA0, _pB2, 0);
                _sum81 = vfmaq_laneq_f32(_sum81, _pA1, _pB2, 0);
                _sum90 = vfmaq_laneq_f32(_sum90, _pA0, _pB2, 1);
                _sum91 = vfmaq_laneq_f32(_sum91, _pA1, _pB2, 1);
                _suma0 = vfmaq_laneq_f32(_suma0, _pA0, _pB2, 2);
                _suma1 = vfmaq_laneq_f32(_suma1, _pA1, _pB2, 2);
                _sumb0 = vfmaq_laneq_f32(_sumb0, _pA0, _pB2, 3);
                _sumb1 = vfmaq_laneq_f32(_sumb1, _pA1, _pB2, 3);

                pA += 8;
                pB += 12;

                _pA0 = vld1q_f32(pA);
                _pA1 = vld1q_f32(pA + 4);

                _pB0 = vld1q_f32(pB);
                _pB1 = vld1q_f32(pB + 4);
                _pB2 = vld1q_f32(pB + 8);

                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32(_sum71, _pA1, _pB1, 3);
                _sum80 = vfmaq_laneq_f32(_sum80, _pA0, _pB2, 0);
                _sum81 = vfmaq_laneq_f32(_sum81, _pA1, _pB2, 0);
                _sum90 = vfmaq_laneq_f32(_sum90, _pA0, _pB2, 1);
                _sum91 = vfmaq_laneq_f32(_sum91, _pA1, _pB2, 1);
                _suma0 = vfmaq_laneq_f32(_suma0, _pA0, _pB2, 2);
                _suma1 = vfmaq_laneq_f32(_suma1, _pA1, _pB2, 2);
                _sumb0 = vfmaq_laneq_f32(_sumb0, _pA0, _pB2, 3);
                _sumb1 = vfmaq_laneq_f32(_sumb1, _pA1, _pB2, 3);

                pA += 8;
                pB += 12;
            }
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = vld1q_f32(pA);
                float32x4_t _pA1 = vld1q_f32(pA + 4);

                float32x4_t _pB0 = vld1q_f32(pB);
                float32x4_t _pB1 = vld1q_f32(pB + 4);
                float32x4_t _pB2 = vld1q_f32(pB + 8);

                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32(_sum71, _pA1, _pB1, 3);
                _sum80 = vfmaq_laneq_f32(_sum80, _pA0, _pB2, 0);
                _sum81 = vfmaq_laneq_f32(_sum81, _pA1, _pB2, 0);
                _sum90 = vfmaq_laneq_f32(_sum90, _pA0, _pB2, 1);
                _sum91 = vfmaq_laneq_f32(_sum91, _pA1, _pB2, 1);
                _suma0 = vfmaq_laneq_f32(_suma0, _pA0, _pB2, 2);
                _suma1 = vfmaq_laneq_f32(_suma1, _pA1, _pB2, 2);
                _sumb0 = vfmaq_laneq_f32(_sumb0, _pA0, _pB2, 3);
                _sumb1 = vfmaq_laneq_f32(_sumb1, _pA1, _pB2, 3);

                pA += 8;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum10);
                    vst1q_f32(outptr0 + 4 * 2, _sum20);
                    vst1q_f32(outptr0 + 4 * 3, _sum30);
                    vst1q_f32(outptr0 + 4 * 4, _sum40);
                    vst1q_f32(outptr0 + 4 * 5, _sum50);
                    vst1q_f32(outptr0 + 4 * 6, _sum60);
                    vst1q_f32(outptr0 + 4 * 7, _sum70);
                    vst1q_f32(outptr0 + 4 * 8, _sum80);
                    vst1q_f32(outptr0 + 4 * 9, _sum90);
                    vst1q_f32(outptr0 + 4 * 10, _suma0);
                    vst1q_f32(outptr0 + 4 * 11, _sumb0);

                    vst1q_f32(outptr0 + out_hstep * 4, _sum01);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4, _sum11);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 2, _sum21);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 3, _sum31);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 4, _sum41);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 5, _sum51);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 6, _sum61);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 7, _sum71);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 8, _sum81);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 9, _sum91);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 10, _suma1);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 11, _sumb1);

                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose8x12_ps(_sum00, _sum01, _sum10, _sum11, _sum20, _sum21, _sum30, _sum31, _sum40, _sum41, _sum50, _sum51, _sum60, _sum61, _sum70, _sum71, _sum80, _sum81, _sum90, _sum91, _suma0, _suma1, _sumb0, _sumb1);

                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum01);
                    vst1q_f32(outptr0 + 8, _sum10);
                    vst1q_f32(outptr0 + out_hstep, _sum11);
                    vst1q_f32(outptr0 + out_hstep + 4, _sum20);
                    vst1q_f32(outptr0 + out_hstep + 8, _sum21);
                    vst1q_f32(outptr0 + out_hstep * 2, _sum30);
                    vst1q_f32(outptr0 + out_hstep * 2 + 4, _sum31);
                    vst1q_f32(outptr0 + out_hstep * 2 + 8, _sum40);
                    vst1q_f32(outptr0 + out_hstep * 3, _sum41);
                    vst1q_f32(outptr0 + out_hstep * 3 + 4, _sum50);
                    vst1q_f32(outptr0 + out_hstep * 3 + 8, _sum51);
                    vst1q_f32(outptr0 + out_hstep * 4, _sum60);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4, _sum61);
                    vst1q_f32(outptr0 + out_hstep * 4 + 8, _sum70);
                    vst1q_f32(outptr0 + out_hstep * 5, _sum71);
                    vst1q_f32(outptr0 + out_hstep * 5 + 4, _sum80);
                    vst1q_f32(outptr0 + out_hstep * 5 + 8, _sum81);
                    vst1q_f32(outptr0 + out_hstep * 6, _sum90);
                    vst1q_f32(outptr0 + out_hstep * 6 + 4, _sum91);
                    vst1q_f32(outptr0 + out_hstep * 6 + 8, _suma0);
                    vst1q_f32(outptr0 + out_hstep * 7, _suma1);
                    vst1q_f32(outptr0 + out_hstep * 7 + 4, _sumb0);
                    vst1q_f32(outptr0 + out_hstep * 7 + 8, _sumb1);

                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
                vst1q_f32(outptr + 4 * 4, _sum20);
                vst1q_f32(outptr + 4 * 5, _sum21);
                vst1q_f32(outptr + 4 * 6, _sum30);
                vst1q_f32(outptr + 4 * 7, _sum31);
                vst1q_f32(outptr + 4 * 8, _sum40);
                vst1q_f32(outptr + 4 * 9, _sum41);
                vst1q_f32(outptr + 4 * 10, _sum50);
                vst1q_f32(outptr + 4 * 11, _sum51);
                vst1q_f32(outptr + 4 * 12, _sum60);
                vst1q_f32(outptr + 4 * 13, _sum61);
                vst1q_f32(outptr + 4 * 14, _sum70);
                vst1q_f32(outptr + 4 * 15, _sum71);
                vst1q_f32(outptr + 4 * 16, _sum80);
                vst1q_f32(outptr + 4 * 17, _sum81);
                vst1q_f32(outptr + 4 * 18, _sum90);
                vst1q_f32(outptr + 4 * 19, _sum91);
                vst1q_f32(outptr + 4 * 20, _suma0);
                vst1q_f32(outptr + 4 * 21, _suma1);
                vst1q_f32(outptr + 4 * 22, _sumb0);
                vst1q_f32(outptr + 4 * 23, _sumb1);
            }

            outptr += 96;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum20;
            float32x4_t _sum21;
            float32x4_t _sum30;
            float32x4_t _sum31;
            float32x4_t _sum40;
            float32x4_t _sum41;
            float32x4_t _sum50;
            float32x4_t _sum51;
            float32x4_t _sum60;
            float32x4_t _sum61;
            float32x4_t _sum70;
            float32x4_t _sum71;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);
                _sum20 = vdupq_n_f32(0.f);
                _sum21 = vdupq_n_f32(0.f);
                _sum30 = vdupq_n_f32(0.f);
                _sum31 = vdupq_n_f32(0.f);
                _sum40 = vdupq_n_f32(0.f);
                _sum41 = vdupq_n_f32(0.f);
                _sum50 = vdupq_n_f32(0.f);
                _sum51 = vdupq_n_f32(0.f);
                _sum60 = vdupq_n_f32(0.f);
                _sum61 = vdupq_n_f32(0.f);
                _sum70 = vdupq_n_f32(0.f);
                _sum71 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[0]);
                        _sum11 = vdupq_n_f32(pC[0]);
                        _sum20 = vdupq_n_f32(pC[0]);
                        _sum21 = vdupq_n_f32(pC[0]);
                        _sum30 = vdupq_n_f32(pC[0]);
                        _sum31 = vdupq_n_f32(pC[0]);
                        _sum40 = vdupq_n_f32(pC[0]);
                        _sum41 = vdupq_n_f32(pC[0]);
                        _sum50 = vdupq_n_f32(pC[0]);
                        _sum51 = vdupq_n_f32(pC[0]);
                        _sum60 = vdupq_n_f32(pC[0]);
                        _sum61 = vdupq_n_f32(pC[0]);
                        _sum70 = vdupq_n_f32(pC[0]);
                        _sum71 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        _sum20 = _sum00;
                        _sum21 = _sum01;
                        _sum30 = _sum00;
                        _sum31 = _sum01;
                        _sum40 = _sum00;
                        _sum41 = _sum01;
                        _sum50 = _sum00;
                        _sum51 = _sum01;
                        _sum60 = _sum00;
                        _sum61 = _sum01;
                        _sum70 = _sum00;
                        _sum71 = _sum01;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4 * 1);
                        _sum10 = vld1q_f32(pC + 4 * 2);
                        _sum11 = vld1q_f32(pC + 4 * 3);
                        _sum20 = vld1q_f32(pC + 4 * 4);
                        _sum21 = vld1q_f32(pC + 4 * 5);
                        _sum30 = vld1q_f32(pC + 4 * 6);
                        _sum31 = vld1q_f32(pC + 4 * 7);
                        _sum40 = vld1q_f32(pC + 4 * 8);
                        _sum41 = vld1q_f32(pC + 4 * 9);
                        _sum50 = vld1q_f32(pC + 4 * 10);
                        _sum51 = vld1q_f32(pC + 4 * 11);
                        _sum60 = vld1q_f32(pC + 4 * 12);
                        _sum61 = vld1q_f32(pC + 4 * 13);
                        _sum70 = vld1q_f32(pC + 4 * 14);
                        _sum71 = vld1q_f32(pC + 4 * 15);
                        pC += 64;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum20 = vdupq_n_f32(pC[2]);
                        _sum30 = vdupq_n_f32(pC[3]);
                        _sum40 = vdupq_n_f32(pC[4]);
                        _sum50 = vdupq_n_f32(pC[5]);
                        _sum60 = vdupq_n_f32(pC[6]);
                        _sum70 = vdupq_n_f32(pC[7]);
                        _sum01 = _sum00;
                        _sum11 = _sum10;
                        _sum21 = _sum20;
                        _sum31 = _sum30;
                        _sum41 = _sum40;
                        _sum51 = _sum50;
                        _sum61 = _sum60;
                        _sum71 = _sum70;
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
                _sum20 = vld1q_f32(outptr + 4 * 4);
                _sum21 = vld1q_f32(outptr + 4 * 5);
                _sum30 = vld1q_f32(outptr + 4 * 6);
                _sum31 = vld1q_f32(outptr + 4 * 7);
                _sum40 = vld1q_f32(outptr + 4 * 8);
                _sum41 = vld1q_f32(outptr + 4 * 9);
                _sum50 = vld1q_f32(outptr + 4 * 10);
                _sum51 = vld1q_f32(outptr + 4 * 11);
                _sum60 = vld1q_f32(outptr + 4 * 12);
                _sum61 = vld1q_f32(outptr + 4 * 13);
                _sum70 = vld1q_f32(outptr + 4 * 14);
                _sum71 = vld1q_f32(outptr + 4 * 15);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = vld1q_f32(pA);
                float32x4_t _pA1 = vld1q_f32(pA + 4);

                float32x4_t _pB0 = vld1q_f32(pB);
                float32x4_t _pB1 = vld1q_f32(pB + 4);

                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32(_sum71, _pA1, _pB1, 3);

                pA += 8;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum10);
                    vst1q_f32(outptr0 + 4 * 2, _sum20);
                    vst1q_f32(outptr0 + 4 * 3, _sum30);
                    vst1q_f32(outptr0 + 4 * 4, _sum40);
                    vst1q_f32(outptr0 + 4 * 5, _sum50);
                    vst1q_f32(outptr0 + 4 * 6, _sum60);
                    vst1q_f32(outptr0 + 4 * 7, _sum70);

                    vst1q_f32(outptr0 + out_hstep * 4, _sum01);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4, _sum11);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 2, _sum21);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 3, _sum31);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 4, _sum41);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 5, _sum51);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 6, _sum61);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 7, _sum71);

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose8x8_ps(_sum00, _sum01, _sum10, _sum11, _sum20, _sum21, _sum30, _sum31, _sum40, _sum41, _sum50, _sum51, _sum60, _sum61, _sum70, _sum71);

                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum01);
                    vst1q_f32(outptr0 + out_hstep, _sum10);
                    vst1q_f32(outptr0 + out_hstep + 4, _sum11);
                    vst1q_f32(outptr0 + out_hstep * 2, _sum20);
                    vst1q_f32(outptr0 + out_hstep * 2 + 4, _sum21);
                    vst1q_f32(outptr0 + out_hstep * 3, _sum30);
                    vst1q_f32(outptr0 + out_hstep * 3 + 4, _sum31);
                    vst1q_f32(outptr0 + out_hstep * 4, _sum40);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4, _sum41);
                    vst1q_f32(outptr0 + out_hstep * 5, _sum50);
                    vst1q_f32(outptr0 + out_hstep * 5 + 4, _sum51);
                    vst1q_f32(outptr0 + out_hstep * 6, _sum60);
                    vst1q_f32(outptr0 + out_hstep * 6 + 4, _sum61);
                    vst1q_f32(outptr0 + out_hstep * 7, _sum70);
                    vst1q_f32(outptr0 + out_hstep * 7 + 4, _sum71);

                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
                vst1q_f32(outptr + 4 * 4, _sum20);
                vst1q_f32(outptr + 4 * 5, _sum21);
                vst1q_f32(outptr + 4 * 6, _sum30);
                vst1q_f32(outptr + 4 * 7, _sum31);
                vst1q_f32(outptr + 4 * 8, _sum40);
                vst1q_f32(outptr + 4 * 9, _sum41);
                vst1q_f32(outptr + 4 * 10, _sum50);
                vst1q_f32(outptr + 4 * 11, _sum51);
                vst1q_f32(outptr + 4 * 12, _sum60);
                vst1q_f32(outptr + 4 * 13, _sum61);
                vst1q_f32(outptr + 4 * 14, _sum70);
                vst1q_f32(outptr + 4 * 15, _sum71);
            }

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum20;
            float32x4_t _sum21;
            float32x4_t _sum30;
            float32x4_t _sum31;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);
                _sum20 = vdupq_n_f32(0.f);
                _sum21 = vdupq_n_f32(0.f);
                _sum30 = vdupq_n_f32(0.f);
                _sum31 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[0]);
                        _sum11 = vdupq_n_f32(pC[0]);
                        _sum20 = vdupq_n_f32(pC[0]);
                        _sum21 = vdupq_n_f32(pC[0]);
                        _sum30 = vdupq_n_f32(pC[0]);
                        _sum31 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        _sum20 = _sum00;
                        _sum21 = _sum01;
                        _sum30 = _sum00;
                        _sum31 = _sum01;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4 * 1);
                        _sum10 = vld1q_f32(pC + 4 * 2);
                        _sum11 = vld1q_f32(pC + 4 * 3);
                        _sum20 = vld1q_f32(pC + 4 * 4);
                        _sum21 = vld1q_f32(pC + 4 * 5);
                        _sum30 = vld1q_f32(pC + 4 * 6);
                        _sum31 = vld1q_f32(pC + 4 * 7);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum20 = vdupq_n_f32(pC[2]);
                        _sum30 = vdupq_n_f32(pC[3]);
                        _sum01 = _sum00;
                        _sum11 = _sum10;
                        _sum21 = _sum20;
                        _sum31 = _sum30;
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
                _sum20 = vld1q_f32(outptr + 4 * 4);
                _sum21 = vld1q_f32(outptr + 4 * 5);
                _sum30 = vld1q_f32(outptr + 4 * 6);
                _sum31 = vld1q_f32(outptr + 4 * 7);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = vld1q_f32(pA);
                float32x4_t _pA1 = vld1q_f32(pA + 4);

                float32x4_t _pB0 = vld1q_f32(pB);

                _sum00 = vfmaq_laneq_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32(_sum31, _pA1, _pB0, 3);

                pA += 8;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum10);
                    vst1q_f32(outptr0 + 4 * 2, _sum20);
                    vst1q_f32(outptr0 + 4 * 3, _sum30);

                    vst1q_f32(outptr0 + out_hstep * 4, _sum01);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4, _sum11);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 2, _sum21);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4 * 3, _sum31);

                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose8x4_ps(_sum00, _sum01, _sum10, _sum11, _sum20, _sum21, _sum30, _sum31);

                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + out_hstep * 1, _sum01);
                    vst1q_f32(outptr0 + out_hstep * 2, _sum10);
                    vst1q_f32(outptr0 + out_hstep * 3, _sum11);
                    vst1q_f32(outptr0 + out_hstep * 4, _sum20);
                    vst1q_f32(outptr0 + out_hstep * 5, _sum21);
                    vst1q_f32(outptr0 + out_hstep * 6, _sum30);
                    vst1q_f32(outptr0 + out_hstep * 7, _sum31);

                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
                vst1q_f32(outptr + 4 * 4, _sum20);
                vst1q_f32(outptr + 4 * 5, _sum21);
                vst1q_f32(outptr + 4 * 6, _sum30);
                vst1q_f32(outptr + 4 * 7, _sum31);
            }

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[0]);
                        _sum11 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4 * 1);
                        _sum10 = vld1q_f32(pC + 4 * 2);
                        _sum11 = vld1q_f32(pC + 4 * 3);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum01 = _sum00;
                        _sum11 = _sum10;
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
                _sum10 = vld1q_f32(outptr + 4 * 2);
                _sum11 = vld1q_f32(outptr + 4 * 3);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = vld1q_f32(pA);
                float32x4_t _pA1 = vld1q_f32(pA + 4);

                float32x2_t _pB0 = vld1_f32(pB);

                _sum00 = vfmaq_lane_f32(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_lane_f32(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_lane_f32(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_lane_f32(_sum11, _pA1, _pB0, 1);

                pA += 8;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum10);

                    vst1q_f32(outptr0 + out_hstep * 4, _sum01);
                    vst1q_f32(outptr0 + out_hstep * 4 + 4, _sum11);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    float sum1[8];
                    vst1q_f32(sum0, _sum00);
                    vst1q_f32(sum0 + 4, _sum01);
                    vst1q_f32(sum1, _sum10);
                    vst1q_f32(sum1 + 4, _sum11);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];

                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0[out_hstep * 4 + 1] = sum1[4];
                    outptr0[out_hstep * 5 + 1] = sum1[5];
                    outptr0[out_hstep * 6 + 1] = sum1[6];
                    outptr0[out_hstep * 7 + 1] = sum1[7];
                    outptr0 += 2;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
                vst1q_f32(outptr + 4 * 2, _sum10);
                vst1q_f32(outptr + 4 * 3, _sum11);
            }

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = _sum00;
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum00 = vld1q_f32(outptr);
                _sum01 = vld1q_f32(outptr + 4 * 1);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA0 = vld1q_f32(pA);
                float32x4_t _pA1 = vld1q_f32(pA + 4);

                float32x4_t _pB = vld1q_dup_f32(pB);

                _sum00 = vfmaq_f32(_sum00, _pA0, _pB);
                _sum01 = vfmaq_f32(_sum01, _pA1, _pB);

                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + out_hstep * 4, _sum01);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    vst1q_f32(sum0, _sum00);
                    vst1q_f32(sum0 + 4, _sum01);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep * 1] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[out_hstep * 4] = sum0[4];
                    outptr0[out_hstep * 5] = sum0[5];
                    outptr0[out_hstep * 6] = sum0[6];
                    outptr0[out_hstep * 7] = sum0[7];
                    outptr0++;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum00);
                vst1q_f32(outptr + 4, _sum01);
            }

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
#endif // __aarch64__
    for (; ii + 3 < max_ii; ii += 4)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j * out_elempack;

        const float* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;
            float32x4_t _sum4;
            float32x4_t _sum5;
            float32x4_t _sum6;
            float32x4_t _sum7;
            float32x4_t _sum8;
            float32x4_t _sum9;
            float32x4_t _suma;
            float32x4_t _sumb;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
                _sum2 = vdupq_n_f32(0.f);
                _sum3 = vdupq_n_f32(0.f);
                _sum4 = vdupq_n_f32(0.f);
                _sum5 = vdupq_n_f32(0.f);
                _sum6 = vdupq_n_f32(0.f);
                _sum7 = vdupq_n_f32(0.f);
                _sum8 = vdupq_n_f32(0.f);
                _sum9 = vdupq_n_f32(0.f);
                _suma = vdupq_n_f32(0.f);
                _sumb = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[0]);
                        _sum2 = vdupq_n_f32(pC[0]);
                        _sum3 = vdupq_n_f32(pC[0]);
                        _sum4 = vdupq_n_f32(pC[0]);
                        _sum5 = vdupq_n_f32(pC[0]);
                        _sum6 = vdupq_n_f32(pC[0]);
                        _sum7 = vdupq_n_f32(pC[0]);
                        _sum8 = vdupq_n_f32(pC[0]);
                        _sum9 = vdupq_n_f32(pC[0]);
                        _suma = vdupq_n_f32(pC[0]);
                        _sumb = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                        _sum8 = _sum0;
                        _sum9 = _sum0;
                        _suma = _sum0;
                        _sumb = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        _sum2 = vld1q_f32(pC + 8);
                        _sum3 = vld1q_f32(pC + 12);
                        _sum4 = vld1q_f32(pC + 16);
                        _sum5 = vld1q_f32(pC + 20);
                        _sum6 = vld1q_f32(pC + 24);
                        _sum7 = vld1q_f32(pC + 28);
                        _sum8 = vld1q_f32(pC + 32);
                        _sum9 = vld1q_f32(pC + 36);
                        _suma = vld1q_f32(pC + 40);
                        _sumb = vld1q_f32(pC + 44);
                        pC += 48;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[1]);
                        _sum2 = vdupq_n_f32(pC[2]);
                        _sum3 = vdupq_n_f32(pC[3]);
                        _sum4 = vdupq_n_f32(pC[4]);
                        _sum5 = vdupq_n_f32(pC[5]);
                        _sum6 = vdupq_n_f32(pC[6]);
                        _sum7 = vdupq_n_f32(pC[7]);
                        _sum8 = vdupq_n_f32(pC[8]);
                        _sum9 = vdupq_n_f32(pC[9]);
                        _suma = vdupq_n_f32(pC[10]);
                        _sumb = vdupq_n_f32(pC[11]);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4 * 1);
                _sum2 = vld1q_f32(outptr + 4 * 2);
                _sum3 = vld1q_f32(outptr + 4 * 3);
                _sum4 = vld1q_f32(outptr + 4 * 4);
                _sum5 = vld1q_f32(outptr + 4 * 5);
                _sum6 = vld1q_f32(outptr + 4 * 6);
                _sum7 = vld1q_f32(outptr + 4 * 7);
                _sum8 = vld1q_f32(outptr + 4 * 8);
                _sum9 = vld1q_f32(outptr + 4 * 9);
                _suma = vld1q_f32(outptr + 4 * 10);
                _sumb = vld1q_f32(outptr + 4 * 11);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = vld1q_f32(pA);
                float32x4_t _pB0 = vld1q_f32(pB);
                float32x4_t _pB1 = vld1q_f32(pB + 4);
                float32x4_t _pB2 = vld1q_f32(pB + 8);

                _sum0 = vfmaq_laneq_f32(_sum0, _pA, _pB0, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _pA, _pB0, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _pA, _pB0, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _pA, _pB0, 3);
                _sum4 = vfmaq_laneq_f32(_sum4, _pA, _pB1, 0);
                _sum5 = vfmaq_laneq_f32(_sum5, _pA, _pB1, 1);
                _sum6 = vfmaq_laneq_f32(_sum6, _pA, _pB1, 2);
                _sum7 = vfmaq_laneq_f32(_sum7, _pA, _pB1, 3);
                _sum8 = vfmaq_laneq_f32(_sum8, _pA, _pB2, 0);
                _sum9 = vfmaq_laneq_f32(_sum9, _pA, _pB2, 1);
                _suma = vfmaq_laneq_f32(_suma, _pA, _pB2, 2);
                _sumb = vfmaq_laneq_f32(_sumb, _pA, _pB2, 3);

                pA += 4;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
                    vst1q_f32(outptr0 + 4 * 2, _sum2);
                    vst1q_f32(outptr0 + 4 * 3, _sum3);
                    vst1q_f32(outptr0 + 4 * 4, _sum4);
                    vst1q_f32(outptr0 + 4 * 5, _sum5);
                    vst1q_f32(outptr0 + 4 * 6, _sum6);
                    vst1q_f32(outptr0 + 4 * 7, _sum7);
                    vst1q_f32(outptr0 + 4 * 8, _sum8);
                    vst1q_f32(outptr0 + 4 * 9, _sum9);
                    vst1q_f32(outptr0 + 4 * 10, _suma);
                    vst1q_f32(outptr0 + 4 * 11, _sumb);
                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose4x12_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, _sum8, _sum9, _suma, _sumb);

                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
                    vst1q_f32(outptr0 + 8, _sum2);
                    vst1q_f32(outptr0 + out_hstep, _sum3);
                    vst1q_f32(outptr0 + out_hstep + 4, _sum4);
                    vst1q_f32(outptr0 + out_hstep + 8, _sum5);
                    vst1q_f32(outptr0 + out_hstep * 2, _sum6);
                    vst1q_f32(outptr0 + out_hstep * 2 + 4, _sum7);
                    vst1q_f32(outptr0 + out_hstep * 2 + 8, _sum8);
                    vst1q_f32(outptr0 + out_hstep * 3, _sum9);
                    vst1q_f32(outptr0 + out_hstep * 3 + 4, _suma);
                    vst1q_f32(outptr0 + out_hstep * 3 + 8, _sumb);
                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 4 * 2, _sum2);
                vst1q_f32(outptr + 4 * 3, _sum3);
                vst1q_f32(outptr + 4 * 4, _sum4);
                vst1q_f32(outptr + 4 * 5, _sum5);
                vst1q_f32(outptr + 4 * 6, _sum6);
                vst1q_f32(outptr + 4 * 7, _sum7);
                vst1q_f32(outptr + 4 * 8, _sum8);
                vst1q_f32(outptr + 4 * 9, _sum9);
                vst1q_f32(outptr + 4 * 10, _suma);
                vst1q_f32(outptr + 4 * 11, _sumb);
            }

            outptr += 48;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;
            float32x4_t _sum4;
            float32x4_t _sum5;
            float32x4_t _sum6;
            float32x4_t _sum7;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
                _sum2 = vdupq_n_f32(0.f);
                _sum3 = vdupq_n_f32(0.f);
                _sum4 = vdupq_n_f32(0.f);
                _sum5 = vdupq_n_f32(0.f);
                _sum6 = vdupq_n_f32(0.f);
                _sum7 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[0]);
                        _sum2 = vdupq_n_f32(pC[0]);
                        _sum3 = vdupq_n_f32(pC[0]);
                        _sum4 = vdupq_n_f32(pC[0]);
                        _sum5 = vdupq_n_f32(pC[0]);
                        _sum6 = vdupq_n_f32(pC[0]);
                        _sum7 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                        _sum4 = _sum0;
                        _sum5 = _sum0;
                        _sum6 = _sum0;
                        _sum7 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        _sum2 = vld1q_f32(pC + 8);
                        _sum3 = vld1q_f32(pC + 12);
                        _sum4 = vld1q_f32(pC + 16);
                        _sum5 = vld1q_f32(pC + 20);
                        _sum6 = vld1q_f32(pC + 24);
                        _sum7 = vld1q_f32(pC + 28);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[1]);
                        _sum2 = vdupq_n_f32(pC[2]);
                        _sum3 = vdupq_n_f32(pC[3]);
                        _sum4 = vdupq_n_f32(pC[4]);
                        _sum5 = vdupq_n_f32(pC[5]);
                        _sum6 = vdupq_n_f32(pC[6]);
                        _sum7 = vdupq_n_f32(pC[7]);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4 * 1);
                _sum2 = vld1q_f32(outptr + 4 * 2);
                _sum3 = vld1q_f32(outptr + 4 * 3);
                _sum4 = vld1q_f32(outptr + 4 * 4);
                _sum5 = vld1q_f32(outptr + 4 * 5);
                _sum6 = vld1q_f32(outptr + 4 * 6);
                _sum7 = vld1q_f32(outptr + 4 * 7);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = vld1q_f32(pA);
                float32x4_t _pB0 = vld1q_f32(pB);
                float32x4_t _pB1 = vld1q_f32(pB + 4);

#if __aarch64__
                _sum0 = vfmaq_laneq_f32(_sum0, _pA, _pB0, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _pA, _pB0, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _pA, _pB0, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _pA, _pB0, 3);
                _sum4 = vfmaq_laneq_f32(_sum4, _pA, _pB1, 0);
                _sum5 = vfmaq_laneq_f32(_sum5, _pA, _pB1, 1);
                _sum6 = vfmaq_laneq_f32(_sum6, _pA, _pB1, 2);
                _sum7 = vfmaq_laneq_f32(_sum7, _pA, _pB1, 3);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pA, vget_low_f32(_pB0), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pA, vget_low_f32(_pB0), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _pA, vget_high_f32(_pB0), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _pA, vget_high_f32(_pB0), 1);
                _sum4 = vmlaq_lane_f32(_sum4, _pA, vget_low_f32(_pB1), 0);
                _sum5 = vmlaq_lane_f32(_sum5, _pA, vget_low_f32(_pB1), 1);
                _sum6 = vmlaq_lane_f32(_sum6, _pA, vget_high_f32(_pB1), 0);
                _sum7 = vmlaq_lane_f32(_sum7, _pA, vget_high_f32(_pB1), 1);
#endif

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
                    vst1q_f32(outptr0 + 4 * 2, _sum2);
                    vst1q_f32(outptr0 + 4 * 3, _sum3);
                    vst1q_f32(outptr0 + 4 * 4, _sum4);
                    vst1q_f32(outptr0 + 4 * 5, _sum5);
                    vst1q_f32(outptr0 + 4 * 6, _sum6);
                    vst1q_f32(outptr0 + 4 * 7, _sum7);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose4x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
                    vst1q_f32(outptr0 + out_hstep, _sum2);
                    vst1q_f32(outptr0 + out_hstep + 4, _sum3);
                    vst1q_f32(outptr0 + out_hstep * 2, _sum4);
                    vst1q_f32(outptr0 + out_hstep * 2 + 4, _sum5);
                    vst1q_f32(outptr0 + out_hstep * 3, _sum6);
                    vst1q_f32(outptr0 + out_hstep * 3 + 4, _sum7);
                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 4 * 2, _sum2);
                vst1q_f32(outptr + 4 * 3, _sum3);
                vst1q_f32(outptr + 4 * 4, _sum4);
                vst1q_f32(outptr + 4 * 5, _sum5);
                vst1q_f32(outptr + 4 * 6, _sum6);
                vst1q_f32(outptr + 4 * 7, _sum7);
            }

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;
            float32x4_t _sum3;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
                _sum2 = vdupq_n_f32(0.f);
                _sum3 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[0]);
                        _sum2 = vdupq_n_f32(pC[0]);
                        _sum3 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        _sum2 = vld1q_f32(pC + 8);
                        _sum3 = vld1q_f32(pC + 12);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[1]);
                        _sum2 = vdupq_n_f32(pC[2]);
                        _sum3 = vdupq_n_f32(pC[3]);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4 * 1);
                _sum2 = vld1q_f32(outptr + 4 * 2);
                _sum3 = vld1q_f32(outptr + 4 * 3);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = vld1q_f32(pA);
                float32x4_t _pB = vld1q_f32(pB);

#if __aarch64__
                _sum0 = vfmaq_laneq_f32(_sum0, _pA, _pB, 0);
                _sum1 = vfmaq_laneq_f32(_sum1, _pA, _pB, 1);
                _sum2 = vfmaq_laneq_f32(_sum2, _pA, _pB, 2);
                _sum3 = vfmaq_laneq_f32(_sum3, _pA, _pB, 3);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pA, vget_low_f32(_pB), 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pA, vget_low_f32(_pB), 1);
                _sum2 = vmlaq_lane_f32(_sum2, _pA, vget_high_f32(_pB), 0);
                _sum3 = vmlaq_lane_f32(_sum3, _pA, vget_high_f32(_pB), 1);
#endif

                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
                    vst1q_f32(outptr0 + 4 * 2, _sum2);
                    vst1q_f32(outptr0 + 4 * 3, _sum3);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);

                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + out_hstep * 1, _sum1);
                    vst1q_f32(outptr0 + out_hstep * 2, _sum2);
                    vst1q_f32(outptr0 + out_hstep * 3, _sum3);
                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 4 * 2, _sum2);
                vst1q_f32(outptr + 4 * 3, _sum3);
            }

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[1]);
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = vld1q_f32(pA);
                float32x2_t _pB = vld1_f32(pB);

#if __aarch64__
                _sum0 = vfmaq_lane_f32(_sum0, _pA, _pB, 0);
                _sum1 = vfmaq_lane_f32(_sum1, _pA, _pB, 1);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pA, _pB, 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pA, _pB, 1);
#endif

                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    float sum1[4];
                    vst1q_f32(sum0, _sum0);
                    vst1q_f32(sum1, _sum1);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0[1] = sum1[0];
                    outptr0[out_hstep + 1] = sum1[1];
                    outptr0[out_hstep * 2 + 1] = sum1[2];
                    outptr0[out_hstep * 3 + 1] = sum1[3];
                    outptr0 += 2;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            float32x4_t _sum0;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vld1q_f32(pC);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vld1q_f32(pC);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pA = vld1q_f32(pA);
                float32x4_t _pB = vdupq_n_f32(pB[0]);

#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA, _pB);
#else
                _sum0 = vmlaq_f32(_sum0, _pA, _pB);
#endif

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vst1q_f32(outptr0, _sum0);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    vst1q_f32(sum0, _sum0);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0++;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
#endif // __ARM_NEON
    for (; ii + 1 < max_ii; ii += 2)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum02;
            float32x4_t _sum10;
            float32x4_t _sum11;
            float32x4_t _sum12;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum02 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);
                _sum12 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                        _sum02 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[0]);
                        _sum11 = vdupq_n_f32(pC[0]);
                        _sum12 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                        _sum02 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum11 = vdupq_n_f32(pC[1]);
                        _sum12 = vdupq_n_f32(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4x2_t _tmp01 = vld2q_f32(pC);
                        float32x4x2_t _tmp23 = vld2q_f32(pC + 8);
                        float32x4x2_t _tmp45 = vld2q_f32(pC + 16);
                        _sum00 = _tmp01.val[0];
                        _sum01 = _tmp23.val[0];
                        _sum02 = _tmp45.val[0];
                        _sum10 = _tmp01.val[1];
                        _sum11 = _tmp23.val[1];
                        _sum12 = _tmp45.val[1];
                        pC += 24;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        _sum02 = vld1q_f32(pC + 8);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        _sum12 = _sum02;
                        pC += 12;
                    }
                }
            }
            else
            {
                float32x4x2_t _tmp01 = vld2q_f32(outptr);
                float32x4x2_t _tmp23 = vld2q_f32(outptr + 8);
                float32x4x2_t _tmp45 = vld2q_f32(outptr + 16);
                _sum00 = _tmp01.val[0];
                _sum01 = _tmp23.val[0];
                _sum02 = _tmp45.val[0];
                _sum10 = _tmp01.val[1];
                _sum11 = _tmp23.val[1];
                _sum12 = _tmp45.val[1];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB0 = vld1q_f32(pB);
                float32x4_t _pB1 = vld1q_f32(pB + 4);
                float32x4_t _pB2 = vld1q_f32(pB + 8);

                float32x2_t _pA = vld1_f32(pA);

                _sum00 = vfmaq_lane_f32(_sum00, _pB0, _pA, 0);
                _sum01 = vfmaq_lane_f32(_sum01, _pB1, _pA, 0);
                _sum02 = vfmaq_lane_f32(_sum02, _pB2, _pA, 0);
                _sum10 = vfmaq_lane_f32(_sum10, _pB0, _pA, 1);
                _sum11 = vfmaq_lane_f32(_sum11, _pB1, _pA, 1);
                _sum12 = vfmaq_lane_f32(_sum12, _pB2, _pA, 1);

                pA += 2;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum01);
                    vst1q_f32(outptr0 + 8, _sum02);
                    vst1q_f32(outptr0 + out_hstep, _sum10);
                    vst1q_f32(outptr0 + out_hstep + 4, _sum11);
                    vst1q_f32(outptr0 + out_hstep + 8, _sum12);
                    outptr0 += 12;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                _tmp01.val[0] = _sum00;
                _tmp01.val[1] = _sum10;
                float32x4x2_t _tmp23;
                _tmp23.val[0] = _sum01;
                _tmp23.val[1] = _sum11;
                float32x4x2_t _tmp45;
                _tmp45.val[0] = _sum02;
                _tmp45.val[1] = _sum12;
                vst2q_f32(outptr, _tmp01);
                vst2q_f32(outptr + 8, _tmp23);
                vst2q_f32(outptr + 16, _tmp45);
            }

            outptr += 24;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum00;
            float32x4_t _sum01;
            float32x4_t _sum10;
            float32x4_t _sum11;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32(0.f);
                _sum01 = vdupq_n_f32(0.f);
                _sum10 = vdupq_n_f32(0.f);
                _sum11 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[0]);
                        _sum11 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vdupq_n_f32(pC[0]);
                        _sum01 = vdupq_n_f32(pC[0]);
                        _sum10 = vdupq_n_f32(pC[1]);
                        _sum11 = vdupq_n_f32(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4x2_t _tmp01 = vld2q_f32(pC);
                        float32x4x2_t _tmp23 = vld2q_f32(pC + 8);
                        _sum00 = _tmp01.val[0];
                        _sum01 = _tmp23.val[0];
                        _sum10 = _tmp01.val[1];
                        _sum11 = _tmp23.val[1];
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vld1q_f32(pC);
                        _sum01 = vld1q_f32(pC + 4);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        pC += 8;
                    }
                }
            }
            else
            {
                float32x4x2_t _tmp01 = vld2q_f32(outptr);
                float32x4x2_t _tmp23 = vld2q_f32(outptr + 8);
                _sum00 = _tmp01.val[0];
                _sum01 = _tmp23.val[0];
                _sum10 = _tmp01.val[1];
                _sum11 = _tmp23.val[1];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB0 = vld1q_f32(pB);
                float32x4_t _pB1 = vld1q_f32(pB + 4);

                float32x2_t _pA = vld1_f32(pA);
#if __aarch64__
                _sum00 = vfmaq_lane_f32(_sum00, _pB0, _pA, 0);
                _sum01 = vfmaq_lane_f32(_sum01, _pB1, _pA, 0);
                _sum10 = vfmaq_lane_f32(_sum10, _pB0, _pA, 1);
                _sum11 = vfmaq_lane_f32(_sum11, _pB1, _pA, 1);
#else
                _sum00 = vmlaq_lane_f32(_sum00, _pB0, _pA, 0);
                _sum01 = vmlaq_lane_f32(_sum01, _pB1, _pA, 0);
                _sum10 = vmlaq_lane_f32(_sum10, _pB0, _pA, 1);
                _sum11 = vmlaq_lane_f32(_sum11, _pB1, _pA, 1);
#endif

                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_f32(outptr0, _sum00);
                    vst1q_f32(outptr0 + 4, _sum01);
                    vst1q_f32(outptr0 + out_hstep, _sum10);
                    vst1q_f32(outptr0 + out_hstep + 4, _sum11);
                    outptr0 += 8;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                _tmp01.val[0] = _sum00;
                _tmp01.val[1] = _sum10;
                float32x4x2_t _tmp23;
                _tmp23.val[0] = _sum01;
                _tmp23.val[1] = _sum11;
                vst2q_f32(outptr, _tmp01);
                vst2q_f32(outptr + 8, _tmp23);
            }

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4x2_t _tmp01 = vld2q_f32(pC);
                        _sum0 = _tmp01.val[0];
                        _sum1 = _tmp01.val[1];
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = _sum0;
                        pC += 4;
                    }
                }
            }
            else
            {
                float32x4x2_t _tmp01 = vld2q_f32(outptr);
                _sum0 = _tmp01.val[0];
                _sum1 = _tmp01.val[1];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB = vld1q_f32(pB);

                float32x2_t _pA = vld1_f32(pA);
#if __aarch64__
                _sum0 = vfmaq_lane_f32(_sum0, _pB, _pA, 0);
                _sum1 = vfmaq_lane_f32(_sum1, _pB, _pA, 1);
#else
                _sum0 = vmlaq_lane_f32(_sum0, _pB, _pA, 0);
                _sum1 = vmlaq_lane_f32(_sum1, _pB, _pA, 1);
#endif

                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + out_hstep, _sum1);
                    outptr0 += 4;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                _tmp01.val[0] = _sum0;
                _tmp01.val[1] = _sum1;
                vst2q_f32(outptr, _tmp01);
            }

            outptr += 8;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum00;
            float sum01;
            float sum10;
            float sum11;

            if (k == 0)
            {
                sum00 = 0.f;
                sum01 = 0.f;
                sum10 = 0.f;
                sum11 = 0.f;

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        sum00 = pC[0];
                        sum01 = pC[0];
                        sum10 = pC[0];
                        sum11 = pC[0];
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum00 = pC[0];
                        sum01 = pC[1];
                        sum10 = pC[0];
                        sum11 = pC[1];
                    }
                    if (broadcast_type_C == 3)
                    {
                        sum00 = pC[0];
                        sum01 = pC[1];
                        sum10 = pC[2];
                        sum11 = pC[3];
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum00 = pC[0];
                        sum01 = pC[0];
                        sum10 = pC[1];
                        sum11 = pC[1];
                        pC += 2;
                    }
                }
            }
            else
            {
                sum00 = outptr[0];
                sum01 = outptr[1];
                sum10 = outptr[2];
                sum11 = outptr[3];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum00 += pA[0] * pB[0];
                sum01 += pA[1] * pB[0];
                sum10 += pA[0] * pB[1];
                sum11 += pA[1] * pB[1];

                pA += 2;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum00;
                    outptr0[1] = sum10;
                    outptr0[out_hstep] = sum01;
                    outptr0[out_hstep + 1] = sum11;
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum00;
                outptr[1] = sum01;
                outptr[2] = sum10;
                outptr[3] = sum11;
            }

            outptr += 4;
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        sum0 = pC[0];
                        sum1 = pC[0];
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum0 = pC[0];
                        sum1 = pC[1];
                    }
                    if (broadcast_type_C == 3)
                    {
                        sum0 = pC[0];
                        sum1 = pC[1];
                        pC += 2;
                    }
                    if (broadcast_type_C == 4)
                    {
                        sum0 = pC[0];
                        sum1 = pC[0];
                        pC += 1;
                    }
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[1] * pB[0];
                pA += 2;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[out_hstep] = sum1;
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }

        pAT += max_kk * 2;
    }
    for (; ii < max_ii; ii += 1)
    {
        float* outptr0 = (float*)top_blob + (i + ii) * out_hstep + j;

        const float* pB = pBT;

        if (pC)
        {
            if (broadcast_type_C == 1 || broadcast_type_C == 2)
            {
                pC = (const float*)CT_tile + i + ii;
            }
            if (broadcast_type_C == 4)
            {
                pC = (const float*)CT_tile + j;
            }
        }

        int jj = 0;
#if __ARM_NEON
#if __aarch64__
        for (; jj + 11 < max_jj; jj += 12)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;
            float32x4_t _sum2;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);
                _sum2 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[0]);
                        _sum2 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        _sum2 = vld1q_f32(pC + 8);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
                _sum2 = vld1q_f32(outptr + 8);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB0 = vld1q_f32(pB);
                float32x4_t _pB1 = vld1q_f32(pB + 4);
                float32x4_t _pB2 = vld1q_f32(pB + 8);

                float32x4_t _pA0 = vdupq_n_f32(pA[0]);

                _sum0 = vfmaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA0, _pB1);
                _sum2 = vfmaq_f32(_sum2, _pA0, _pB2);

                pA += 1;
                pB += 12;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
                    vst1q_f32(outptr0 + 8, _sum2);
                    outptr0 += 12;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
                vst1q_f32(outptr + 8, _sum2);
            }

            outptr += 12;
        }
#endif // __aarch64__
        for (; jj + 7 < max_jj; jj += 8)
        {
            float32x4_t _sum0;
            float32x4_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32(0.f);
                _sum1 = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vdupq_n_f32(pC[0]);
                        _sum1 = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = vld1q_f32(pC);
                        _sum1 = vld1q_f32(pC + 4);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = vld1q_f32(outptr);
                _sum1 = vld1q_f32(outptr + 4);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB0 = vld1q_f32(pB);
                float32x4_t _pB1 = vld1q_f32(pB + 4);

                float32x4_t _pA0 = vdupq_n_f32(pA[0]);
#if __aarch64__
                _sum0 = vfmaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA0, _pB1);
#else
                _sum0 = vmlaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vmlaq_f32(_sum1, _pA0, _pB1);
#endif

                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_f32(outptr0, _sum0);
                    vst1q_f32(outptr0 + 4, _sum1);
                    outptr0 += 8;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum0);
                vst1q_f32(outptr + 4, _sum1);
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            float32x4_t _sum;

            if (k == 0)
            {
                _sum = vdupq_n_f32(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum = vdupq_n_f32(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum = vld1q_f32(pC);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum = vld1q_f32(outptr);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                float32x4_t _pB = vld1q_f32(pB);
                float32x4_t _pA = vdupq_n_f32(pA[0]);

#if __aarch64__
                _sum = vfmaq_f32(_sum, _pA, _pB);
#else
                _sum = vmlaq_f32(_sum, _pA, _pB);
#endif

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vst1q_f32(outptr0, _sum);
                    outptr0 += 4;
                }
            }
            else
            {
                vst1q_f32(outptr, _sum);
            }

            outptr += 4;
        }
#endif // __ARM_NEON
        for (; jj + 1 < max_jj; jj += 2)
        {
            float sum0;
            float sum1;

            if (k == 0)
            {
                sum0 = 0.f;
                sum1 = 0.f;

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum0 = pC[0];
                        sum1 = pC[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        sum0 = pC[0];
                        sum1 = pC[1];
                        pC += 2;
                    }
                }
            }
            else
            {
                sum0 = outptr[0];
                sum1 = outptr[1];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum0 += pA[0] * pB[0];
                sum1 += pA[0] * pB[1];

                pA += 1;
                pB += 2;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum0;
                    outptr0[1] = sum1;
                    outptr0 += 2;
                }
            }
            else
            {
                outptr[0] = sum0;
                outptr[1] = sum1;
            }

            outptr += 2;
        }
        for (; jj < max_jj; jj += 1)
        {
            float sum;

            if (k == 0)
            {
                sum = 0.f;

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        sum = pC[0];
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        sum = pC[0];
                        pC += 1;
                    }
                }
            }
            else
            {
                sum = outptr[0];
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                sum += pA[0] * pB[0];
                pA += 1;
                pB += 1;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    outptr0[0] = sum;
                    outptr0++;
                }
            }
            else
            {
                outptr[0] = sum;
            }

            outptr += 1;
        }

        pAT += max_kk;
    }
}