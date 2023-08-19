#include <riscv_vector.h>
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
            vfloat32m1_t _sum00;
            vfloat32m1_t _sum01;
            vfloat32m1_t _sum10;
            vfloat32m1_t _sum11;
            vfloat32m1_t _sum20;
            vfloat32m1_t _sum21;
            vfloat32m1_t _sum30;
            vfloat32m1_t _sum31;
            vfloat32m1_t _sum40;
            vfloat32m1_t _sum41;
            vfloat32m1_t _sum50;
            vfloat32m1_t _sum51;
            vfloat32m1_t _sum60;
            vfloat32m1_t _sum61;
            vfloat32m1_t _sum70;
            vfloat32m1_t _sum71;
            vfloat32m1_t _sum80;
            vfloat32m1_t _sum81;
            vfloat32m1_t _sum90;
            vfloat32m1_t _sum91;
            vfloat32m1_t _suma0;
            vfloat32m1_t _suma1;
            vfloat32m1_t _sumb0;
            vfloat32m1_t _sumb1;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32_riscv(0.f);
                _sum01 = vdupq_n_f32_riscv(0.f);
                _sum10 = vdupq_n_f32_riscv(0.f);
                _sum11 = vdupq_n_f32_riscv(0.f);
                _sum20 = vdupq_n_f32_riscv(0.f);
                _sum21 = vdupq_n_f32_riscv(0.f);
                _sum30 = vdupq_n_f32_riscv(0.f);
                _sum31 = vdupq_n_f32_riscv(0.f);
                _sum40 = vdupq_n_f32_riscv(0.f);
                _sum41 = vdupq_n_f32_riscv(0.f);
                _sum50 = vdupq_n_f32_riscv(0.f);
                _sum51 = vdupq_n_f32_riscv(0.f);
                _sum60 = vdupq_n_f32_riscv(0.f);
                _sum61 = vdupq_n_f32_riscv(0.f);
                _sum70 = vdupq_n_f32_riscv(0.f);
                _sum71 = vdupq_n_f32_riscv(0.f);
                _sum80 = vdupq_n_f32_riscv(0.f);
                _sum81 = vdupq_n_f32_riscv(0.f);
                _sum90 = vdupq_n_f32_riscv(0.f);
                _sum91 = vdupq_n_f32_riscv(0.f);
                _suma0 = vdupq_n_f32_riscv(0.f);
                _suma1 = vdupq_n_f32_riscv(0.f);
                _sumb0 = vdupq_n_f32_riscv(0.f);
                _sumb1 = vdupq_n_f32_riscv(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32_riscv(pC[0]);
                        _sum01 = vdupq_n_f32_riscv(pC[0]);
                        _sum10 = vdupq_n_f32_riscv(pC[0]);
                        _sum11 = vdupq_n_f32_riscv(pC[0]);
                        _sum20 = vdupq_n_f32_riscv(pC[0]);
                        _sum21 = vdupq_n_f32_riscv(pC[0]);
                        _sum30 = vdupq_n_f32_riscv(pC[0]);
                        _sum31 = vdupq_n_f32_riscv(pC[0]);
                        _sum40 = vdupq_n_f32_riscv(pC[0]);
                        _sum41 = vdupq_n_f32_riscv(pC[0]);
                        _sum50 = vdupq_n_f32_riscv(pC[0]);
                        _sum51 = vdupq_n_f32_riscv(pC[0]);
                        _sum60 = vdupq_n_f32_riscv(pC[0]);
                        _sum61 = vdupq_n_f32_riscv(pC[0]);
                        _sum70 = vdupq_n_f32_riscv(pC[0]);
                        _sum71 = vdupq_n_f32_riscv(pC[0]);
                        _sum80 = vdupq_n_f32_riscv(pC[0]);
                        _sum81 = vdupq_n_f32_riscv(pC[0]);
                        _sum90 = vdupq_n_f32_riscv(pC[0]);
                        _sum91 = vdupq_n_f32_riscv(pC[0]);
                        _suma0 = vdupq_n_f32_riscv(pC[0]);
                        _suma1 = vdupq_n_f32_riscv(pC[0]);
                        _sumb0 = vdupq_n_f32_riscv(pC[0]);
                        _sumb1 = vdupq_n_f32_riscv(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vle32_v_f32m1(pC, VL);
                        _sum01 = vle32_v_f32m1(pC + 4, VL);
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
                        _sum00 = vle32_v_f32m1(pC, VL);
                        _sum01 = vle32_v_f32m1(pC + 4 * 1, VL);
                        _sum10 = vle32_v_f32m1(pC + 4 * 2, VL);
                        _sum11 = vle32_v_f32m1(pC + 4 * 3, VL);
                        _sum20 = vle32_v_f32m1(pC + 4 * 4, VL);
                        _sum21 = vle32_v_f32m1(pC + 4 * 5, VL);
                        _sum30 = vle32_v_f32m1(pC + 4 * 6, VL);
                        _sum31 = vle32_v_f32m1(pC + 4 * 7, VL);
                        _sum40 = vle32_v_f32m1(pC + 4 * 8, VL);
                        _sum41 = vle32_v_f32m1(pC + 4 * 9, VL);
                        _sum50 = vle32_v_f32m1(pC + 4 * 10, VL);
                        _sum51 = vle32_v_f32m1(pC + 4 * 11, VL);
                        _sum60 = vle32_v_f32m1(pC + 4 * 12, VL);
                        _sum61 = vle32_v_f32m1(pC + 4 * 13, VL);
                        _sum70 = vle32_v_f32m1(pC + 4 * 14, VL);
                        _sum71 = vle32_v_f32m1(pC + 4 * 15, VL);
                        _sum80 = vle32_v_f32m1(pC + 4 * 16, VL);
                        _sum81 = vle32_v_f32m1(pC + 4 * 17, VL);
                        _sum90 = vle32_v_f32m1(pC + 4 * 18, VL);
                        _sum91 = vle32_v_f32m1(pC + 4 * 19, VL);
                        _suma0 = vle32_v_f32m1(pC + 4 * 20, VL);
                        _suma1 = vle32_v_f32m1(pC + 4 * 21, VL);
                        _sumb0 = vle32_v_f32m1(pC + 4 * 22, VL);
                        _sumb1 = vle32_v_f32m1(pC + 4 * 23, VL);
                        pC += 96;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32_riscv(pC[0]);
                        _sum10 = vdupq_n_f32_riscv(pC[1]);
                        _sum20 = vdupq_n_f32_riscv(pC[2]);
                        _sum30 = vdupq_n_f32_riscv(pC[3]);
                        _sum40 = vdupq_n_f32_riscv(pC[4]);
                        _sum50 = vdupq_n_f32_riscv(pC[5]);
                        _sum60 = vdupq_n_f32_riscv(pC[6]);
                        _sum70 = vdupq_n_f32_riscv(pC[7]);
                        _sum80 = vdupq_n_f32_riscv(pC[8]);
                        _sum90 = vdupq_n_f32_riscv(pC[9]);
                        _suma0 = vdupq_n_f32_riscv(pC[10]);
                        _sumb0 = vdupq_n_f32_riscv(pC[11]);
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
                _sum00 = vle32_v_f32m1(outptr, VL);
                _sum01 = vle32_v_f32m1(outptr + 4 * 1, VL);
                _sum10 = vle32_v_f32m1(outptr + 4 * 2, VL);
                _sum11 = vle32_v_f32m1(outptr + 4 * 3, VL);
                _sum20 = vle32_v_f32m1(outptr + 4 * 4, VL);
                _sum21 = vle32_v_f32m1(outptr + 4 * 5, VL);
                _sum30 = vle32_v_f32m1(outptr + 4 * 6, VL);
                _sum31 = vle32_v_f32m1(outptr + 4 * 7, VL);
                _sum40 = vle32_v_f32m1(outptr + 4 * 8, VL);
                _sum41 = vle32_v_f32m1(outptr + 4 * 9, VL);
                _sum50 = vle32_v_f32m1(outptr + 4 * 10, VL);
                _sum51 = vle32_v_f32m1(outptr + 4 * 11, VL);
                _sum60 = vle32_v_f32m1(outptr + 4 * 12, VL);
                _sum61 = vle32_v_f32m1(outptr + 4 * 13, VL);
                _sum70 = vle32_v_f32m1(outptr + 4 * 14, VL);
                _sum71 = vle32_v_f32m1(outptr + 4 * 15, VL);
                _sum80 = vle32_v_f32m1(outptr + 4 * 16, VL);
                _sum81 = vle32_v_f32m1(outptr + 4 * 17, VL);
                _sum90 = vle32_v_f32m1(outptr + 4 * 18, VL);
                _sum91 = vle32_v_f32m1(outptr + 4 * 19, VL);
                _suma0 = vle32_v_f32m1(outptr + 4 * 20, VL);
                _suma1 = vle32_v_f32m1(outptr + 4 * 21, VL);
                _sumb0 = vle32_v_f32m1(outptr + 4 * 22, VL);
                _sumb1 = vle32_v_f32m1(outptr + 4 * 23, VL);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk + 3 < max_kk; kk += 4)
            {
                vfloat32m1_t _pA0 = vle32_v_f32m1(pA, VL);
                vfloat32m1_t _pA1 = vle32_v_f32m1(pA + 4, VL);

                vfloat32m1_t _pB0 = vle32_v_f32m1(pB, VL);
                vfloat32m1_t _pB1 = vle32_v_f32m1(pB + 4, VL);
                vfloat32m1_t _pB2 = vle32_v_f32m1(pB + 8, VL);

                _sum00 = vfmaq_laneq_f32_riscv(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32_riscv(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32_riscv(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32_riscv(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32_riscv(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32_riscv(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32_riscv(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32_riscv(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32_riscv(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32_riscv(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32_riscv(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32_riscv(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32_riscv(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32_riscv(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32_riscv(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32_riscv(_sum71, _pA1, _pB1, 3);
                _sum80 = vfmaq_laneq_f32_riscv(_sum80, _pA0, _pB2, 0);
                _sum81 = vfmaq_laneq_f32_riscv(_sum81, _pA1, _pB2, 0);
                _sum90 = vfmaq_laneq_f32_riscv(_sum90, _pA0, _pB2, 1);
                _sum91 = vfmaq_laneq_f32_riscv(_sum91, _pA1, _pB2, 1);
                _suma0 = vfmaq_laneq_f32_riscv(_suma0, _pA0, _pB2, 2);
                _suma1 = vfmaq_laneq_f32_riscv(_suma1, _pA1, _pB2, 2);
                _sumb0 = vfmaq_laneq_f32_riscv(_sumb0, _pA0, _pB2, 3);
                _sumb1 = vfmaq_laneq_f32_riscv(_sumb1, _pA1, _pB2, 3);

                pA += 8;
                pB += 12;

                _pA0 = vle32_v_f32m1(pA, VL);
                _pA1 = vle32_v_f32m1(pA + 4, VL);

                _pB0 = vle32_v_f32m1(pB, VL);
                _pB1 = vle32_v_f32m1(pB + 4, VL);
                _pB2 = vle32_v_f32m1(pB + 8, VL);

                _sum00 = vfmaq_laneq_f32_riscv(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32_riscv(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32_riscv(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32_riscv(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32_riscv(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32_riscv(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32_riscv(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32_riscv(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32_riscv(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32_riscv(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32_riscv(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32_riscv(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32_riscv(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32_riscv(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32_riscv(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32_riscv(_sum71, _pA1, _pB1, 3);
                _sum80 = vfmaq_laneq_f32_riscv(_sum80, _pA0, _pB2, 0);
                _sum81 = vfmaq_laneq_f32_riscv(_sum81, _pA1, _pB2, 0);
                _sum90 = vfmaq_laneq_f32_riscv(_sum90, _pA0, _pB2, 1);
                _sum91 = vfmaq_laneq_f32_riscv(_sum91, _pA1, _pB2, 1);
                _suma0 = vfmaq_laneq_f32_riscv(_suma0, _pA0, _pB2, 2);
                _suma1 = vfmaq_laneq_f32_riscv(_suma1, _pA1, _pB2, 2);
                _sumb0 = vfmaq_laneq_f32_riscv(_sumb0, _pA0, _pB2, 3);
                _sumb1 = vfmaq_laneq_f32_riscv(_sumb1, _pA1, _pB2, 3);

                pA += 8;
                pB += 12;

                _pA0 = vle32_v_f32m1(pA, VL);
                _pA1 = vle32_v_f32m1(pA + 4, VL);

                _pB0 = vle32_v_f32m1(pB, VL);
                _pB1 = vle32_v_f32m1(pB + 4, VL);
                _pB2 = vle32_v_f32m1(pB + 8, VL);

                _sum00 = vfmaq_laneq_f32_riscv(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32_riscv(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32_riscv(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32_riscv(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32_riscv(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32_riscv(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32_riscv(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32_riscv(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32_riscv(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32_riscv(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32_riscv(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32_riscv(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32_riscv(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32_riscv(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32_riscv(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32_riscv(_sum71, _pA1, _pB1, 3);
                _sum80 = vfmaq_laneq_f32_riscv(_sum80, _pA0, _pB2, 0);
                _sum81 = vfmaq_laneq_f32_riscv(_sum81, _pA1, _pB2, 0);
                _sum90 = vfmaq_laneq_f32_riscv(_sum90, _pA0, _pB2, 1);
                _sum91 = vfmaq_laneq_f32_riscv(_sum91, _pA1, _pB2, 1);
                _suma0 = vfmaq_laneq_f32_riscv(_suma0, _pA0, _pB2, 2);
                _suma1 = vfmaq_laneq_f32_riscv(_suma1, _pA1, _pB2, 2);
                _sumb0 = vfmaq_laneq_f32_riscv(_sumb0, _pA0, _pB2, 3);
                _sumb1 = vfmaq_laneq_f32_riscv(_sumb1, _pA1, _pB2, 3);

                pA += 8;
                pB += 12;

                _pA0 = vle32_v_f32m1(pA, VL);
                _pA1 = vle32_v_f32m1(pA + 4, VL);

                _pB0 = vle32_v_f32m1(pB, VL);
                _pB1 = vle32_v_f32m1(pB + 4, VL);
                _pB2 = vle32_v_f32m1(pB + 8, VL);

                _sum00 = vfmaq_laneq_f32_riscv(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32_riscv(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32_riscv(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32_riscv(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32_riscv(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32_riscv(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32_riscv(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32_riscv(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32_riscv(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32_riscv(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32_riscv(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32_riscv(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32_riscv(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32_riscv(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32_riscv(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32_riscv(_sum71, _pA1, _pB1, 3);
                _sum80 = vfmaq_laneq_f32_riscv(_sum80, _pA0, _pB2, 0);
                _sum81 = vfmaq_laneq_f32_riscv(_sum81, _pA1, _pB2, 0);
                _sum90 = vfmaq_laneq_f32_riscv(_sum90, _pA0, _pB2, 1);
                _sum91 = vfmaq_laneq_f32_riscv(_sum91, _pA1, _pB2, 1);
                _suma0 = vfmaq_laneq_f32_riscv(_suma0, _pA0, _pB2, 2);
                _suma1 = vfmaq_laneq_f32_riscv(_suma1, _pA1, _pB2, 2);
                _sumb0 = vfmaq_laneq_f32_riscv(_sumb0, _pA0, _pB2, 3);
                _sumb1 = vfmaq_laneq_f32_riscv(_sumb1, _pA1, _pB2, 3);

                pA += 8;
                pB += 12;
            }
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA0 = vle32_v_f32m1(pA, VL);
                vfloat32m1_t _pA1 = vle32_v_f32m1(pA + 4, VL);

                vfloat32m1_t _pB0 = vle32_v_f32m1(pB, VL);
                vfloat32m1_t _pB1 = vle32_v_f32m1(pB + 4, VL);
                vfloat32m1_t _pB2 = vle32_v_f32m1(pB + 8, VL);

                _sum00 = vfmaq_laneq_f32_riscv(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32_riscv(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32_riscv(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32_riscv(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32_riscv(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32_riscv(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32_riscv(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32_riscv(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32_riscv(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32_riscv(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32_riscv(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32_riscv(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32_riscv(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32_riscv(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32_riscv(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32_riscv(_sum71, _pA1, _pB1, 3);
                _sum80 = vfmaq_laneq_f32_riscv(_sum80, _pA0, _pB2, 0);
                _sum81 = vfmaq_laneq_f32_riscv(_sum81, _pA1, _pB2, 0);
                _sum90 = vfmaq_laneq_f32_riscv(_sum90, _pA0, _pB2, 1);
                _sum91 = vfmaq_laneq_f32_riscv(_sum91, _pA1, _pB2, 1);
                _suma0 = vfmaq_laneq_f32_riscv(_suma0, _pA0, _pB2, 2);
                _suma1 = vfmaq_laneq_f32_riscv(_suma1, _pA1, _pB2, 2);
                _sumb0 = vfmaq_laneq_f32_riscv(_sumb0, _pA0, _pB2, 3);
                _sumb1 = vfmaq_laneq_f32_riscv(_sumb1, _pA1, _pB2, 3);

                pA += 8;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum00, VL);
                    vse32_v_f32m1(outptr0 + 4, _sum10, VL);
                    vse32_v_f32m1(outptr0 + 4 * 2, _sum20, VL);
                    vse32_v_f32m1(outptr0 + 4 * 3, _sum30, VL);
                    vse32_v_f32m1(outptr0 + 4 * 4, _sum40, VL);
                    vse32_v_f32m1(outptr0 + 4 * 5, _sum50, VL);
                    vse32_v_f32m1(outptr0 + 4 * 6, _sum60, VL);
                    vse32_v_f32m1(outptr0 + 4 * 7, _sum70, VL);
                    vse32_v_f32m1(outptr0 + 4 * 8, _sum80, VL);
                    vse32_v_f32m1(outptr0 + 4 * 9, _sum90, VL);
                    vse32_v_f32m1(outptr0 + 4 * 10, _suma0, VL);
                    vse32_v_f32m1(outptr0 + 4 * 11, _sumb0, VL);

                    vse32_v_f32m1(outptr0 + out_hstep * 4, _sum01, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4, _sum11, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 2, _sum21, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 3, _sum31, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 4, _sum41, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 5, _sum51, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 6, _sum61, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 7, _sum71, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 8, _sum81, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 9, _sum91, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 10, _suma1, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 11, _sumb1, VL);

                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose8x12_ps(_sum00, _sum01, _sum10, _sum11, _sum20, _sum21, _sum30, _sum31, _sum40, _sum41, _sum50, _sum51, _sum60, _sum61, _sum70, _sum71, _sum80, _sum81, _sum90, _sum91, _suma0, _suma1, _sumb0, _sumb1);

                    vse32_v_f32m1(outptr0, _sum00, VL);
                    vse32_v_f32m1(outptr0 + 4, _sum01, VL);
                    vse32_v_f32m1(outptr0 + 8, _sum10, VL);
                    vse32_v_f32m1(outptr0 + out_hstep, _sum11, VL);
                    vse32_v_f32m1(outptr0 + out_hstep + 4, _sum20, VL);
                    vse32_v_f32m1(outptr0 + out_hstep + 8, _sum21, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 2, _sum30, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 2 + 4, _sum31, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 2 + 8, _sum40, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 3, _sum41, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 3 + 4, _sum50, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 3 + 8, _sum51, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4, _sum60, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4, _sum61, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 8, _sum70, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 5, _sum71, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 5 + 4, _sum80, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 5 + 8, _sum81, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 6, _sum90, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 6 + 4, _sum91, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 6 + 8, _suma0, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 7, _suma1, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 7 + 4, _sumb0, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 7 + 8, _sumb1, VL);

                    outptr0 += 12;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum00, VL);
                vse32_v_f32m1(outptr + 4, _sum01, VL);
                vse32_v_f32m1(outptr + 4 * 2, _sum10, VL);
                vse32_v_f32m1(outptr + 4 * 3, _sum11, VL);
                vse32_v_f32m1(outptr + 4 * 4, _sum20, VL);
                vse32_v_f32m1(outptr + 4 * 5, _sum21, VL);
                vse32_v_f32m1(outptr + 4 * 6, _sum30, VL);
                vse32_v_f32m1(outptr + 4 * 7, _sum31, VL);
                vse32_v_f32m1(outptr + 4 * 8, _sum40, VL);
                vse32_v_f32m1(outptr + 4 * 9, _sum41, VL);
                vse32_v_f32m1(outptr + 4 * 10, _sum50, VL);
                vse32_v_f32m1(outptr + 4 * 11, _sum51, VL);
                vse32_v_f32m1(outptr + 4 * 12, _sum60, VL);
                vse32_v_f32m1(outptr + 4 * 13, _sum61, VL);
                vse32_v_f32m1(outptr + 4 * 14, _sum70, VL);
                vse32_v_f32m1(outptr + 4 * 15, _sum71, VL);
                vse32_v_f32m1(outptr + 4 * 16, _sum80, VL);
                vse32_v_f32m1(outptr + 4 * 17, _sum81, VL);
                vse32_v_f32m1(outptr + 4 * 18, _sum90, VL);
                vse32_v_f32m1(outptr + 4 * 19, _sum91, VL);
                vse32_v_f32m1(outptr + 4 * 20, _suma0, VL);
                vse32_v_f32m1(outptr + 4 * 21, _suma1, VL);
                vse32_v_f32m1(outptr + 4 * 22, _sumb0, VL);
                vse32_v_f32m1(outptr + 4 * 23, _sumb1, VL);
            }

            outptr += 96;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat32m1_t _sum00;
            vfloat32m1_t _sum01;
            vfloat32m1_t _sum10;
            vfloat32m1_t _sum11;
            vfloat32m1_t _sum20;
            vfloat32m1_t _sum21;
            vfloat32m1_t _sum30;
            vfloat32m1_t _sum31;
            vfloat32m1_t _sum40;
            vfloat32m1_t _sum41;
            vfloat32m1_t _sum50;
            vfloat32m1_t _sum51;
            vfloat32m1_t _sum60;
            vfloat32m1_t _sum61;
            vfloat32m1_t _sum70;
            vfloat32m1_t _sum71;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32_riscv(0.f);
                _sum01 = vdupq_n_f32_riscv(0.f);
                _sum10 = vdupq_n_f32_riscv(0.f);
                _sum11 = vdupq_n_f32_riscv(0.f);
                _sum20 = vdupq_n_f32_riscv(0.f);
                _sum21 = vdupq_n_f32_riscv(0.f);
                _sum30 = vdupq_n_f32_riscv(0.f);
                _sum31 = vdupq_n_f32_riscv(0.f);
                _sum40 = vdupq_n_f32_riscv(0.f);
                _sum41 = vdupq_n_f32_riscv(0.f);
                _sum50 = vdupq_n_f32_riscv(0.f);
                _sum51 = vdupq_n_f32_riscv(0.f);
                _sum60 = vdupq_n_f32_riscv(0.f);
                _sum61 = vdupq_n_f32_riscv(0.f);
                _sum70 = vdupq_n_f32_riscv(0.f);
                _sum71 = vdupq_n_f32_riscv(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32_riscv(pC[0]);
                        _sum01 = vdupq_n_f32_riscv(pC[0]);
                        _sum10 = vdupq_n_f32_riscv(pC[0]);
                        _sum11 = vdupq_n_f32_riscv(pC[0]);
                        _sum20 = vdupq_n_f32_riscv(pC[0]);
                        _sum21 = vdupq_n_f32_riscv(pC[0]);
                        _sum30 = vdupq_n_f32_riscv(pC[0]);
                        _sum31 = vdupq_n_f32_riscv(pC[0]);
                        _sum40 = vdupq_n_f32_riscv(pC[0]);
                        _sum41 = vdupq_n_f32_riscv(pC[0]);
                        _sum50 = vdupq_n_f32_riscv(pC[0]);
                        _sum51 = vdupq_n_f32_riscv(pC[0]);
                        _sum60 = vdupq_n_f32_riscv(pC[0]);
                        _sum61 = vdupq_n_f32_riscv(pC[0]);
                        _sum70 = vdupq_n_f32_riscv(pC[0]);
                        _sum71 = vdupq_n_f32_riscv(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vle32_v_f32m1(pC, VL);
                        _sum01 = vle32_v_f32m1(pC + 4, VL);
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
                        _sum00 = vle32_v_f32m1(pC, VL);
                        _sum01 = vle32_v_f32m1(pC + 4 * 1, VL);
                        _sum10 = vle32_v_f32m1(pC + 4 * 2, VL);
                        _sum11 = vle32_v_f32m1(pC + 4 * 3, VL);
                        _sum20 = vle32_v_f32m1(pC + 4 * 4, VL);
                        _sum21 = vle32_v_f32m1(pC + 4 * 5, VL);
                        _sum30 = vle32_v_f32m1(pC + 4 * 6, VL);
                        _sum31 = vle32_v_f32m1(pC + 4 * 7, VL);
                        _sum40 = vle32_v_f32m1(pC + 4 * 8, VL);
                        _sum41 = vle32_v_f32m1(pC + 4 * 9, VL);
                        _sum50 = vle32_v_f32m1(pC + 4 * 10, VL);
                        _sum51 = vle32_v_f32m1(pC + 4 * 11, VL);
                        _sum60 = vle32_v_f32m1(pC + 4 * 12, VL);
                        _sum61 = vle32_v_f32m1(pC + 4 * 13, VL);
                        _sum70 = vle32_v_f32m1(pC + 4 * 14, VL);
                        _sum71 = vle32_v_f32m1(pC + 4 * 15, VL);
                        pC += 64;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32_riscv(pC[0]);
                        _sum10 = vdupq_n_f32_riscv(pC[1]);
                        _sum20 = vdupq_n_f32_riscv(pC[2]);
                        _sum30 = vdupq_n_f32_riscv(pC[3]);
                        _sum40 = vdupq_n_f32_riscv(pC[4]);
                        _sum50 = vdupq_n_f32_riscv(pC[5]);
                        _sum60 = vdupq_n_f32_riscv(pC[6]);
                        _sum70 = vdupq_n_f32_riscv(pC[7]);
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
                _sum00 = vle32_v_f32m1(outptr, VL);
                _sum01 = vle32_v_f32m1(outptr + 4 * 1, VL);
                _sum10 = vle32_v_f32m1(outptr + 4 * 2, VL);
                _sum11 = vle32_v_f32m1(outptr + 4 * 3, VL);
                _sum20 = vle32_v_f32m1(outptr + 4 * 4, VL);
                _sum21 = vle32_v_f32m1(outptr + 4 * 5, VL);
                _sum30 = vle32_v_f32m1(outptr + 4 * 6, VL);
                _sum31 = vle32_v_f32m1(outptr + 4 * 7, VL);
                _sum40 = vle32_v_f32m1(outptr + 4 * 8, VL);
                _sum41 = vle32_v_f32m1(outptr + 4 * 9, VL);
                _sum50 = vle32_v_f32m1(outptr + 4 * 10, VL);
                _sum51 = vle32_v_f32m1(outptr + 4 * 11, VL);
                _sum60 = vle32_v_f32m1(outptr + 4 * 12, VL);
                _sum61 = vle32_v_f32m1(outptr + 4 * 13, VL);
                _sum70 = vle32_v_f32m1(outptr + 4 * 14, VL);
                _sum71 = vle32_v_f32m1(outptr + 4 * 15, VL);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA0 = vle32_v_f32m1(pA, VL);
                vfloat32m1_t _pA1 = vle32_v_f32m1(pA + 4, VL);

                vfloat32m1_t _pB0 = vle32_v_f32m1(pB, VL);
                vfloat32m1_t _pB1 = vle32_v_f32m1(pB + 4, VL);

                _sum00 = vfmaq_laneq_f32_riscv(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32_riscv(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32_riscv(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32_riscv(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32_riscv(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32_riscv(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32_riscv(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32_riscv(_sum31, _pA1, _pB0, 3);
                _sum40 = vfmaq_laneq_f32_riscv(_sum40, _pA0, _pB1, 0);
                _sum41 = vfmaq_laneq_f32_riscv(_sum41, _pA1, _pB1, 0);
                _sum50 = vfmaq_laneq_f32_riscv(_sum50, _pA0, _pB1, 1);
                _sum51 = vfmaq_laneq_f32_riscv(_sum51, _pA1, _pB1, 1);
                _sum60 = vfmaq_laneq_f32_riscv(_sum60, _pA0, _pB1, 2);
                _sum61 = vfmaq_laneq_f32_riscv(_sum61, _pA1, _pB1, 2);
                _sum70 = vfmaq_laneq_f32_riscv(_sum70, _pA0, _pB1, 3);
                _sum71 = vfmaq_laneq_f32_riscv(_sum71, _pA1, _pB1, 3);

                pA += 8;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum00, VL);
                    vse32_v_f32m1(outptr0 + 4, _sum10, VL);
                    vse32_v_f32m1(outptr0 + 4 * 2, _sum20, VL);
                    vse32_v_f32m1(outptr0 + 4 * 3, _sum30, VL);
                    vse32_v_f32m1(outptr0 + 4 * 4, _sum40, VL);
                    vse32_v_f32m1(outptr0 + 4 * 5, _sum50, VL);
                    vse32_v_f32m1(outptr0 + 4 * 6, _sum60, VL);
                    vse32_v_f32m1(outptr0 + 4 * 7, _sum70, VL);

                    vse32_v_f32m1(outptr0 + out_hstep * 4, _sum01, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4, _sum11, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 2, _sum21, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 3, _sum31, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 4, _sum41, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 5, _sum51, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 6, _sum61, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 7, _sum71, VL);

                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose8x8_ps(_sum00, _sum01, _sum10, _sum11, _sum20, _sum21, _sum30, _sum31, _sum40, _sum41, _sum50, _sum51, _sum60, _sum61, _sum70, _sum71);

                    vse32_v_f32m1(outptr0, _sum00, VL);
                    vse32_v_f32m1(outptr0 + 4, _sum01, VL);
                    vse32_v_f32m1(outptr0 + out_hstep, _sum10, VL);
                    vse32_v_f32m1(outptr0 + out_hstep + 4, _sum11, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 2, _sum20, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 2 + 4, _sum21, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 3, _sum30, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 3 + 4, _sum31, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4, _sum40, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4, _sum41, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 5, _sum50, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 5 + 4, _sum51, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 6, _sum60, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 6 + 4, _sum61, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 7, _sum70, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 7 + 4, _sum71, VL);

                    outptr0 += 8;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum00, VL);
                vse32_v_f32m1(outptr + 4, _sum01, VL);
                vse32_v_f32m1(outptr + 4 * 2, _sum10, VL);
                vse32_v_f32m1(outptr + 4 * 3, _sum11, VL);
                vse32_v_f32m1(outptr + 4 * 4, _sum20, VL);
                vse32_v_f32m1(outptr + 4 * 5, _sum21, VL);
                vse32_v_f32m1(outptr + 4 * 6, _sum30, VL);
                vse32_v_f32m1(outptr + 4 * 7, _sum31, VL);
                vse32_v_f32m1(outptr + 4 * 8, _sum40, VL);
                vse32_v_f32m1(outptr + 4 * 9, _sum41, VL);
                vse32_v_f32m1(outptr + 4 * 10, _sum50, VL);
                vse32_v_f32m1(outptr + 4 * 11, _sum51, VL);
                vse32_v_f32m1(outptr + 4 * 12, _sum60, VL);
                vse32_v_f32m1(outptr + 4 * 13, _sum61, VL);
                vse32_v_f32m1(outptr + 4 * 14, _sum70, VL);
                vse32_v_f32m1(outptr + 4 * 15, _sum71, VL);
            }

            outptr += 64;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat32m1_t _sum00;
            vfloat32m1_t _sum01;
            vfloat32m1_t _sum10;
            vfloat32m1_t _sum11;
            vfloat32m1_t _sum20;
            vfloat32m1_t _sum21;
            vfloat32m1_t _sum30;
            vfloat32m1_t _sum31;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32_riscv(0.f);
                _sum01 = vdupq_n_f32_riscv(0.f);
                _sum10 = vdupq_n_f32_riscv(0.f);
                _sum11 = vdupq_n_f32_riscv(0.f);
                _sum20 = vdupq_n_f32_riscv(0.f);
                _sum21 = vdupq_n_f32_riscv(0.f);
                _sum30 = vdupq_n_f32_riscv(0.f);
                _sum31 = vdupq_n_f32_riscv(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32_riscv(pC[0]);
                        _sum01 = vdupq_n_f32_riscv(pC[0]);
                        _sum10 = vdupq_n_f32_riscv(pC[0]);
                        _sum11 = vdupq_n_f32_riscv(pC[0]);
                        _sum20 = vdupq_n_f32_riscv(pC[0]);
                        _sum21 = vdupq_n_f32_riscv(pC[0]);
                        _sum30 = vdupq_n_f32_riscv(pC[0]);
                        _sum31 = vdupq_n_f32_riscv(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vle32_v_f32m1(pC, VL);
                        _sum01 = vle32_v_f32m1(pC + 4, VL);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                        _sum20 = _sum00;
                        _sum21 = _sum01;
                        _sum30 = _sum00;
                        _sum31 = _sum01;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vle32_v_f32m1(pC, VL);
                        _sum01 = vle32_v_f32m1(pC + 4 * 1, VL);
                        _sum10 = vle32_v_f32m1(pC + 4 * 2, VL);
                        _sum11 = vle32_v_f32m1(pC + 4 * 3, VL);
                        _sum20 = vle32_v_f32m1(pC + 4 * 4, VL);
                        _sum21 = vle32_v_f32m1(pC + 4 * 5, VL);
                        _sum30 = vle32_v_f32m1(pC + 4 * 6, VL);
                        _sum31 = vle32_v_f32m1(pC + 4 * 7, VL);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32_riscv(pC[0]);
                        _sum10 = vdupq_n_f32_riscv(pC[1]);
                        _sum20 = vdupq_n_f32_riscv(pC[2]);
                        _sum30 = vdupq_n_f32_riscv(pC[3]);
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
                _sum00 = vle32_v_f32m1(outptr, VL);
                _sum01 = vle32_v_f32m1(outptr + 4 * 1, VL);
                _sum10 = vle32_v_f32m1(outptr + 4 * 2, VL);
                _sum11 = vle32_v_f32m1(outptr + 4 * 3, VL);
                _sum20 = vle32_v_f32m1(outptr + 4 * 4, VL);
                _sum21 = vle32_v_f32m1(outptr + 4 * 5, VL);
                _sum30 = vle32_v_f32m1(outptr + 4 * 6, VL);
                _sum31 = vle32_v_f32m1(outptr + 4 * 7, VL);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA0 = vle32_v_f32m1(pA, VL);
                vfloat32m1_t _pA1 = vle32_v_f32m1(pA + 4, VL);

                vfloat32m1_t _pB0 = vle32_v_f32m1(pB, VL);

                _sum00 = vfmaq_laneq_f32_riscv(_sum00, _pA0, _pB0, 0);
                _sum01 = vfmaq_laneq_f32_riscv(_sum01, _pA1, _pB0, 0);
                _sum10 = vfmaq_laneq_f32_riscv(_sum10, _pA0, _pB0, 1);
                _sum11 = vfmaq_laneq_f32_riscv(_sum11, _pA1, _pB0, 1);
                _sum20 = vfmaq_laneq_f32_riscv(_sum20, _pA0, _pB0, 2);
                _sum21 = vfmaq_laneq_f32_riscv(_sum21, _pA1, _pB0, 2);
                _sum30 = vfmaq_laneq_f32_riscv(_sum30, _pA0, _pB0, 3);
                _sum31 = vfmaq_laneq_f32_riscv(_sum31, _pA1, _pB0, 3);

                pA += 8;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum00, VL);
                    vse32_v_f32m1(outptr0 + 4, _sum10, VL);
                    vse32_v_f32m1(outptr0 + 4 * 2, _sum20, VL);
                    vse32_v_f32m1(outptr0 + 4 * 3, _sum30, VL);

                    vse32_v_f32m1(outptr0 + out_hstep * 4, _sum01, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4, _sum11, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 2, _sum21, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4 * 3, _sum31, VL);

                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose8x4_ps(_sum00, _sum01, _sum10, _sum11, _sum20, _sum21, _sum30, _sum31);

                    vse32_v_f32m1(outptr0, _sum00, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 1, _sum01, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 2, _sum10, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 3, _sum11, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4, _sum20, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 5, _sum21, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 6, _sum30, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 7, _sum31, VL);

                    outptr0 += 4;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum00, VL);
                vse32_v_f32m1(outptr + 4, _sum01, VL);
                vse32_v_f32m1(outptr + 4 * 2, _sum10, VL);
                vse32_v_f32m1(outptr + 4 * 3, _sum11, VL);
                vse32_v_f32m1(outptr + 4 * 4, _sum20, VL);
                vse32_v_f32m1(outptr + 4 * 5, _sum21, VL);
                vse32_v_f32m1(outptr + 4 * 6, _sum30, VL);
                vse32_v_f32m1(outptr + 4 * 7, _sum31, VL);
            }

            outptr += 32;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            vfloat32m1_t _sum00;
            vfloat32m1_t _sum01;
            vfloat32m1_t _sum10;
            vfloat32m1_t _sum11;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32_riscv(0.f);
                _sum01 = vdupq_n_f32_riscv(0.f);
                _sum10 = vdupq_n_f32_riscv(0.f);
                _sum11 = vdupq_n_f32_riscv(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32_riscv(pC[0]);
                        _sum01 = vdupq_n_f32_riscv(pC[0]);
                        _sum10 = vdupq_n_f32_riscv(pC[0]);
                        _sum11 = vdupq_n_f32_riscv(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vle32_v_f32m1(pC, VL);
                        _sum01 = vle32_v_f32m1(pC + 4, VL);
                        _sum10 = _sum00;
                        _sum11 = _sum01;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vle32_v_f32m1(pC, VL);
                        _sum01 = vle32_v_f32m1(pC + 4 * 1, VL);
                        _sum10 = vle32_v_f32m1(pC + 4 * 2, VL);
                        _sum11 = vle32_v_f32m1(pC + 4 * 3, VL);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32_riscv(pC[0]);
                        _sum10 = vdupq_n_f32_riscv(pC[1]);
                        _sum01 = _sum00;
                        _sum11 = _sum10;
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum00 = vle32_v_f32m1(outptr, VL);
                _sum01 = vle32_v_f32m1(outptr + 4 * 1, VL);
                _sum10 = vle32_v_f32m1(outptr + 4 * 2, VL);
                _sum11 = vle32_v_f32m1(outptr + 4 * 3, VL);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA0 = vle32_v_f32m1(pA, VL);
                vfloat32m1_t _pA1 = vle32_v_f32m1(pA + 4, VL);

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
                    vse32_v_f32m1(outptr0, _sum00, VL);
                    vse32_v_f32m1(outptr0 + 4, _sum10, VL);

                    vse32_v_f32m1(outptr0 + out_hstep * 4, _sum01, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4 + 4, _sum11, VL);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    float sum1[8];
                    vse32_v_f32m1(sum0, _sum00, VL);
                    vse32_v_f32m1(sum0 + 4, _sum01, VL);
                    vse32_v_f32m1(sum1, _sum10, VL);
                    vse32_v_f32m1(sum1 + 4, _sum11, VL);

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
                vse32_v_f32m1(outptr, _sum00, VL);
                vse32_v_f32m1(outptr + 4, _sum01, VL);
                vse32_v_f32m1(outptr + 4 * 2, _sum10, VL);
                vse32_v_f32m1(outptr + 4 * 3, _sum11, VL);
            }

            outptr += 16;
        }
        for (; jj < max_jj; jj += 1)
        {
            vfloat32m1_t _sum00;
            vfloat32m1_t _sum01;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32_riscv(0.f);
                _sum01 = vdupq_n_f32_riscv(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32_riscv(pC[0]);
                        _sum01 = vdupq_n_f32_riscv(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vle32_v_f32m1(pC, VL);
                        _sum01 = vle32_v_f32m1(pC + 4, VL);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum00 = vle32_v_f32m1(pC, VL);
                        _sum01 = vle32_v_f32m1(pC + 4, VL);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vdupq_n_f32_riscv(pC[0]);
                        _sum01 = _sum00;
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum00 = vle32_v_f32m1(outptr, VL);
                _sum01 = vle32_v_f32m1(outptr + 4 * 1, VL);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA0 = vle32_v_f32m1(pA, VL);
                vfloat32m1_t _pA1 = vle32_v_f32m1(pA + 4, VL);

                vfloat32m1_t _pB = vld1q_dup_f32(pB);

                _sum00 = vfmaq_f32(_sum00, _pA0, _pB);
                _sum01 = vfmaq_f32(_sum01, _pA1, _pB);

                pA += 8;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum00, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 4, _sum01, VL);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[8];
                    vse32_v_f32m1(sum0, _sum00, VL);
                    vse32_v_f32m1(sum0 + 4, _sum01, VL);

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
                vse32_v_f32m1(outptr, _sum00, VL);
                vse32_v_f32m1(outptr + 4, _sum01, VL);
            }

            outptr += 8;
        }

        pAT += max_kk * 8;
    }
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
        for (; jj + 11 < max_jj; jj += 12)
        {
            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;
            vfloat32m1_t _sum2;
            vfloat32m1_t _sum3;
            vfloat32m1_t _sum4;
            vfloat32m1_t _sum5;
            vfloat32m1_t _sum6;
            vfloat32m1_t _sum7;
            vfloat32m1_t _sum8;
            vfloat32m1_t _sum9;
            vfloat32m1_t _suma;
            vfloat32m1_t _sumb;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32_riscv(0.f);
                _sum1 = vdupq_n_f32_riscv(0.f);
                _sum2 = vdupq_n_f32_riscv(0.f);
                _sum3 = vdupq_n_f32_riscv(0.f);
                _sum4 = vdupq_n_f32_riscv(0.f);
                _sum5 = vdupq_n_f32_riscv(0.f);
                _sum6 = vdupq_n_f32_riscv(0.f);
                _sum7 = vdupq_n_f32_riscv(0.f);
                _sum8 = vdupq_n_f32_riscv(0.f);
                _sum9 = vdupq_n_f32_riscv(0.f);
                _suma = vdupq_n_f32_riscv(0.f);
                _sumb = vdupq_n_f32_riscv(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32_riscv(pC[0]);
                        _sum1 = vdupq_n_f32_riscv(pC[0]);
                        _sum2 = vdupq_n_f32_riscv(pC[0]);
                        _sum3 = vdupq_n_f32_riscv(pC[0]);
                        _sum4 = vdupq_n_f32_riscv(pC[0]);
                        _sum5 = vdupq_n_f32_riscv(pC[0]);
                        _sum6 = vdupq_n_f32_riscv(pC[0]);
                        _sum7 = vdupq_n_f32_riscv(pC[0]);
                        _sum8 = vdupq_n_f32_riscv(pC[0]);
                        _sum9 = vdupq_n_f32_riscv(pC[0]);
                        _suma = vdupq_n_f32_riscv(pC[0]);
                        _sumb = vdupq_n_f32_riscv(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m1(pC, VL);
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
                        _sum0 = vle32_v_f32m1(pC, VL);
                        _sum1 = vle32_v_f32m1(pC + 4, VL);
                        _sum2 = vle32_v_f32m1(pC + 8, VL);
                        _sum3 = vle32_v_f32m1(pC + 12, VL);
                        _sum4 = vle32_v_f32m1(pC + 16, VL);
                        _sum5 = vle32_v_f32m1(pC + 20, VL);
                        _sum6 = vle32_v_f32m1(pC + 24, VL);
                        _sum7 = vle32_v_f32m1(pC + 28, VL);
                        _sum8 = vle32_v_f32m1(pC + 32, VL);
                        _sum9 = vle32_v_f32m1(pC + 36, VL);
                        _suma = vle32_v_f32m1(pC + 40, VL);
                        _sumb = vle32_v_f32m1(pC + 44, VL);
                        pC += 48;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32_riscv(pC[0]);
                        _sum1 = vdupq_n_f32_riscv(pC[1]);
                        _sum2 = vdupq_n_f32_riscv(pC[2]);
                        _sum3 = vdupq_n_f32_riscv(pC[3]);
                        _sum4 = vdupq_n_f32_riscv(pC[4]);
                        _sum5 = vdupq_n_f32_riscv(pC[5]);
                        _sum6 = vdupq_n_f32_riscv(pC[6]);
                        _sum7 = vdupq_n_f32_riscv(pC[7]);
                        _sum8 = vdupq_n_f32_riscv(pC[8]);
                        _sum9 = vdupq_n_f32_riscv(pC[9]);
                        _suma = vdupq_n_f32_riscv(pC[10]);
                        _sumb = vdupq_n_f32_riscv(pC[11]);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m1(outptr, VL);
                _sum1 = vle32_v_f32m1(outptr + 4 * 1, VL);
                _sum2 = vle32_v_f32m1(outptr + 4 * 2, VL);
                _sum3 = vle32_v_f32m1(outptr + 4 * 3, VL);
                _sum4 = vle32_v_f32m1(outptr + 4 * 4, VL);
                _sum5 = vle32_v_f32m1(outptr + 4 * 5, VL);
                _sum6 = vle32_v_f32m1(outptr + 4 * 6, VL);
                _sum7 = vle32_v_f32m1(outptr + 4 * 7, VL);
                _sum8 = vle32_v_f32m1(outptr + 4 * 8, VL);
                _sum9 = vle32_v_f32m1(outptr + 4 * 9, VL);
                _suma = vle32_v_f32m1(outptr + 4 * 10, VL);
                _sumb = vle32_v_f32m1(outptr + 4 * 11, VL);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA = vle32_v_f32m1(pA, VL);
                vfloat32m1_t _pB0 = vle32_v_f32m1(pB, VL);
                vfloat32m1_t _pB1 = vle32_v_f32m1(pB + 4, VL);
                vfloat32m1_t _pB2 = vle32_v_f32m1(pB + 8, VL);

                _sum0 = vfmaq_laneq_f32_riscv(_sum0, _pA, _pB0, 0);
                _sum1 = vfmaq_laneq_f32_riscv(_sum1, _pA, _pB0, 1);
                _sum2 = vfmaq_laneq_f32_riscv(_sum2, _pA, _pB0, 2);
                _sum3 = vfmaq_laneq_f32_riscv(_sum3, _pA, _pB0, 3);
                _sum4 = vfmaq_laneq_f32_riscv(_sum4, _pA, _pB1, 0);
                _sum5 = vfmaq_laneq_f32_riscv(_sum5, _pA, _pB1, 1);
                _sum6 = vfmaq_laneq_f32_riscv(_sum6, _pA, _pB1, 2);
                _sum7 = vfmaq_laneq_f32_riscv(_sum7, _pA, _pB1, 3);
                _sum8 = vfmaq_laneq_f32_riscv(_sum8, _pA, _pB2, 0);
                _sum9 = vfmaq_laneq_f32_riscv(_sum9, _pA, _pB2, 1);
                _suma = vfmaq_laneq_f32_riscv(_suma, _pA, _pB2, 2);
                _sumb = vfmaq_laneq_f32_riscv(_sumb, _pA, _pB2, 3);

                pA += 4;
                pB += 12;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum0, VL);
                    vse32_v_f32m1(outptr0 + 4, _sum1, VL);
                    vse32_v_f32m1(outptr0 + 4 * 2, _sum2, VL);
                    vse32_v_f32m1(outptr0 + 4 * 3, _sum3, VL);
                    vse32_v_f32m1(outptr0 + 4 * 4, _sum4, VL);
                    vse32_v_f32m1(outptr0 + 4 * 5, _sum5, VL);
                    vse32_v_f32m1(outptr0 + 4 * 6, _sum6, VL);
                    vse32_v_f32m1(outptr0 + 4 * 7, _sum7, VL);
                    vse32_v_f32m1(outptr0 + 4 * 8, _sum8, VL);
                    vse32_v_f32m1(outptr0 + 4 * 9, _sum9, VL);
                    vse32_v_f32m1(outptr0 + 4 * 10, _suma, VL);
                    vse32_v_f32m1(outptr0 + 4 * 11, _sumb, VL);
                    outptr0 += 48;
                }
                if (out_elempack == 1)
                {
                    transpose4x12_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7, _sum8, _sum9, _suma, _sumb);

                    vse32_v_f32m1(outptr0, _sum0, VL);
                    vse32_v_f32m1(outptr0 + 4, _sum1, VL);
                    vse32_v_f32m1(outptr0 + 8, _sum2, VL);
                    vse32_v_f32m1(outptr0 + out_hstep, _sum3, VL);
                    vse32_v_f32m1(outptr0 + out_hstep + 4, _sum4, VL);
                    vse32_v_f32m1(outptr0 + out_hstep + 8, _sum5, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 2, _sum6, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 2 + 4, _sum7, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 2 + 8, _sum8, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 3, _sum9, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 3 + 4, _suma, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 3 + 8, _sumb, VL);
                    outptr0 += 12;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum0, VL);
                vse32_v_f32m1(outptr + 4, _sum1, VL);
                vse32_v_f32m1(outptr + 4 * 2, _sum2, VL);
                vse32_v_f32m1(outptr + 4 * 3, _sum3, VL);
                vse32_v_f32m1(outptr + 4 * 4, _sum4, VL);
                vse32_v_f32m1(outptr + 4 * 5, _sum5, VL);
                vse32_v_f32m1(outptr + 4 * 6, _sum6, VL);
                vse32_v_f32m1(outptr + 4 * 7, _sum7, VL);
                vse32_v_f32m1(outptr + 4 * 8, _sum8, VL);
                vse32_v_f32m1(outptr + 4 * 9, _sum9, VL);
                vse32_v_f32m1(outptr + 4 * 10, _suma, VL);
                vse32_v_f32m1(outptr + 4 * 11, _sumb, VL);
            }

            outptr += 48;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;
            vfloat32m1_t _sum2;
            vfloat32m1_t _sum3;
            vfloat32m1_t _sum4;
            vfloat32m1_t _sum5;
            vfloat32m1_t _sum6;
            vfloat32m1_t _sum7;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32_riscv(0.f);
                _sum1 = vdupq_n_f32_riscv(0.f);
                _sum2 = vdupq_n_f32_riscv(0.f);
                _sum3 = vdupq_n_f32_riscv(0.f);
                _sum4 = vdupq_n_f32_riscv(0.f);
                _sum5 = vdupq_n_f32_riscv(0.f);
                _sum6 = vdupq_n_f32_riscv(0.f);
                _sum7 = vdupq_n_f32_riscv(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32_riscv(pC[0]);
                        _sum1 = vdupq_n_f32_riscv(pC[0]);
                        _sum2 = vdupq_n_f32_riscv(pC[0]);
                        _sum3 = vdupq_n_f32_riscv(pC[0]);
                        _sum4 = vdupq_n_f32_riscv(pC[0]);
                        _sum5 = vdupq_n_f32_riscv(pC[0]);
                        _sum6 = vdupq_n_f32_riscv(pC[0]);
                        _sum7 = vdupq_n_f32_riscv(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m1(pC, VL);
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
                        _sum0 = vle32_v_f32m1(pC, VL);
                        _sum1 = vle32_v_f32m1(pC + 4, VL);
                        _sum2 = vle32_v_f32m1(pC + 8, VL);
                        _sum3 = vle32_v_f32m1(pC + 12, VL);
                        _sum4 = vle32_v_f32m1(pC + 16, VL);
                        _sum5 = vle32_v_f32m1(pC + 20, VL);
                        _sum6 = vle32_v_f32m1(pC + 24, VL);
                        _sum7 = vle32_v_f32m1(pC + 28, VL);
                        pC += 32;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32_riscv(pC[0]);
                        _sum1 = vdupq_n_f32_riscv(pC[1]);
                        _sum2 = vdupq_n_f32_riscv(pC[2]);
                        _sum3 = vdupq_n_f32_riscv(pC[3]);
                        _sum4 = vdupq_n_f32_riscv(pC[4]);
                        _sum5 = vdupq_n_f32_riscv(pC[5]);
                        _sum6 = vdupq_n_f32_riscv(pC[6]);
                        _sum7 = vdupq_n_f32_riscv(pC[7]);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m1(outptr, VL);
                _sum1 = vle32_v_f32m1(outptr + 4 * 1, VL);
                _sum2 = vle32_v_f32m1(outptr + 4 * 2, VL);
                _sum3 = vle32_v_f32m1(outptr + 4 * 3, VL);
                _sum4 = vle32_v_f32m1(outptr + 4 * 4, VL);
                _sum5 = vle32_v_f32m1(outptr + 4 * 5, VL);
                _sum6 = vle32_v_f32m1(outptr + 4 * 6, VL);
                _sum7 = vle32_v_f32m1(outptr + 4 * 7, VL);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA = vle32_v_f32m1(pA, VL);
                vfloat32m1_t _pB0 = vle32_v_f32m1(pB, VL);
                vfloat32m1_t _pB1 = vle32_v_f32m1(pB + 4, VL);

                _sum0 = vfmaq_laneq_f32_riscv(_sum0, _pA, _pB0, 0);
                _sum1 = vfmaq_laneq_f32_riscv(_sum1, _pA, _pB0, 1);
                _sum2 = vfmaq_laneq_f32_riscv(_sum2, _pA, _pB0, 2);
                _sum3 = vfmaq_laneq_f32_riscv(_sum3, _pA, _pB0, 3);
                _sum4 = vfmaq_laneq_f32_riscv(_sum4, _pA, _pB1, 0);
                _sum5 = vfmaq_laneq_f32_riscv(_sum5, _pA, _pB1, 1);
                _sum6 = vfmaq_laneq_f32_riscv(_sum6, _pA, _pB1, 2);
                _sum7 = vfmaq_laneq_f32_riscv(_sum7, _pA, _pB1, 3);

                pA += 4;
                pB += 8;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum0, VL);
                    vse32_v_f32m1(outptr0 + 4, _sum1, VL);
                    vse32_v_f32m1(outptr0 + 4 * 2, _sum2, VL);
                    vse32_v_f32m1(outptr0 + 4 * 3, _sum3, VL);
                    vse32_v_f32m1(outptr0 + 4 * 4, _sum4, VL);
                    vse32_v_f32m1(outptr0 + 4 * 5, _sum5, VL);
                    vse32_v_f32m1(outptr0 + 4 * 6, _sum6, VL);
                    vse32_v_f32m1(outptr0 + 4 * 7, _sum7, VL);
                    outptr0 += 32;
                }
                if (out_elempack == 1)
                {
                    transpose4x8_ps(_sum0, _sum1, _sum2, _sum3, _sum4, _sum5, _sum6, _sum7);

                    vse32_v_f32m1(outptr0, _sum0, VL);
                    vse32_v_f32m1(outptr0 + 4, _sum1, VL);
                    vse32_v_f32m1(outptr0 + out_hstep, _sum2, VL);
                    vse32_v_f32m1(outptr0 + out_hstep + 4, _sum3, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 2, _sum4, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 2 + 4, _sum5, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 3, _sum6, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 3 + 4, _sum7, VL);
                    outptr0 += 8;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum0, VL);
                vse32_v_f32m1(outptr + 4, _sum1, VL);
                vse32_v_f32m1(outptr + 4 * 2, _sum2, VL);
                vse32_v_f32m1(outptr + 4 * 3, _sum3, VL);
                vse32_v_f32m1(outptr + 4 * 4, _sum4, VL);
                vse32_v_f32m1(outptr + 4 * 5, _sum5, VL);
                vse32_v_f32m1(outptr + 4 * 6, _sum6, VL);
                vse32_v_f32m1(outptr + 4 * 7, _sum7, VL);
            }

            outptr += 32;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;
            vfloat32m1_t _sum2;
            vfloat32m1_t _sum3;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32_riscv(0.f);
                _sum1 = vdupq_n_f32_riscv(0.f);
                _sum2 = vdupq_n_f32_riscv(0.f);
                _sum3 = vdupq_n_f32_riscv(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32_riscv(pC[0]);
                        _sum1 = vdupq_n_f32_riscv(pC[0]);
                        _sum2 = vdupq_n_f32_riscv(pC[0]);
                        _sum3 = vdupq_n_f32_riscv(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m1(pC, VL);
                        _sum1 = _sum0;
                        _sum2 = _sum0;
                        _sum3 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vle32_v_f32m1(pC, VL);
                        _sum1 = vle32_v_f32m1(pC + 4, VL);
                        _sum2 = vle32_v_f32m1(pC + 8, VL);
                        _sum3 = vle32_v_f32m1(pC + 12, VL);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32_riscv(pC[0]);
                        _sum1 = vdupq_n_f32_riscv(pC[1]);
                        _sum2 = vdupq_n_f32_riscv(pC[2]);
                        _sum3 = vdupq_n_f32_riscv(pC[3]);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m1(outptr, VL);
                _sum1 = vle32_v_f32m1(outptr + 4 * 1, VL);
                _sum2 = vle32_v_f32m1(outptr + 4 * 2, VL);
                _sum3 = vle32_v_f32m1(outptr + 4 * 3, VL);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA = vle32_v_f32m1(pA, VL);
                vfloat32m1_t _pB = vle32_v_f32m1(pB, VL);

                _sum0 = vfmaq_laneq_f32_riscv(_sum0, _pA, _pB, 0);
                _sum1 = vfmaq_laneq_f32_riscv(_sum1, _pA, _pB, 1);
                _sum2 = vfmaq_laneq_f32_riscv(_sum2, _pA, _pB, 2);
                _sum3 = vfmaq_laneq_f32_riscv(_sum3, _pA, _pB, 3);
                pA += 4;
                pB += 4;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum0, VL);
                    vse32_v_f32m1(outptr0 + 4, _sum1, VL);
                    vse32_v_f32m1(outptr0 + 4 * 2, _sum2, VL);
                    vse32_v_f32m1(outptr0 + 4 * 3, _sum3, VL);
                    outptr0 += 16;
                }
                if (out_elempack == 1)
                {
                    transpose4x4_ps(_sum0, _sum1, _sum2, _sum3);

                    vse32_v_f32m1(outptr0, _sum0, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 1, _sum1, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 2, _sum2, VL);
                    vse32_v_f32m1(outptr0 + out_hstep * 3, _sum3, VL);
                    outptr0 += 4;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum0, VL);
                vse32_v_f32m1(outptr + 4, _sum1, VL);
                vse32_v_f32m1(outptr + 4 * 2, _sum2, VL);
                vse32_v_f32m1(outptr + 4 * 3, _sum3, VL);
            }

            outptr += 16;
        }
        for (; jj + 1 < max_jj; jj += 2)
        {
            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32_riscv(0.f);
                _sum1 = vdupq_n_f32_riscv(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32_riscv(pC[0]);
                        _sum1 = vdupq_n_f32_riscv(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m1(pC, VL);
                        _sum1 = _sum0;
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vle32_v_f32m1(pC, VL);
                        _sum1 = vle32_v_f32m1(pC + 4, VL);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32_riscv(pC[0]);
                        _sum1 = vdupq_n_f32_riscv(pC[1]);
                        pC += 2;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m1(outptr, VL);
                _sum1 = vle32_v_f32m1(outptr + 4, VL);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA = vle32_v_f32m1(pA, VL);
                float32x2_t _pB = vld1_f32(pB);

                _sum0 = vfmaq_lane_f32(_sum0, _pA, _pB, 0);
                _sum1 = vfmaq_lane_f32(_sum1, _pA, _pB, 1);


                pA += 4;
                pB += 2;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum0, VL);
                    vse32_v_f32m1(outptr0 + 4, _sum1, VL);
                    outptr0 += 8;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    float sum1[4];
                    vse32_v_f32m1(sum0, _sum0, VL);
                    vse32_v_f32m1(sum1, _sum1, VL);

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
                vse32_v_f32m1(outptr, _sum0, VL);
                vse32_v_f32m1(outptr + 4, _sum1, VL);
            }

            outptr += 8;
        }
        for (; jj < max_jj; jj += 1)
        {
            vfloat32m1_t _sum0;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32_riscv(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32_riscv(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vle32_v_f32m1(pC, VL);
                    }
                    if (broadcast_type_C == 3)
                    {
                        _sum0 = vle32_v_f32m1(pC, VL);
                        pC += 4;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vdupq_n_f32_riscv(pC[0]);
                        pC += 1;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m1(outptr, VL);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pA = vle32_v_f32m1(pA, VL);
                vfloat32m1_t _pB = vdupq_n_f32_riscv(pB[0]);

                _sum0 = vfmaq_f32(_sum0, _pA, _pB);

                pA += 4;
                pB += 1;
            }

            if (k_end)
            {
                if (out_elempack == 4)
                {
                    vse32_v_f32m1(outptr0, _sum0, VL);
                    outptr0 += 4;
                }
                if (out_elempack == 1)
                {
                    float sum0[4];
                    vse32_v_f32m1(sum0, _sum0, VL);

                    outptr0[0] = sum0[0];
                    outptr0[out_hstep] = sum0[1];
                    outptr0[out_hstep * 2] = sum0[2];
                    outptr0[out_hstep * 3] = sum0[3];
                    outptr0++;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum0, VL);
            }

            outptr += 4;
        }

        pAT += max_kk * 4;
    }
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
        for (; jj + 11 < max_jj; jj += 12)
        {
            vfloat32m1_t _sum00;
            vfloat32m1_t _sum01;
            vfloat32m1_t _sum02;
            vfloat32m1_t _sum10;
            vfloat32m1_t _sum11;
            vfloat32m1_t _sum12;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32_riscv(0.f);
                _sum01 = vdupq_n_f32_riscv(0.f);
                _sum02 = vdupq_n_f32_riscv(0.f);
                _sum10 = vdupq_n_f32_riscv(0.f);
                _sum11 = vdupq_n_f32_riscv(0.f);
                _sum12 = vdupq_n_f32_riscv(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32_riscv(pC[0]);
                        _sum01 = vdupq_n_f32_riscv(pC[0]);
                        _sum02 = vdupq_n_f32_riscv(pC[0]);
                        _sum10 = vdupq_n_f32_riscv(pC[0]);
                        _sum11 = vdupq_n_f32_riscv(pC[0]);
                        _sum12 = vdupq_n_f32_riscv(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vdupq_n_f32_riscv(pC[0]);
                        _sum01 = vdupq_n_f32_riscv(pC[0]);
                        _sum02 = vdupq_n_f32_riscv(pC[0]);
                        _sum10 = vdupq_n_f32_riscv(pC[1]);
                        _sum11 = vdupq_n_f32_riscv(pC[1]);
                        _sum12 = vdupq_n_f32_riscv(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4x2_t _tmp01 = vld2q_f32(pC);
                        float32x4x2_t _tmp23 = vld2q_f32(pC + 8);
                        float32x4x2_t _tmp45 = vld2q_f32(pC + 16);
                        _sum00 = vget_f32m1x4_f32m1(_tmp01, 0);
                        _sum01 = vget_f32m1x4_f32m1(_tmp23, 0);
                        _sum02 = vget_f32m1x4_f32m1(_tmp45, 0);
                        _sum10 = vget_f32m1x4_f32m1(_tmp01, 1);
                        _sum11 = vget_f32m1x4_f32m1(_tmp23, 1);
                        _sum12 = vget_f32m1x4_f32m1(_tmp45, 1);
                        pC += 24;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vle32_v_f32m1(pC, VL);
                        _sum01 = vle32_v_f32m1(pC + 4, VL);
                        _sum02 = vle32_v_f32m1(pC + 8, VL);
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
                _sum00 = vget_f32m1x4_f32m1(_tmp01, 0);
                _sum01 = vget_f32m1x4_f32m1(_tmp23, 0);
                _sum02 = vget_f32m1x4_f32m1(_tmp45, 0);
                _sum10 = vget_f32m1x4_f32m1(_tmp01, 1);
                _sum11 = vget_f32m1x4_f32m1(_tmp23, 1);
                _sum12 = vget_f32m1x4_f32m1(_tmp45, 1);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pB0 = vle32_v_f32m1(pB, VL);
                vfloat32m1_t _pB1 = vle32_v_f32m1(pB + 4, VL);
                vfloat32m1_t _pB2 = vle32_v_f32m1(pB + 8, VL);

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
                    vse32_v_f32m1(outptr0, _sum00, VL);
                    vse32_v_f32m1(outptr0 + 4, _sum01, VL);
                    vse32_v_f32m1(outptr0 + 8, _sum02, VL);
                    vse32_v_f32m1(outptr0 + out_hstep, _sum10, VL);
                    vse32_v_f32m1(outptr0 + out_hstep + 4, _sum11, VL);
                    vse32_v_f32m1(outptr0 + out_hstep + 8, _sum12, VL);
                    outptr0 += 12;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                vget_f32m1x4_f32m1(_tmp01, 0) = _sum00;
                vget_f32m1x4_f32m1(_tmp01, 1) = _sum10;
                float32x4x2_t _tmp23;
                vget_f32m1x4_f32m1(_tmp23, 0) = _sum01;
                vget_f32m1x4_f32m1(_tmp23, 1) = _sum11;
                float32x4x2_t _tmp45;
                vget_f32m1x4_f32m1(_tmp45, 0) = _sum02;
                vget_f32m1x4_f32m1(_tmp45, 1) = _sum12;
                vst2q_f32(outptr, _tmp01);
                vst2q_f32(outptr + 8, _tmp23);
                vst2q_f32(outptr + 16, _tmp45);
            }

            outptr += 24;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat32m1_t _sum00;
            vfloat32m1_t _sum01;
            vfloat32m1_t _sum10;
            vfloat32m1_t _sum11;

            if (k == 0)
            {
                _sum00 = vdupq_n_f32_riscv(0.f);
                _sum01 = vdupq_n_f32_riscv(0.f);
                _sum10 = vdupq_n_f32_riscv(0.f);
                _sum11 = vdupq_n_f32_riscv(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum00 = vdupq_n_f32_riscv(pC[0]);
                        _sum01 = vdupq_n_f32_riscv(pC[0]);
                        _sum10 = vdupq_n_f32_riscv(pC[0]);
                        _sum11 = vdupq_n_f32_riscv(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum00 = vdupq_n_f32_riscv(pC[0]);
                        _sum01 = vdupq_n_f32_riscv(pC[0]);
                        _sum10 = vdupq_n_f32_riscv(pC[1]);
                        _sum11 = vdupq_n_f32_riscv(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4x2_t _tmp01 = vld2q_f32(pC);
                        float32x4x2_t _tmp23 = vld2q_f32(pC + 8);
                        _sum00 = vget_f32m1x4_f32m1(_tmp01, 0);
                        _sum01 = vget_f32m1x4_f32m1(_tmp23, 0);
                        _sum10 = vget_f32m1x4_f32m1(_tmp01, 1);
                        _sum11 = vget_f32m1x4_f32m1(_tmp23, 1);
                        pC += 16;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum00 = vle32_v_f32m1(pC, VL);
                        _sum01 = vle32_v_f32m1(pC + 4, VL);
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
                _sum00 = vget_f32m1x4_f32m1(_tmp01, 0);
                _sum01 = vget_f32m1x4_f32m1(_tmp23, 0);
                _sum10 = vget_f32m1x4_f32m1(_tmp01, 1);
                _sum11 = vget_f32m1x4_f32m1(_tmp23, 1);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pB0 = vle32_v_f32m1(pB, VL);
                vfloat32m1_t _pB1 = vle32_v_f32m1(pB + 4, VL);

                float32x2_t _pA = vld1_f32(pA);
                _sum00 = vfmaq_lane_f32(_sum00, _pB0, _pA, 0);
                _sum01 = vfmaq_lane_f32(_sum01, _pB1, _pA, 0);
                _sum10 = vfmaq_lane_f32(_sum10, _pB0, _pA, 1);
                _sum11 = vfmaq_lane_f32(_sum11, _pB1, _pA, 1);
                pA += 2;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vse32_v_f32m1(outptr0, _sum00, VL);
                    vse32_v_f32m1(outptr0 + 4, _sum01, VL);
                    vse32_v_f32m1(outptr0 + out_hstep, _sum10, VL);
                    vse32_v_f32m1(outptr0 + out_hstep + 4, _sum11, VL);
                    outptr0 += 8;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                vget_f32m1x4_f32m1(_tmp01, 0) = _sum00;
                vget_f32m1x4_f32m1(_tmp01, 1) = _sum10;
                float32x4x2_t _tmp23;
                vget_f32m1x4_f32m1(_tmp23, 0) = _sum01;
                vget_f32m1x4_f32m1(_tmp23, 1) = _sum11;
                vst2q_f32(outptr, _tmp01);
                vst2q_f32(outptr + 8, _tmp23);
            }

            outptr += 16;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32_riscv(0.f);
                _sum1 = vdupq_n_f32_riscv(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0)
                    {
                        _sum0 = vdupq_n_f32_riscv(pC[0]);
                        _sum1 = vdupq_n_f32_riscv(pC[0]);
                    }
                    if (broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vdupq_n_f32_riscv(pC[0]);
                        _sum1 = vdupq_n_f32_riscv(pC[1]);
                    }
                    if (broadcast_type_C == 3)
                    {
                        float32x4x2_t _tmp01 = vld2q_f32(pC);
                        _sum0 = vget_f32m1x4_f32m1(_tmp01, 0);
                        _sum1 = vget_f32m1x4_f32m1(_tmp01, 1);
                        pC += 8;
                    }
                    if (broadcast_type_C == 4)
                    {
                        _sum0 = vle32_v_f32m1(pC, VL);
                        _sum1 = _sum0;
                        pC += 4;
                    }
                }
            }
            else
            {
                float32x4x2_t _tmp01 = vld2q_f32(outptr);
                _sum0 = vget_f32m1x4_f32m1(_tmp01, 0);
                _sum1 = vget_f32m1x4_f32m1(_tmp01, 1);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pB = vle32_v_f32m1(pB, VL);

                float32x2_t _pA = vld1_f32(pA);
                _sum0 = vfmaq_lane_f32(_sum0, _pB, _pA, 0);
                _sum1 = vfmaq_lane_f32(_sum1, _pB, _pA, 1);


                pA += 2;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vse32_v_f32m1(outptr0, _sum0, VL);
                    vse32_v_f32m1(outptr0 + out_hstep, _sum1, VL);
                    outptr0 += 4;
                }
            }
            else
            {
                float32x4x2_t _tmp01;
                vget_f32m1x4_f32m1(_tmp01, 0) = _sum0;
                vget_f32m1x4_f32m1(_tmp01, 1) = _sum1;
                vst2q_f32(outptr, _tmp01);
            }

            outptr += 8;
        }
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

        for (; jj + 11 < max_jj; jj += 12)
        {
            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;
            vfloat32m1_t _sum2;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32_riscv(0.f);
                _sum1 = vdupq_n_f32_riscv(0.f);
                _sum2 = vdupq_n_f32_riscv(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vdupq_n_f32_riscv(pC[0]);
                        _sum1 = vdupq_n_f32_riscv(pC[0]);
                        _sum2 = vdupq_n_f32_riscv(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = vle32_v_f32m1(pC, VL);
                        _sum1 = vle32_v_f32m1(pC + 4, VL);
                        _sum2 = vle32_v_f32m1(pC + 8, VL);
                        pC += 12;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m1(outptr, VL);
                _sum1 = vle32_v_f32m1(outptr + 4, VL);
                _sum2 = vle32_v_f32m1(outptr + 8, VL);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pB0 = vle32_v_f32m1(pB, VL);
                vfloat32m1_t _pB1 = vle32_v_f32m1(pB + 4, VL);
                vfloat32m1_t _pB2 = vle32_v_f32m1(pB + 8, VL);

                vfloat32m1_t _pA0 = vdupq_n_f32_riscv(pA[0]);

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
                    vse32_v_f32m1(outptr0, _sum0, VL);
                    vse32_v_f32m1(outptr0 + 4, _sum1, VL);
                    vse32_v_f32m1(outptr0 + 8, _sum2, VL);
                    outptr0 += 12;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum0, VL);
                vse32_v_f32m1(outptr + 4, _sum1, VL);
                vse32_v_f32m1(outptr + 8, _sum2, VL);
            }

            outptr += 12;
        }
        for (; jj + 7 < max_jj; jj += 8)
        {
            vfloat32m1_t _sum0;
            vfloat32m1_t _sum1;

            if (k == 0)
            {
                _sum0 = vdupq_n_f32_riscv(0.f);
                _sum1 = vdupq_n_f32_riscv(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum0 = vdupq_n_f32_riscv(pC[0]);
                        _sum1 = vdupq_n_f32_riscv(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum0 = vle32_v_f32m1(pC, VL);
                        _sum1 = vle32_v_f32m1(pC + 4, VL);
                        pC += 8;
                    }
                }
            }
            else
            {
                _sum0 = vle32_v_f32m1(outptr, VL);
                _sum1 = vle32_v_f32m1(outptr + 4, VL);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pB0 = vle32_v_f32m1(pB, VL);
                vfloat32m1_t _pB1 = vle32_v_f32m1(pB + 4, VL);

                vfloat32m1_t _pA0 = vdupq_n_f32_riscv(pA[0]);
                _sum0 = vfmaq_f32(_sum0, _pA0, _pB0);
                _sum1 = vfmaq_f32(_sum1, _pA0, _pB1);


                pA += 1;
                pB += 8;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vse32_v_f32m1(outptr0, _sum0, VL);
                    vse32_v_f32m1(outptr0 + 4, _sum1, VL);
                    outptr0 += 8;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum0, VL);
                vse32_v_f32m1(outptr + 4, _sum1, VL);
            }

            outptr += 8;
        }
        for (; jj + 3 < max_jj; jj += 4)
        {
            vfloat32m1_t _sum;

            if (k == 0)
            {
                _sum = vdupq_n_f32_riscv(0.f);

                if (pC)
                {
                    if (broadcast_type_C == 0 || broadcast_type_C == 1 || broadcast_type_C == 2)
                    {
                        _sum = vdupq_n_f32_riscv(pC[0]);
                    }
                    if (broadcast_type_C == 3 || broadcast_type_C == 4)
                    {
                        _sum = vle32_v_f32m1(pC, VL);
                        pC += 4;
                    }
                }
            }
            else
            {
                _sum = vle32_v_f32m1(outptr, VL);
            }

            const float* pA = pAT;
            int kk = 0;
            for (; kk < max_kk; kk += 1)
            {
                vfloat32m1_t _pB = vle32_v_f32m1(pB, VL);
                vfloat32m1_t _pA = vdupq_n_f32_riscv(pA[0]);

                _sum = vfmaq_f32(_sum, _pA, _pB);

                pA += 1;
                pB += 4;
            }

            if (k_end)
            {
                // if (out_elempack == 1)
                {
                    vse32_v_f32m1(outptr0, _sum, VL);
                    outptr0 += 4;
                }
            }
            else
            {
                vse32_v_f32m1(outptr, _sum, VL);
            }

            outptr += 4;
        }
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