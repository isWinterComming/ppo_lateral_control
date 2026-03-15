/*
 * File: lat_mpc.c
 *
 * Code generated for Simulink model 'lat_mpc'.
 *
 * Model version                  : 1.4
 * Simulink Coder version         : 9.4 (R2020b) 29-Jul-2020
 * C/C++ source code generated on : Wed Jul 13 11:30:10 2022
 *
 * Target selection: ert.tlc
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#include "lat_mpc.h"
#include "lat_mpc_private.h"

/* Block signals (default storage) */
B_lat_mpc_T lat_mpc_B;

/* Real-time model */
static RT_MODEL_lat_mpc_T lat_mpc_M_;
RT_MODEL_lat_mpc_T *const lat_mpc_M = &lat_mpc_M_;
void LUf_boolean_Tint32_Treal32_T(real32_T outU[], real32_T outP[], int32_T N,
  boolean_T outS[])
{
  int32_T c;
  int32_T idx1;
  int32_T idx1_tmp;
  int32_T k;
  int32_T p;
  int32_T r;
  int32_T tmp;
  real32_T mTmp1;
  real32_T mTmp2;

  /* S-Function (sdsplu2): '<S4>/LU Factorization' */
  /* initialize status output */
  outS[0U] = false;

  /* initialize row-pivot indices */
  for (k = 0; k < N; k++) {
    outP[k] = (real32_T)(k + 1);
  }

  for (k = 0; k < N; k++) {
    p = k;

    /* Scan the lower triangular part of this column only. */
    /* Record row of largest value */
    idx1_tmp = k * N;
    mTmp1 = outU[idx1_tmp + k];
    if (mTmp1 < 0.0F) {
      mTmp1 = -mTmp1;
    }

    for (r = k + 1; r < N; r++) {
      mTmp2 = outU[idx1_tmp + r];
      if (mTmp2 < 0.0F) {
        mTmp2 = -mTmp2;
      }

      if (mTmp2 > mTmp1) {
        p = r;
        mTmp1 = mTmp2;
      }
    }

    /* swap rows if required */
    if (p != k) {
      for (c = 0; c < N; c++) {
        idx1 = c * N;
        r = idx1 + p;
        mTmp1 = outU[r];
        tmp = idx1 + k;
        outU[r] = outU[tmp];
        outU[tmp] = mTmp1;
      }

      /* swap pivot row indices */
      mTmp1 = outP[p];
      outP[p] = outP[k];
      outP[k] = mTmp1;
    }

    idx1 = k * N + k;
    if (outU[idx1] == 0.0F) {
      outS[0U] = true;
    } else {
      for (r = k + 1; r < N; r++) {
        tmp = idx1_tmp + r;
        outU[tmp] /= outU[idx1];
      }

      for (c = k + 1; c < N; c++) {
        idx1 = c * N;
        for (r = k + 1; r < N; r++) {
          tmp = idx1 + r;
          outU[tmp] -= outU[idx1_tmp + r] * outU[idx1 + k];
        }
      }
    }
  }

  /* End of S-Function (sdsplu2): '<S4>/LU Factorization' */
}

/* Model step function */
void lat_mpc_step(real32_T arg_v_ego, real32_T arg_T_in[30], real32_T arg_y_pts
                  [30], real32_T arg_head_pts[30], real32_T arg_k_pos, real32_T
                  arg_k_heading, real32_T arg_k_control, real32_T arg_u_delay,
                  uint8_T *arg_is_MPC_valid, real32_T arg_mpc_solution[30])
{
  static const int8_T ab[900] = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
  };

  static const int8_T y[3] = { 0, 0, 1 };

  int32_T g_tmp;
  int32_T h_tmp;
  int32_T i;
  int32_T k;
  int32_T l;
  int32_T loop_ub;
  int32_T loop_ub_0;
  int32_T p;
  int32_T s_idx_1;
  int32_T tmp_2_tmp;
  real32_T K[900];
  real32_T rtb_BackwardSubstitution[900];
  real32_T rtb_LUFactorization_o1[900];
  real32_T A_set[270];
  real32_T M[270];
  real32_T X_ref[90];
  real32_T tmp[90];
  real32_T rtb_LUFactorization_o2[30];
  real32_T A_set_0[9];
  real32_T tmp_1[9];
  real32_T tmp_2[9];
  real32_T y_ref[3];
  real32_T tmp_data[2];
  real32_T Ts;
  real32_T theta_ref;
  int8_T B_set[90];
  int8_T b_I[9];
  boolean_T rtb_LogicalOperator;

  /* MATLAB Function: '<S1>/MPC_LatController' incorporates:
   *  Inport: '<Root>/T_in'
   *  Inport: '<Root>/head_pts'
   *  Inport: '<Root>/k_control'
   *  Inport: '<Root>/k_heading'
   *  Inport: '<Root>/k_pos'
   *  Inport: '<Root>/v_ego'
   *  Inport: '<Root>/y_pts'
   */
  memset(&lat_mpc_B.Q[0], 0, 8100U * sizeof(real32_T));
  memset(&M[0], 0, 270U * sizeof(real32_T));
  memset(&lat_mpc_B.K[0], 0, 2700U * sizeof(real32_T));
  memset(&X_ref[0], 0, 90U * sizeof(real32_T));
  y_ref[2] = 0.0F;
  tmp_data[0] = 0.0F;
  tmp_1[0] = arg_k_pos;
  tmp_1[3] = 0.0F;
  tmp_1[6] = 0.0F;
  tmp_1[1] = 0.0F;
  tmp_1[4] = arg_k_heading;
  tmp_1[7] = 0.0F;
  tmp_1[2] = 0.0F;
  tmp_1[5] = 0.0F;
  tmp_1[8] = 0.0F;
  for (i = 0; i < 30; i++) {
    if (i + 1 == 1) {
      Ts = arg_T_in[0];
    } else {
      Ts = arg_T_in[i] - arg_T_in[i - 1];
    }

    theta_ref = arg_head_pts[i];
    g_tmp = i * 3;
    h_tmp = (i + 1) * 3;
    if (g_tmp + 1 > h_tmp) {
      l = 0;
      k = 0;
    } else {
      l = g_tmp;
      k = h_tmp;
    }

    y_ref[0] = arg_y_pts[i];
    y_ref[1] = theta_ref;
    loop_ub = k - l;
    for (k = 0; k < loop_ub; k++) {
      X_ref[l + k] = y_ref[k];
    }

    tmp_data[1] = Ts;
    tmp_2[0] = 1.0F;
    tmp_2[3] = arg_v_ego * cosf(theta_ref) * Ts;
    tmp_2[1] = 0.0F;
    tmp_2[4] = 1.0F;
    for (k = 0; k < 2; k++) {
      tmp_2[k + 6] = tmp_data[k];
    }

    tmp_2[2] = 0.0F;
    tmp_2[5] = 0.0F;
    tmp_2[8] = 1.0F;
    for (k = 0; k < 3; k++) {
      l = i + 90 * k;
      A_set[l] = tmp_2[3 * k];
      A_set[l + 30] = tmp_2[3 * k + 1];
      A_set[l + 60] = tmp_2[3 * k + 2];
      B_set[i + 30 * k] = y[k];
    }

    if (g_tmp + 1 > h_tmp) {
      l = 0;
      k = 0;
      g_tmp = 0;
      h_tmp = 0;
    } else {
      l = g_tmp;
      k = h_tmp;
    }

    loop_ub = k - l;
    s_idx_1 = h_tmp - g_tmp;
    for (k = 0; k < s_idx_1; k++) {
      for (h_tmp = 0; h_tmp < loop_ub; h_tmp++) {
        lat_mpc_B.Q[(l + h_tmp) + 90 * (g_tmp + k)] = tmp_1[loop_ub * k + h_tmp];
      }
    }
  }

  for (k = 0; k < 9; k++) {
    b_I[k] = 0;
  }

  b_I[0] = 1;
  b_I[4] = 1;
  b_I[8] = 1;
  for (k = 0; k < 9; k++) {
    tmp_1[k] = b_I[k];
    b_I[k] = 0;
  }

  b_I[0] = 1;
  b_I[4] = 1;
  b_I[8] = 1;
  for (k = 0; k < 9; k++) {
    tmp_2[k] = b_I[k];
  }

  for (i = 0; i < 30; i++) {
    for (k = 0; k < 3; k++) {
      for (h_tmp = 0; h_tmp < 3; h_tmp++) {
        l = k + 3 * h_tmp;
        A_set_0[l] = 0.0F;
        g_tmp = 30 * k + i;
        A_set_0[l] += A_set[g_tmp] * tmp_1[3 * h_tmp];
        A_set_0[l] += A_set[g_tmp + 90] * tmp_1[3 * h_tmp + 1];
        A_set_0[l] += A_set[g_tmp + 180] * tmp_1[3 * h_tmp + 2];
      }
    }

    for (k = 0; k < 9; k++) {
      tmp_1[k] = A_set_0[k];
    }

    g_tmp = i * 3;
    loop_ub = (i + 1) * 3;
    if (g_tmp + 1 > loop_ub) {
      l = 0;
      k = 0;
    } else {
      l = g_tmp;
      k = loop_ub;
    }

    s_idx_1 = k - l;
    for (k = 0; k < 3; k++) {
      for (h_tmp = 0; h_tmp < s_idx_1; h_tmp++) {
        M[(l + h_tmp) + 90 * k] = tmp_1[s_idx_1 * k + h_tmp];
      }
    }

    l = (int32_T)(((-1.0 - ((real_T)i + 1.0)) + 1.0) / -1.0);
    if (0 <= l - 1) {
      if (g_tmp + 1 > loop_ub) {
        p = 0;
        loop_ub = 0;
      } else {
        p = g_tmp;
      }

      loop_ub_0 = loop_ub - p;
    }

    for (loop_ub = 0; loop_ub < l; loop_ub++) {
      g_tmp = (i - loop_ub) + 1;
      if (i + 1 == g_tmp) {
        for (k = 0; k < 9; k++) {
          b_I[k] = 0;
        }

        b_I[0] = 1;
        b_I[4] = 1;
        b_I[8] = 1;
        for (k = 0; k < 9; k++) {
          tmp_2[k] = b_I[k];
        }
      } else {
        for (k = 0; k < 3; k++) {
          for (h_tmp = 0; h_tmp < 3; h_tmp++) {
            s_idx_1 = k + 3 * h_tmp;
            A_set_0[s_idx_1] = 0.0F;
            tmp_2_tmp = 90 * h_tmp + g_tmp;
            A_set_0[s_idx_1] += A_set[tmp_2_tmp] * tmp_2[k];
            A_set_0[s_idx_1] += A_set[tmp_2_tmp + 30] * tmp_2[k + 3];
            A_set_0[s_idx_1] += A_set[tmp_2_tmp + 60] * tmp_2[k + 6];
          }
        }

        for (k = 0; k < 9; k++) {
          tmp_2[k] = A_set_0[k];
        }
      }

      for (k = 0; k < 3; k++) {
        y_ref[k] = tmp_2[k + 6] * (real32_T)B_set[g_tmp + 59] + (tmp_2[k + 3] *
          (real32_T)B_set[g_tmp + 29] + (real32_T)B_set[g_tmp - 1] * tmp_2[k]);
      }

      for (k = 0; k < loop_ub_0; k++) {
        lat_mpc_B.K[(p + k) + 90 * (g_tmp - 1)] = y_ref[k];
      }
    }
  }

  for (k = 0; k < 30; k++) {
    for (h_tmp = 0; h_tmp < 90; h_tmp++) {
      p = k + 30 * h_tmp;
      lat_mpc_B.K_m[p] = 0.0F;
      for (loop_ub_0 = 0; loop_ub_0 < 90; loop_ub_0++) {
        lat_mpc_B.K_m[p] += lat_mpc_B.K[90 * k + loop_ub_0] * lat_mpc_B.Q[90 *
          h_tmp + loop_ub_0];
      }
    }

    for (h_tmp = 0; h_tmp < 30; h_tmp++) {
      Ts = 0.0F;
      for (loop_ub_0 = 0; loop_ub_0 < 90; loop_ub_0++) {
        Ts += lat_mpc_B.K_m[30 * loop_ub_0 + k] * lat_mpc_B.K[90 * h_tmp +
          loop_ub_0];
      }

      p = 30 * h_tmp + k;
      K[p] = (real32_T)ab[p] * arg_k_control + Ts;
    }
  }

  for (k = 0; k < 900; k++) {
    /* S-Function (sdsplu2): '<S4>/LU Factorization' */
    rtb_LUFactorization_o1[k] = 2.0F * K[k];
  }

  /* S-Function (sdsplu2): '<S4>/LU Factorization' */
  LUf_boolean_Tint32_Treal32_T(&rtb_LUFactorization_o1[0],
    &rtb_LUFactorization_o2[0], 30, &rtb_LogicalOperator);

  /* Logic: '<S4>/Logical Operator' */
  rtb_LogicalOperator = !rtb_LogicalOperator;

  /* Outport: '<Root>/is_MPC_valid' incorporates:
   *  DataTypeConversion: '<S4>/Data Type Conversion'
   */
  *arg_is_MPC_valid = rtb_LogicalOperator;

  /* If: '<S4>/If' incorporates:
   *  Constant: '<S2>/Constant'
   *  DataTypeConversion: '<S4>/Data Type Conversion'
   *  Inport: '<S6>/In1'
   *  Merge: '<S4>/Merge'
   */
  if (rtb_LogicalOperator) {
    /* Outputs for IfAction SubSystem: '<S4>/Subsystem' incorporates:
     *  ActionPort: '<S5>/Action Port'
     */
    /* S-Function (sdspperm2): '<S5>/Permute Matrix' incorporates:
     *  Constant: '<S2>/Constant'
     *  S-Function (sdspfbsub2): '<S5>/Backward Substitution'
     */
    for (p = 0; p < 30; p++) {
      loop_ub_0 = (int32_T)floorf(rtb_LUFactorization_o2[p]) - 1;
      if (loop_ub_0 < 0) {
        loop_ub_0 = 0;
      } else {
        if (loop_ub_0 >= 30) {
          loop_ub_0 = 29;
        }
      }

      for (k = 0; k < 30; k++) {
        rtb_BackwardSubstitution[30 * k + p] = lat_mpc_ConstP.Constant_Value[30 *
          k + loop_ub_0];
      }
    }

    /* End of S-Function (sdspperm2): '<S5>/Permute Matrix' */

    /* S-Function (sdspfbsub2): '<S5>/Forward Substitution' incorporates:
     *  S-Function (sdspfbsub2): '<S5>/Backward Substitution'
     *  S-Function (sdsplu2): '<S4>/LU Factorization'
     */
    for (g_tmp = 0; g_tmp < 30; g_tmp++) {
      p = g_tmp * 30;
      for (i = 0; i < 29; i++) {
        loop_ub_0 = i + 1;
        h_tmp = (p + i) + 1;
        Ts = rtb_BackwardSubstitution[h_tmp];
        for (k = 0; k < i + 1; k++) {
          Ts -= rtb_BackwardSubstitution[p + k] *
            rtb_LUFactorization_o1[loop_ub_0];
          loop_ub_0 += 30;
        }

        rtb_BackwardSubstitution[h_tmp] = Ts;
      }
    }

    /* End of S-Function (sdspfbsub2): '<S5>/Forward Substitution' */

    /* S-Function (sdspfbsub2): '<S5>/Backward Substitution' incorporates:
     *  S-Function (sdsplu2): '<S4>/LU Factorization'
     */
    for (g_tmp = 0; g_tmp < 30; g_tmp++) {
      p = g_tmp * 30;
      rtb_BackwardSubstitution[p + 29] /= rtb_LUFactorization_o1[899];
      for (i = 28; i >= 0; i--) {
        loop_ub_0 = i + 870;
        h_tmp = p + i;
        Ts = rtb_BackwardSubstitution[h_tmp];
        for (k = 29; k > i; k--) {
          Ts -= rtb_BackwardSubstitution[p + k] *
            rtb_LUFactorization_o1[loop_ub_0];
          loop_ub_0 -= 30;
        }

        rtb_BackwardSubstitution[h_tmp] = Ts / rtb_LUFactorization_o1[loop_ub_0];
      }
    }

    /* End of S-Function (sdspfbsub2): '<S5>/Backward Substitution' */
    /* End of Outputs for SubSystem: '<S4>/Subsystem' */
  } else {
    /* Outputs for IfAction SubSystem: '<S4>/Subsystem1' incorporates:
     *  ActionPort: '<S6>/Action Port'
     */
    memcpy(&rtb_BackwardSubstitution[0], &lat_mpc_ConstP.Constant_Value[0], 900U
           * sizeof(real32_T));

    /* End of Outputs for SubSystem: '<S4>/Subsystem1' */
  }

  /* End of If: '<S4>/If' */

  /* MATLAB Function: '<S1>/MPC_LatController' incorporates:
   *  Inport: '<Root>/u_delay'
   */
  for (k = 0; k < 90; k++) {
    tmp[k] = ((M[k + 180] * arg_u_delay + (M[k + 90] * 0.0F + M[k] * 0.0F)) -
              X_ref[k]) * 2.0F;
  }

  for (k = 0; k < 90; k++) {
    X_ref[k] = 0.0F;
    for (h_tmp = 0; h_tmp < 90; h_tmp++) {
      X_ref[k] += lat_mpc_B.Q[90 * k + h_tmp] * tmp[h_tmp];
    }
  }

  for (k = 0; k < 30; k++) {
    rtb_LUFactorization_o2[k] = 0.0F;
    for (h_tmp = 0; h_tmp < 90; h_tmp++) {
      rtb_LUFactorization_o2[k] += lat_mpc_B.K[90 * k + h_tmp] * X_ref[h_tmp];
    }
  }

  /* Outport: '<Root>/mpc_solution' incorporates:
   *  Gain: '<S1>/Gain1'
   *  Merge: '<S4>/Merge'
   *  Product: '<S1>/Product'
   */
  for (k = 0; k < 30; k++) {
    arg_mpc_solution[k] = 0.0F;
    for (h_tmp = 0; h_tmp < 30; h_tmp++) {
      arg_mpc_solution[k] += rtb_BackwardSubstitution[30 * h_tmp + k] *
        -rtb_LUFactorization_o2[h_tmp];
    }
  }

  /* End of Outport: '<Root>/mpc_solution' */
}

/* Model initialize function */
void lat_mpc_initialize(void)
{
  /* (no initialization code required) */
}

/* Model terminate function */
void lat_mpc_terminate(void)
{
  /* (no terminate code required) */
}

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
