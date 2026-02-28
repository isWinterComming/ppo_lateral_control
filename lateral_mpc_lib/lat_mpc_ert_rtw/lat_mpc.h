/*
 * File: lat_mpc.h
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

#ifndef RTW_HEADER_lat_mpc_h_
#define RTW_HEADER_lat_mpc_h_
#include <math.h>
#include <string.h>
#ifndef lat_mpc_COMMON_INCLUDES_
#define lat_mpc_COMMON_INCLUDES_
#include "rtwtypes.h"
#endif                                 /* lat_mpc_COMMON_INCLUDES_ */

#include "lat_mpc_types.h"

/* Macros for accessing real-time model data structure */
#ifndef rtmGetErrorStatus
#define rtmGetErrorStatus(rtm)         ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
#define rtmSetErrorStatus(rtm, val)    ((rtm)->errorStatus = (val))
#endif

/* Block signals (default storage) */
typedef struct {
  real32_T Q[8100];
  real32_T K[2700];
  real32_T K_m[2700];
} B_lat_mpc_T;

/* Constant parameters (default storage) */
typedef struct {
  /* Computed Parameter: Constant_Value
   * Referenced by: '<S2>/Constant'
   */
  real32_T Constant_Value[900];
} ConstP_lat_mpc_T;

/* Real-time Model Data Structure */
struct tag_RTM_lat_mpc_T {
  const char_T *errorStatus;
};

/* Block signals (default storage) */
extern B_lat_mpc_T lat_mpc_B;

/* Constant parameters (default storage) */
extern const ConstP_lat_mpc_T lat_mpc_ConstP;

/* Model entry point functions */
extern void lat_mpc_initialize(void);
extern void lat_mpc_terminate(void);

/* Customized model step function */
extern void lat_mpc_step(real32_T arg_v_ego, real32_T arg_T_in[30], real32_T
  arg_y_pts[30], real32_T arg_head_pts[30], real32_T arg_k_pos, real32_T
  arg_k_heading, real32_T arg_k_control, real32_T arg_u_delay, uint8_T
  *arg_is_MPC_valid, real32_T arg_mpc_solution[30]);

/* Real-time Model object */
extern RT_MODEL_lat_mpc_T *const lat_mpc_M;

/*-
 * These blocks were eliminated from the model due to optimizations:
 *
 * Block '<S7>/Check Signal Attributes' : Unused code path elimination
 * Block '<S1>/Reshape' : Reshape block reduction
 */

/*-
 * The generated code includes comments that allow you to trace directly
 * back to the appropriate location in the model.  The basic format
 * is <system>/block_name, where system is the system number (uniquely
 * assigned by Simulink) and block_name is the name of the block.
 *
 * Use the MATLAB hilite_system command to trace the generated code back
 * to the model.  For example,
 *
 * hilite_system('<S3>')    - opens system 3
 * hilite_system('<S3>/Kp') - opens and selects block Kp which resides in S3
 *
 * Here is the system hierarchy for this model
 *
 * '<Root>' : 'lat_mpc'
 * '<S1>'   : 'lat_mpc/Subsystem'
 * '<S2>'   : 'lat_mpc/Subsystem/Hissien_Matrix_Inverse'
 * '<S3>'   : 'lat_mpc/Subsystem/MPC_LatController'
 * '<S4>'   : 'lat_mpc/Subsystem/Hissien_Matrix_Inverse/LU Solver'
 * '<S5>'   : 'lat_mpc/Subsystem/Hissien_Matrix_Inverse/LU Solver/Subsystem'
 * '<S6>'   : 'lat_mpc/Subsystem/Hissien_Matrix_Inverse/LU Solver/Subsystem1'
 * '<S7>'   : 'lat_mpc/Subsystem/Hissien_Matrix_Inverse/LU Solver/Subsystem/Check Signal Attributes2'
 */
#endif                                 /* RTW_HEADER_lat_mpc_h_ */

/*
 * File trailer for generated code.
 *
 * [EOF]
 */
