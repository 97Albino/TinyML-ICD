/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Fri Sep 30 01:56:55 2022
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2022 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "d9d36ee6a560374a368a9b7d58445c38"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Fri Sep 30 01:56:55 2022"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_1_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1250, AI_STATIC)
/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  input_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 365, AI_STATIC)
/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  input_8_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 511, AI_STATIC)
/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  input_16_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 259, AI_STATIC)
/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  input_24_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 259, AI_STATIC)
/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  input_32_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 133, AI_STATIC)
/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  input_40_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 133, AI_STATIC)
/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  input_48_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 70, AI_STATIC)
/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  input_56_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 70, AI_STATIC)
/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  onnxReshape_46_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)
/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  node_49_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 2, AI_STATIC)
/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  input_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 85, AI_STATIC)
/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  input_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 5, AI_STATIC)
/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  input_8_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 35, AI_STATIC)
/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  input_8_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)
/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  input_16_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 77, AI_STATIC)
/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  input_16_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)
/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  input_24_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 49, AI_STATIC)
/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  input_24_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)
/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  input_32_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 77, AI_STATIC)
/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  input_32_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)
/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  input_40_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 49, AI_STATIC)
/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  input_40_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)
/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  input_48_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 77, AI_STATIC)
/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  input_48_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)
/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  input_56_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 49, AI_STATIC)
/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  input_56_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 7, AI_STATIC)
/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  node_49_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 14, AI_STATIC)
/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  node_49_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2, AI_STATIC)
/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  input_1_output, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1250, 1), AI_STRIDE_INIT(4, 4, 4, 4, 5000),
  1, &input_1_output_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  input_output, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 73, 1), AI_STRIDE_INIT(4, 4, 4, 20, 1460),
  1, &input_output_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  input_8_output, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 73, 1), AI_STRIDE_INIT(4, 4, 4, 28, 2044),
  1, &input_8_output_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  input_16_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 37, 1), AI_STRIDE_INIT(4, 4, 4, 28, 1036),
  1, &input_16_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  input_24_output, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 37, 1), AI_STRIDE_INIT(4, 4, 4, 28, 1036),
  1, &input_24_output_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  input_32_output, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 19, 1), AI_STRIDE_INIT(4, 4, 4, 28, 532),
  1, &input_32_output_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  input_40_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 19, 1), AI_STRIDE_INIT(4, 4, 4, 28, 532),
  1, &input_40_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  input_48_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 10, 1), AI_STRIDE_INIT(4, 4, 4, 28, 280),
  1, &input_48_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  input_56_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 10, 1), AI_STRIDE_INIT(4, 4, 4, 28, 280),
  1, &input_56_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  onnxReshape_46_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &onnxReshape_46_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  node_49_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &node_49_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  input_weights, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 17, 1, 5), AI_STRIDE_INIT(4, 4, 4, 68, 68),
  1, &input_weights_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  input_bias, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 5, 1, 1), AI_STRIDE_INIT(4, 4, 4, 20, 20),
  1, &input_bias_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  input_8_weights, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 5, 1, 1, 7), AI_STRIDE_INIT(4, 4, 20, 20, 20),
  1, &input_8_weights_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  input_8_bias, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &input_8_bias_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  input_16_weights, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 11, 1, 7), AI_STRIDE_INIT(4, 4, 4, 44, 44),
  1, &input_16_weights_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  input_16_bias, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &input_16_bias_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  input_24_weights, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 7, 1, 1, 7), AI_STRIDE_INIT(4, 4, 28, 28, 28),
  1, &input_24_weights_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  input_24_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &input_24_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  input_32_weights, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 1, 11, 1, 7), AI_STRIDE_INIT(4, 4, 4, 44, 44),
  1, &input_32_weights_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  input_32_bias, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &input_32_bias_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  input_40_weights, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 7, 1, 1, 7), AI_STRIDE_INIT(4, 4, 28, 28, 28),
  1, &input_40_weights_array, NULL)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  input_40_bias, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &input_40_bias_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  input_48_weights, AI_STATIC,
  23, 0x0,
  AI_SHAPE_INIT(4, 1, 11, 1, 7), AI_STRIDE_INIT(4, 4, 4, 44, 44),
  1, &input_48_weights_array, NULL)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  input_48_bias, AI_STATIC,
  24, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &input_48_bias_array, NULL)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  input_56_weights, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 7, 1, 1, 7), AI_STRIDE_INIT(4, 4, 28, 28, 28),
  1, &input_56_weights_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  input_56_bias, AI_STATIC,
  26, 0x0,
  AI_SHAPE_INIT(4, 1, 7, 1, 1), AI_STRIDE_INIT(4, 4, 4, 28, 28),
  1, &input_56_bias_array, NULL)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  node_49_weights, AI_STATIC,
  27, 0x0,
  AI_SHAPE_INIT(4, 7, 2, 1, 1), AI_STRIDE_INIT(4, 4, 28, 56, 56),
  1, &node_49_weights_array, NULL)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  node_49_bias, AI_STATIC,
  28, 0x0,
  AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &node_49_bias_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  node_49_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &onnxReshape_46_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &node_49_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &node_49_weights, &node_49_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  node_49_layer, 14,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &node_49_chain,
  NULL, &node_49_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  onnxReshape_46_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_56_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &onnxReshape_46_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  onnxReshape_46_layer, 12,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap,
  &onnxReshape_46_chain,
  NULL, &node_49_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(10, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(10, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_56_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_48_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_56_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &input_56_weights, &input_56_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_56_layer, 11,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d,
  &input_56_chain,
  NULL, &onnxReshape_46_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_48_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_40_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_48_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &input_48_weights, &input_48_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_48_layer, 10,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d,
  &input_48_chain,
  NULL, &input_56_layer, AI_STATIC, 
  .groups = 7, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 5, 0, 5), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_40_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_40_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &input_40_weights, &input_40_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_40_layer, 9,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d,
  &input_40_chain,
  NULL, &input_48_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_32_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &input_32_weights, &input_32_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_32_layer, 7,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d,
  &input_32_chain,
  NULL, &input_40_layer, AI_STATIC, 
  .groups = 7, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 5, 0, 5), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &input_24_weights, &input_24_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_24_layer, 6,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d,
  &input_24_chain,
  NULL, &input_32_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_16_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &input_16_weights, &input_16_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_16_layer, 4,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d,
  &input_16_chain,
  NULL, &input_24_layer, AI_STATIC, 
  .groups = 7, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 5, 0, 5), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_8_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &input_8_weights, &input_8_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_8_layer, 3,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d,
  &input_8_chain,
  NULL, &input_16_layer, AI_STATIC, 
  .groups = 1, 
  .nl_params = NULL, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  input_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_1_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &input_weights, &input_bias, NULL),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  input_layer, 1,
  CONV2D_TYPE, 0x0, NULL,
  conv2d, forward_conv2d,
  &input_chain,
  NULL, &input_8_layer, AI_STATIC, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(17, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 2272, 1, 1),
    2272, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 6460, 1, 1),
    6460, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_1_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &node_49_output),
  &input_layer, 0, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 2272, 1, 1),
      2272, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 6460, 1, 1),
      6460, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_1_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &node_49_output),
  &input_layer, 0, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/


/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    input_1_output_array.data = AI_PTR(g_network_activations_map[0] + 1460);
    input_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1460);
    
    input_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    input_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    input_8_output_array.data = AI_PTR(g_network_activations_map[0] + 1460);
    input_8_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1460);
    
    input_16_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    input_16_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    input_24_output_array.data = AI_PTR(g_network_activations_map[0] + 1036);
    input_24_output_array.data_start = AI_PTR(g_network_activations_map[0] + 1036);
    
    input_32_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    input_32_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    input_40_output_array.data = AI_PTR(g_network_activations_map[0] + 532);
    input_40_output_array.data_start = AI_PTR(g_network_activations_map[0] + 532);
    
    input_48_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    input_48_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    input_56_output_array.data = AI_PTR(g_network_activations_map[0] + 280);
    input_56_output_array.data_start = AI_PTR(g_network_activations_map[0] + 280);
    
    onnxReshape_46_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    onnxReshape_46_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    
    node_49_output_array.data = AI_PTR(g_network_activations_map[0] + 28);
    node_49_output_array.data_start = AI_PTR(g_network_activations_map[0] + 28);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    input_weights_array.format |= AI_FMT_FLAG_CONST;
    input_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    input_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    
    input_bias_array.format |= AI_FMT_FLAG_CONST;
    input_bias_array.data = AI_PTR(g_network_weights_map[0] + 340);
    input_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 340);
    
    input_8_weights_array.format |= AI_FMT_FLAG_CONST;
    input_8_weights_array.data = AI_PTR(g_network_weights_map[0] + 360);
    input_8_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 360);
    
    input_8_bias_array.format |= AI_FMT_FLAG_CONST;
    input_8_bias_array.data = AI_PTR(g_network_weights_map[0] + 500);
    input_8_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 500);
    
    input_16_weights_array.format |= AI_FMT_FLAG_CONST;
    input_16_weights_array.data = AI_PTR(g_network_weights_map[0] + 528);
    input_16_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 528);
    
    input_16_bias_array.format |= AI_FMT_FLAG_CONST;
    input_16_bias_array.data = AI_PTR(g_network_weights_map[0] + 836);
    input_16_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 836);
    
    input_24_weights_array.format |= AI_FMT_FLAG_CONST;
    input_24_weights_array.data = AI_PTR(g_network_weights_map[0] + 864);
    input_24_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 864);
    
    input_24_bias_array.format |= AI_FMT_FLAG_CONST;
    input_24_bias_array.data = AI_PTR(g_network_weights_map[0] + 1060);
    input_24_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1060);
    
    input_32_weights_array.format |= AI_FMT_FLAG_CONST;
    input_32_weights_array.data = AI_PTR(g_network_weights_map[0] + 1088);
    input_32_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1088);
    
    input_32_bias_array.format |= AI_FMT_FLAG_CONST;
    input_32_bias_array.data = AI_PTR(g_network_weights_map[0] + 1396);
    input_32_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1396);
    
    input_40_weights_array.format |= AI_FMT_FLAG_CONST;
    input_40_weights_array.data = AI_PTR(g_network_weights_map[0] + 1424);
    input_40_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1424);
    
    input_40_bias_array.format |= AI_FMT_FLAG_CONST;
    input_40_bias_array.data = AI_PTR(g_network_weights_map[0] + 1620);
    input_40_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1620);
    
    input_48_weights_array.format |= AI_FMT_FLAG_CONST;
    input_48_weights_array.data = AI_PTR(g_network_weights_map[0] + 1648);
    input_48_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1648);
    
    input_48_bias_array.format |= AI_FMT_FLAG_CONST;
    input_48_bias_array.data = AI_PTR(g_network_weights_map[0] + 1956);
    input_48_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 1956);
    
    input_56_weights_array.format |= AI_FMT_FLAG_CONST;
    input_56_weights_array.data = AI_PTR(g_network_weights_map[0] + 1984);
    input_56_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 1984);
    
    input_56_bias_array.format |= AI_FMT_FLAG_CONST;
    input_56_bias_array.data = AI_PTR(g_network_weights_map[0] + 2180);
    input_56_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2180);
    
    node_49_weights_array.format |= AI_FMT_FLAG_CONST;
    node_49_weights_array.data = AI_PTR(g_network_weights_map[0] + 2208);
    node_49_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 2208);
    
    node_49_bias_array.format |= AI_FMT_FLAG_CONST;
    node_49_bias_array.data = AI_PTR(g_network_weights_map[0] + 2264);
    node_49_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 2264);
    
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/


AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 18119,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 18119,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}

AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
    ai_error err;
    ai_network_params params;

    err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE)
        return err;
    if (ai_network_data_params_get(&params) != true) {
        err = ai_network_get_error(*network);
        return err;
    }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
    if (activations) {
        /* set the addresses of the activations buffers */
        for (int idx=0;idx<params.map_activations.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
    }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
    if (weights) {
        /* set the addresses of the weight buffers */
        for (int idx=0;idx<params.map_weights.size;idx++)
            AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
    }
#endif
    if (ai_network_init(*network, &params) != true) {
        err = ai_network_get_error(*network);
    }
    return err;
}

AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    ((ai_network *)network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}

AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if (!net_ctx) return false;

  ai_bool ok = true;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

