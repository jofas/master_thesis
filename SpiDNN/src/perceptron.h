#include "spiDNN.h"

#define BIAS weights[n_weights - 1]
#define N_POTENTIALS n_weights - 1


/* structs and enums */

//! human readable definitions of each region in SDRAM
typedef enum regions_e {
    __SYSTEM_REGION,
    BASE_PARAMS,
    WEIGHTS,
    SOFTMAX_PARAMS,
    TRAINABLE_PARAMS,
    NEXT_LAYER_WEIGHTS,
} regions_e;

//! human readable definitions of the activation functions (except
//! softmax, which is handled by another type of perceptron)
typedef enum activations_e {
  IDENTITY,
  RELU,
  SIGMOID,
  TANH,
  SOFTMAX,
} activations_e;

//! definitions of each element in the base_params region
typedef struct base_params_region {
  uint32_t forward_key;
  uint32_t min_pre_key;
  uint32_t timer_offset;
  uint32_t n_weights;
  uint32_t activation_function_id;
} base_params_region_t;


/* global variables */

uint forward_key;

uint n_weights;

uint activation_function_id;

float *weights;

float *potentials;

float potential;

float *weights_sdram;
base_params_region_t *base_params_sdram;


/* functions */

void generate_potential() {
  for (uint i = 0; i < n_potentials; i++) {
    potential += potentials[i] * weights[i];
  }
  potential += BIAS;
}

void weights_init() {
  weights_sdram = data_specification_get_region(WEIGHTS, data_spec_meta);

  weights = (float *)malloc(sizeof(float) * n_weights);

  sark_mem_cpy((void *)weights, (void *)weights_sdram,
    sizeof(float) * n_weights);
}


/* function which has to be implemented by a machine vertex including
 * spiDNN.h */
void __init_base_params(
    uint32_t *timer_offset, uint *n_potentials, uint *min_pre_key)
{
  base_params_sdram = data_specification_get_region(BASE_PARAMS, data_spec_meta);

  forward_key = base_params_sdram->forward_key;
  n_weights = base_params_sdram->n_weights;
  activation_function_id = base_params_sdram->activation_function_id;

  *timer_offset = base_params_sdram->timer_offset;
  *n_potentials = n_weights - 1;
  *min_pre_key = base_params_sdram->min_pre_key;
}
