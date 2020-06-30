#include "spiDNN.h"

#define BIAS weights[n_weights - 1]
#define N_POTENTIALS n_weights - 1


/* structs and enums */

//! human readable definitions of each region in SDRAM
typedef enum regions_e {
    __SYSTEM_REGION,
    BASE_PARAMS,
    WEIGHTS,
    INSTANCE_PARAMS,
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
  //SOFTMAX,
} activations_e;

//! definitions of each element in the base_params region
typedef struct base_params_region {
  uint32_t forward_key;
  uint32_t min_pre_key;
  uint32_t timer_offset;
  uint32_t n_weights;
} base_params_region_t;

//! definitions of each element in the instance_params region, when
//! perceptron has other activation function than softmax
typedef struct perceptron_params_region {
  uint32_t activation_function_id;
} perceptron_params_region_t;

//! definitions of each element in the instance_params region, when
//! perceptron has softmax as its activation
typedef struct softmax_params_region {
  uint32_t key;
  uint32_t min_layer_key;
  uint32_t layer_size;
} softmax_params_region_t;

//! definitions of each element in the trainable_params region
typedef struct trainable_params_region {
  uint32_t batch_size;
  uint32_t backward_key;
  uint32_t min_next_key;
  uint32_t n_errors;
  uint32_t is_output_layer;
  float learning_rate;
} trainable_params_region_t;


/* global variables */

uint forward_key;

uint n_weights;

float *weights;

float *potentials;

float potential;

float *weights_sdram;
base_params_region_t *base_params_sdram;


/* instance variables */
#ifdef softmax
  softmax_params_region_t *softmax_params_sdram;

  uint softmax_key;
  uint min_softmax_key;
  uint softmax_layer_size;

  float softmax_denominator;
  uint received_softmax_counter = 0;
#else
  perceptron_params_region_t *perceptron_params_sdram;

  uint activation_function_id;
#endif


/* functions */

void generate_potential() {
  for (uint i = 0; i < n_potentials; i++) {
    potential += potentials[i] * weights[i];
  }
  potential += BIAS;
}

void instance_init() {
#ifdef softmax
  softmax_params_sdram =
    data_specification_get_region(INSTANCE_PARAMS, data_spec_meta);

  softmax_key = softmax_params_sdram->key;
  min_softmax_key = softmax_params_sdram->min_layer_key;
  softmax_layer_size = softmax_params_sdram->layer_size;
#else
  perceptron_params_sdram =
    data_specification_get_region(INSTANCE_PARAMS, data_spec_meta);

  activation_function_id =
    perceptron_params_sdram->activation_function_id;
#endif
}

void weights_init() {
  weights_sdram = data_specification_get_region(WEIGHTS, data_spec_meta);

  weights = (float *)malloc(sizeof(float) * n_weights);

  sark_mem_cpy((void *)weights, (void *)weights_sdram,
    sizeof(float) * n_weights);
}


/* additional softmax functions */
#ifdef softmax
void receive_softmax(float payload) {
  if (received_softmax_counter == 0) {
    softmax_denominator = .0;
  }
  softmax_denominator += payload;
  received_softmax_counter++;
}

bool softmax_pass_complete() {
  if (received_softmax_counter == softmax_layer_size) {
    received_softmax_counter = 0;
    return true;
  }
  return false;
}
#endif


/* function which has to be implemented by a machine vertex including
 * spiDNN.h */
void __init_base_params(
    uint32_t *timer_offset, uint *n_potentials, uint *min_pre_key)
{
  base_params_sdram = data_specification_get_region(BASE_PARAMS, data_spec_meta);

  forward_key = base_params_sdram->forward_key;
  n_weights = base_params_sdram->n_weights;

  *timer_offset = base_params_sdram->timer_offset;
  *n_potentials = n_weights - 1;
  *min_pre_key = base_params_sdram->min_pre_key;
}
