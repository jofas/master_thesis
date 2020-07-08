#include "spiDNN.h"

#define N_WEIGHTS (kernel_size + 1) * n_filters
#define N_KERNEL_ELEMENTS kernel_size * n_channels

/* structs and enums */

//! human readable definitions of each region in SDRAM
typedef enum regions_e {
    __SYSTEM_REGION,
    BASE_PARAMS,
    WEIGHTS,
    SOFTMAX_PARAMS,
    TRAINABLE_PARAMS,
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
  uint32_t kernel_size;
  uint32_t n_channels;
  uint32_t n_filters;
  uint32_t lower_padding;
  uint32_t upper_padding;
  uint32_t activation_function_id;
} base_params_region_t;


/* global variables */

uint forward_key;

uint kernel_size;
uint n_channels;
uint n_filters;

uint lower_padding;
uint upper_padding;

uint activation_function_id;

float *weights;

float *filter_results;

float *weights_sdram;
base_params_region_t *base_params_sdram;


/* functions */

void generate_potential(uint filter) {
  // (N_KERNEL_ELEMENTS + 1) * filter is definetly wrong.
  // should be (N_KERNEL_ELEMENTS + 1) * filter + 1 (I think ...
  // future Jonas will handle that once we are at multiple filters)
  for (uint i = 0; i < n_potentials; i++) {
    filter_results[filter] += potentials[i]
      * weights[(N_KERNEL_ELEMENTS + 1) * filter + i + lower_padding];
  }

  filter_results[filter] += weights[
    (N_KERNEL_ELEMENTS + 1) * filter + N_KERNEL_ELEMENTS];
}

void activate(uint filter) {
  generate_potential(filter);

  switch (activation_function_id) {
    case IDENTITY:
      break;

    case RELU:
      filter_results[filter] = filter_results[filter] > .0 ?
        filter_results[filter] : .0;
      break;

    case SIGMOID:
      filter_results[filter] = 1. / (1. + exp(-filter_results[filter]));
      break;

    case TANH:
      filter_results[filter] = tanh(filter_results[filter]);
      break;

    case SOFTMAX:
      filter_results[filter] = exp(filter_results[filter]);
      break;

    default:
      log_error("Unknown activation function %d - exiting!",
        activation_function_id);
      rt_error(RTE_SWERR);
  }
}

void weights_init() {
  weights_sdram = data_specification_get_region(WEIGHTS, data_spec_meta);

  weights = (float *)malloc(sizeof(float) * N_WEIGHTS);

  sark_mem_cpy((void *)weights, (void *)weights_sdram,
    sizeof(float) * N_WEIGHTS);
}

void reset_filter_results() {
  for (uint i = 0; i < n_filters; i++) {
    filter_results[i] = .0;
  }
}

void receive(uint key, float payload) {
  if (spiDNN_received_potentials_counter == 0)
    reset_filter_results();

  receive_forward_with_channel(key, payload, kernel_size);
}

void update(uint ticks, uint b) {
  use(b);
  use(ticks);

  spiDNN_time++;

  if (forward_pass_complete()) {
    for (uint i = 0; i < n_filters; i++) {
      activate(i);
      send(forward_key, (void *)&filter_results[i]);
    }
  }
}

void c_main(void) {
  base_init();

  weights_init();

  filter_results = (float *)malloc(sizeof(float) * n_filters);

  // register callbacks
  spin1_callback_on(MCPL_PACKET_RECEIVED, receive, MC_PACKET);
  spin1_callback_on(TIMER_TICK, update, TIMER);

  log_info("\nStarting simulation\n");
  simulation_run();
}


/* function which has to be implemented by a machine vertex including
 * spiDNN.h */
void __init_base_params(
    uint32_t *timer_offset, uint *n_potentials, uint *min_pre_key)
{
  base_params_sdram =
    data_specification_get_region(BASE_PARAMS, data_spec_meta);

  forward_key = base_params_sdram->forward_key;
  kernel_size = base_params_sdram->kernel_size;
  n_channels = base_params_sdram->n_channels;
  n_filters = base_params_sdram->n_filters;
  lower_padding = base_params_sdram->lower_padding;
  upper_padding = base_params_sdram->upper_padding;
  activation_function_id = base_params_sdram->activation_function_id;

  *timer_offset = base_params_sdram->timer_offset;
  *n_potentials =
    (kernel_size - lower_padding - upper_padding) * n_channels;
  *min_pre_key = base_params_sdram->min_pre_key;
}
