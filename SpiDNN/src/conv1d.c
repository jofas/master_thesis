#include "spiDNN.h"

#define N_KERNEL_ELEMENTS kernel_size * n_channels
#define N_WEIGHTS (N_KERNEL_ELEMENTS + 1) * n_filters

#define PADDING_OFFSET lower_padding * n_channels
#define FILTER_OFFSET (N_KERNEL_ELEMENTS + 1) * filter

/* structs and enums */

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

uint *forward_keys;

uint kernel_size;
uint n_channels;
uint n_filters;

uint lower_padding;
uint upper_padding;

uint activation_function_id;

float *weights;

uint *channel_counters;
float *filter_results;

base_params_region_t *base_params_sdram;
uint32_t *forward_keys_sdram;
float *weights_sdram;


/* functions */

void generate_potential(uint filter) {
  for (uint i = 0; i < n_potentials; i++) {
    filter_results[filter] += potentials[i] * weights[
      FILTER_OFFSET + PADDING_OFFSET + i];
  }

  // bias
  filter_results[filter] += weights[FILTER_OFFSET + N_KERNEL_ELEMENTS];
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

void weights_init(void) {
  weights_sdram = data_specification_get_region(WEIGHTS, data_spec_meta);

  weights = (float *)malloc(sizeof(float) * N_WEIGHTS);

  sark_mem_cpy((void *)weights, (void *)weights_sdram,
    sizeof(float) * N_WEIGHTS);
}

void keys_init(void) {
  forward_keys_sdram = data_specification_get_region(KEYS, data_spec_meta);

  forward_keys = (uint32_t *)malloc(sizeof(uint32_t) * n_filters);

  sark_mem_cpy((void *)forward_keys, (void *)forward_keys_sdram,
    sizeof(uint32_t) * n_filters);
}

void reset_forward_pass(void) {
  for (uint i = 0; i < n_filters; i++) {
    filter_results[i] = .0;
  }
  for (uint i = 0; i < n_potentials; i++) {
    channel_counters[i] = 0;
  }
}

void receive(uint key, float payload) {
  if (spiDNN_received_potentials_counter == 0)
    reset_forward_pass();

  receive_forward_with_channel(
    key, payload, channel_counters, n_channels);
}

void update(uint ticks, uint b) {
  use(b);
  use(ticks);

  spiDNN_time++;

  if (forward_pass_complete()) {
    if (activation_function_id == SOFTMAX) {
      float softmax_denom = .0;
      for (uint i = 0; i < n_filters; i++) {
        activate(i);
        softmax_denom += filter_results[i];
      }
      for (uint i = 0; i < n_filters; i++) {
        filter_results[i] = filter_results[i] / softmax_denom;
        send(forward_keys[i], (void *)&filter_results[i]);
      }
    } else {
      for (uint i = 0; i < n_filters; i++) {
        activate(i);
        send(forward_keys[i], (void *)&filter_results[i]);
      }
    }
  }
}

void c_main(void) {
  base_init();

  weights_init();
  keys_init();

  channel_counters = (uint *)malloc(sizeof(uint) * n_potentials);
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
