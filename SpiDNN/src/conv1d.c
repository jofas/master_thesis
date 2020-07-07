#include "spiDNN.h"

#define N_WEIGHTS (kernel_size + 1) * n_filters


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
  uint32_t activation_function_id;
} base_params_region_t;


/* global variables */

uint forward_key;

uint kernel_size;
uint n_channels;
uint n_filters;

uint activation_function_id;

float *weights;

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

void activate() {
  generate_potential();

  switch (activation_function_id) {
    case IDENTITY:
      break;

    case RELU:
      potential = potential > .0 ? potential : .0;
      break;

    case SIGMOID:
      potential = 1. / (1. + exp(-potential));
      break;

    case TANH:
      potential = tanh(potential);
      break;

    case SOFTMAX:
      potential = exp(potential);
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

void receive(uint key, float payload) {
  //log_info("received potential from %d: %f", key, payload);

  if (spiDNN_received_potentials_counter == 0) {
    potential = .0;
  }
  receive_forward(key, payload);
}

void update(uint ticks, uint b) {
  use(b);
  use(ticks);

  spiDNN_time++;

#ifdef softmax
  if (softmax_pass_complete()) {
    potential = potential / softmax_denominator;
    send(forward_key, (void *)&potential);
    //log_error("sending shit: %f", potential);
    //rt_error(RTE_SWERR);
  }
#endif

  if (forward_pass_complete()) {
    activate();
#ifdef softmax
    send(softmax_key, (void *)&potential);
#else
    send(forward_key, (void *)&potential);
#endif
  }

#ifdef trainable
  if (backward_pass_complete()) {
    backward_passes_counter++;
    batch_counter++;

    update_gradients();

    if (BATCH_COMPLETE) {
      update_weights();
      if (FIT_COMPLETE) {
        sark_mem_cpy((void *)weights_sdram, (void *)weights,
          sizeof(float) * n_weights);
      }
      reset_batch();
    }

    send(backward_key, (void *)&neuron_error);
  }
#endif
}

void c_main(void) {
  base_init();

  weights_init();

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
  base_params_sdram = data_specification_get_region(BASE_PARAMS, data_spec_meta);

  forward_key = base_params_sdram->forward_key;
  kernel_size = base_params_sdram->kernel_size;
  n_channels = base_params_sdram->n_channels;
  n_filters = base_params_sdram->n_filters;
  activation_function_id = base_params_sdram->activation_function_id;

  *timer_offset = base_params_sdram->timer_offset;
  *n_potentials = kernel_size * n_channels;
  *min_pre_key = base_params_sdram->min_pre_key;
}
