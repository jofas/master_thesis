#include "spiDNN.h"

#ifdef trainable
#include "trainable.h"
#endif

#ifdef softmax
#include "softmax.h"
#endif

#define BIAS weights[n_weights - 1]
#define N_POTENTIALS n_weights - 1


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

float potential;

float *weights_sdram;
base_params_region_t *base_params_sdram;


/* functions */

void generate_potential(void) {
  for (uint i = 0; i < n_potentials; i++) {
    potential += potentials[i] * weights[i];
  }
  potential += BIAS;
}

void activate(void) {
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

void weights_init(void) {
  weights_sdram = data_specification_get_region(WEIGHTS, data_spec_meta);

  weights = (float *)malloc(sizeof(float) * n_weights);

  sark_mem_cpy((void *)weights, (void *)weights_sdram,
    sizeof(float) * n_weights);
}

void receive(uint key, float payload) {
  //log_info("received potential from %d: %f", key, payload);

#ifdef softmax
  // min_pre_key will always be bigger than min_softmax_key, because
  // softmax partitions are touched by the toolchain before forward
  // and backward partitions
  if (key < min_pre_key) {
    receive_softmax(payload);
    return;
  }
#endif

#ifdef trainable
  // min_next_key will always be bigger than min_pre_key, because
  // the forward partition is touched by the toolchain before the
  // backward partition
  if (key >= min_next_key) {
    receive_backward(key, payload, 1, &potential);
    return;
  }
#endif

  if (spiDNN_received_potentials_counter == 0)
    potential = .0;

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

    update_neuron_gradients(activation_function_id, 1, n_potentials,
      0, &potential);

    if (BATCH_COMPLETE) {
      update_neuron_weights(n_weights, weights);
      if (FIT_COMPLETE) {
        sark_mem_cpy((void *)weights_sdram, (void *)weights,
          sizeof(float) * n_weights);
      }
      reset_batch(n_weights);
    }

    send(backward_key, (void *)errors);
  }
#endif
}

void c_main(void) {
  base_init();

  weights_init();

#ifdef softmax
  softmax_init();
#endif

#ifdef trainable
  trainable_init(n_weights, 1);
#endif

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
  n_weights = base_params_sdram->n_weights;
  activation_function_id = base_params_sdram->activation_function_id;

  *timer_offset = base_params_sdram->timer_offset;
  *n_potentials = n_weights - 1;
  *min_pre_key = base_params_sdram->min_pre_key;
}
