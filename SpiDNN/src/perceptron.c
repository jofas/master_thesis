#include "spiDNN.h"

#ifdef softmax
#include "softmax.h"
#endif

#define BIAS weights[n_weights - 1]
#define N_POTENTIALS n_weights - 1


#ifdef trainable

#define BATCH_COMPLETE (batch_counter == batch_size) \
                       || (backward_passes_counter % epoch_size == 0)
#define FIT_COMPLETE (backward_passes_counter == epoch_size * epochs)

#endif


//! definitions of each element in the base_params region
typedef struct base_params_region {
  uint32_t forward_key;
  uint32_t min_pre_key;
  uint32_t timer_offset;
  uint32_t n_weights;
  uint32_t activation_function_id;
} base_params_region_t;

#ifdef trainable

  //! definitions of each element in the trainable_params region
  typedef struct trainable_params_region {
    uint32_t backward_key;
    uint32_t min_next_key;
    uint32_t n_errors;
    uint32_t id;
    uint32_t epochs;
    uint32_t epoch_size;
    uint32_t batch_size;
    float learning_rate;
  } trainable_params_region_t;

#endif


/* global variables */

uint forward_key;

uint n_weights;

uint activation_function_id;

float *weights;

float potential;

float *weights_sdram;
base_params_region_t *base_params_sdram;

#ifdef trainable

  trainable_params_region_t *trainable_params_sdram;

  uint backward_key;
  uint min_next_key;

  uint n_errors;

  uint epochs;
  uint epoch_size;
  uint batch_size;
  float learning_rate;

  float *neuron_gradients;

  uint received_errors_counter = 0;
  uint backward_passes_counter = 0;
  uint batch_counter = 0;

  uint *received_errors;
  uint id;

  float error;

#endif


/* functions */

#ifdef trainable

  void reset_received_errors(void) {
    for (uint i = 0; i < n_errors; i++)
      received_errors[i] = 0;
  }

  void receive_backward(uint key, float payload) {
    if (received_errors_counter == 0)
      error = .0;

    uint idx = key - min_next_key;

    //received_errors[idx]++;

    if (received_errors[idx]++ == id) {
      error += payload;
      received_errors_counter++;
    }
  }

  bool backward_pass_complete(void) {
    if (received_errors_counter == n_errors) {
      received_errors_counter = 0;
      reset_received_errors();
      return true;
    }
    return false;
  }

  float apply_activation_function_derivative(void) {
    switch (activation_function_id) {
      case IDENTITY:
        return error;

      case RELU:
        return potential > .0 ? error : .0;

      case SIGMOID:
        return error * potential * (1 - potential);

      case TANH:
        return error * (1 - potential * potential);

      case SOFTMAX:
        return error * potential * (1 - potential);

      default:
        log_error("Unknown activation function %d - exiting!",
          activation_function_id);
        rt_error(RTE_SWERR);
    }
  }

  void update_neuron_gradients(void) {
    error = apply_activation_function_derivative();

    for (uint idx = 0; idx < n_potentials; idx++)
      neuron_gradients[idx] += error * potentials[idx];

    // special case: bias neuron has potential := 1
    neuron_gradients[n_potentials] += error;
  }

  void update_neuron_weights(void) {
    for (uint i=0; i < n_weights; i++)
      weights[i] -= learning_rate * neuron_gradients[i];
  }

  void reset_batch(void) {
    batch_counter = 0;

    for (uint i=0; i < n_weights; i++)
      neuron_gradients[i] = .0;
  }

  void trainable_init(void) {
    trainable_params_sdram =
      data_specification_get_region(TRAINABLE_PARAMS, data_spec_meta);

    backward_key = trainable_params_sdram->backward_key;
    min_next_key = trainable_params_sdram->min_next_key;
    n_errors = trainable_params_sdram->n_errors;
    id = trainable_params_sdram->id;
    epochs = trainable_params_sdram->epochs;
    epoch_size = trainable_params_sdram->epoch_size;
    batch_size = trainable_params_sdram->batch_size;
    learning_rate = trainable_params_sdram->learning_rate;

    neuron_gradients = (float *)malloc(sizeof(float) * n_weights);

    received_errors = (uint *)malloc(sizeof(uint) * n_errors);
    reset_received_errors();

    reset_batch();
  }

#endif

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
    receive_backward(key, payload);
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

    update_neuron_gradients();

    for (uint i; i < N_POTENTIALS; i++) {
      float error_ = error * weights[i];
      send(backward_key, (void *)&error_);
      spin1_delay_us(12);
    }

    if (BATCH_COMPLETE) {
      update_neuron_weights();
      if (FIT_COMPLETE) {
        sark_mem_cpy((void *)weights_sdram, (void *)weights,
          sizeof(float) * n_weights);
      }
      reset_batch();
    }
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
  trainable_init();
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
