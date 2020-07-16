// GODDAMN! NEED SOMETHING SIMILAR FOR CONV (just pass arguments)
#include "perceptron.h"


#define BATCH_COMPLETE batch_counter == batch_size \
                       || (backward_passes_counter % epoch_size == 0)
#define FIT_COMPLETE backward_passes_counter == epoch_size * epochs

//! definitions of each element in the trainable_params region
typedef struct trainable_params_region {
  uint32_t backward_key;
  uint32_t min_next_key;
  uint32_t n_errors;
  uint32_t is_output_layer;
  uint32_t kernel_update_key; // Only used by Conv layers
  uint32_t n_next_layer_weights;
  uint32_t epochs;
  uint32_t epoch_size;
  uint32_t batch_size;
  float learning_rate;
} trainable_params_region_t;


/* global variables */

trainable_params_region_t *trainable_params_sdram;
float *next_layer_weights_sdram;

uint backward_key;
uint min_next_key;
uint n_errors;
uint is_output_layer;
uint kernel_update_key;
uint n_next_layer_weights;

uint epochs;
uint epoch_size;
uint batch_size;
float learning_rate;

float *next_layer_weights;

float *gradients;
float *next_layer_gradients;

float error;

uint received_errors_counter = 0;
uint backward_passes_counter = 0;
uint batch_counter;

/* functions */

void receive_backward(uint key, float payload) {
  if (received_errors_counter == 0) {
    error = .0;
  }

  // error to array the size of n_filters
  //
  // next_layer_is_dense
  // if (n_errors < n_next_layer_weights)
  //   for (uint j = 0; j < n_next_layer_weights / n_errors; j++) {
  //     errors[i] += payload * next_layer_weights[
  //       key - min_next_key + j];
  //   }
  //   do some
  // else
  //   update error for each filter
  //    errors[i] += payload * next_layer_weights[
  //      key - min_next_key + receive_counter[key - min_next_key]];
  //    receive_counter[key - min_next_key]++;
  //
  //   do some else
  //

  if (is_output_layer) {
    error += payload;
  } else {
    error += payload * next_layer_weights[key - min_next_key];
    next_layer_gradients[key - min_next_key] += payload * potential;
  }

  received_errors_counter++;
}

bool backward_pass_complete(void) {
  if (received_errors_counter == n_errors) {
    received_errors_counter = 0;
    return true;
  }
  return false;
}

void apply_activation_function_derivative(void) {
  switch (activation_function_id) {
    case IDENTITY:
      break;

    case RELU:
      error = potential > .0 ? error : .0;
      break;

    case SIGMOID:
      error = error * potential * (1 - potential);
      break;

    case TANH:
      error = error * (1 - potential * potential);
      break;

    case SOFTMAX:
      error = error * potential * (1 - potential);
      break;

    default:
      log_error("Unknown activation function %d - exiting!",
        activation_function_id);
      rt_error(RTE_SWERR);
  }
}

void update_gradients(void) {
  apply_activation_function_derivative();

  for (uint i=0; i < n_potentials; i++) {
    gradients[i] += error * potentials[i];
  }
  // special case: bias neuron has potential := 1
  gradients[n_potentials] += error;
}

void update_weights(void) {
  for (uint i=0; i < n_weights; i++) {
    weights[i] -= learning_rate * gradients[i];
  }

  if (!is_output_layer) {
    for (uint i=0; i < n_errors; i++) {
      next_layer_weights[i] -= learning_rate * next_layer_gradients[i];
    }
  }
}

void reset_batch(void) {
  batch_counter = 0;

  for (uint i=0; i < n_weights; i++) {
    gradients[i] = .0;
  }

  if (!is_output_layer) {
    for (uint i=0; i < n_errors; i++) {
      next_layer_gradients[i] = .0;
    }
  }
}

void trainable_init(void) {
  trainable_params_sdram =
    data_specification_get_region(TRAINABLE_PARAMS, data_spec_meta);

  backward_key = trainable_params_sdram->backward_key;
  min_next_key = trainable_params_sdram->min_next_key;
  n_errors = trainable_params_sdram->n_errors;
  is_output_layer = trainable_params_sdram->is_output_layer;
  kernel_update_key = trainable_params_sdram->kernel_update_key;
  n_next_layer_weights = trainable_params_sdram->n_next_layer_weights;
  epochs = trainable_params_sdram->epochs;
  epoch_size = trainable_params_sdram->epoch_size;
  batch_size = trainable_params_sdram->batch_size;
  learning_rate = trainable_params_sdram->learning_rate;

  gradients = (float *)malloc(sizeof(float) * n_weights);

  if (!is_output_layer) {
    next_layer_weights_sdram =
      data_specification_get_region(NEXT_LAYER_WEIGHTS, data_spec_meta);

    next_layer_weights = (float *)malloc(sizeof(float) * n_next_layer_weights);

    sark_mem_cpy(
      (void *)next_layer_weights,
      (void *)next_layer_weights_sdram,
      sizeof(float) * n_next_layer_weights
    );

    next_layer_gradients = (float *)malloc(sizeof(float) * n_next_layer_weights);
  }

  reset_batch();
}
