#ifndef _SPIDNN_
#include "spiDNN.h"
#endif

#define BATCH_COMPLETE (batch_counter == batch_size) \
                       || (backward_passes_counter % epoch_size == 0)
#define FIT_COMPLETE (backward_passes_counter == epoch_size * epochs)
#define NEXT_LAYER_IS_DENSE (n_errors < n_next_layer_weights)


//! definitions of each element in the trainable_params region
typedef struct trainable_params_region {
  uint32_t backward_key;
  uint32_t min_next_key;
  uint32_t n_errors;
  uint32_t is_output_layer;
  uint32_t kernel_update_key; // Only used by Conv layers
  uint32_t min_layer_key;     // Only used by Conv layers
  uint32_t layer_size;        // Only used by Conv layers
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
uint min_layer_key;
uint layer_size;

uint n_next_layer_weights;

uint epochs;
uint epoch_size;
uint batch_size;
float learning_rate;

float *next_layer_weights;
uint *next_layer_receive_counters;

float *gradients;
float *next_layer_gradients;

uint *received_gradients_from_neuron_counter;

float *errors;

uint received_errors_counter = 0;
uint backward_passes_counter = 0;
uint received_gradients_counter = 0;
uint batch_counter = 0;

/* functions */

void reset_backward_receive(uint n_filters) {
  for (uint i = 0; i < n_filters; i++)
    errors[i] = .0;

  for (uint i = 0; i < n_next_layer_weights; i++)
    next_layer_receive_counters[i] = 0;
}

void reset_gradient_receive(void) {
  for (uint i = 0; i < layer_size; i++)
    received_gradients_from_neuron_counter[i] = 0;
}

void receive_backward(
    uint key, float payload, uint n_filters, float *potentials)
{
  if (received_errors_counter == 0)
    reset_backward_receive(n_filters);

  uint idx = key - min_next_key;

  // Only interesting when called from a Conv layer. If next layer is
  // dense this layer was flattened, which means the neuron I receive
  // from only sends one error but is associated with every filter of
  // this layer.
  if (NEXT_LAYER_IS_DENSE) {
    // TODO: once I support Conv layers as output layer I need to
    //       catch whether this layer is the output layer
    for (uint i = 0; i < n_filters; i++) {
      errors[i] += payload * next_layer_weights[idx + i];
    }
  } else {

    if (is_output_layer) {
      // TODO: how will this look with Conv as output layer?
      for (uint i = 0; i < n_filters; i++) {
        errors[i] += payload;
      }

    } else {
      for (uint i = 0; i < n_filters; i++) {
        errors[i] += payload * next_layer_weights[
          idx + next_layer_receive_counters[idx]];

        next_layer_gradients[idx + next_layer_receive_counters[idx]] +=
          payload * potentials[i];
      }
    }
  }

  // TODO: next_layer_receive_counters must only be as big as
  //       len(edges), not len(edges) * XXX.n_filters
  next_layer_receive_counters[idx]++;
  received_errors_counter++;
}

void receive_gradient(uint key, float payload) {
  if (received_errors_counter == 0)
    reset_gradient_receive();

  received_gradients_counter++;

  // Don't update gradients with own gradients
  if (key == kernel_update_key)
    return;

  uint idx = key - min_layer_key;

  gradients[received_gradients_from_neuron_counter[idx]] += payload;

  received_gradients_from_neuron_counter[idx]++;
}

bool backward_pass_complete(void) {
  if (received_errors_counter == n_errors) {
    received_errors_counter = 0;
    return true;
  }
  return false;
}

bool gradient_pass_complete(uint n_weights) {
  if (received_gradients_counter == n_weights * layer_size) {
    received_gradients_counter = 0;
    return true;
  }
  return false;
}

float apply_activation_function_derivative(uint activation_function_id, float error, float potential) {
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

void update_gradients(
    uint activation_function_id, uint n_filters, uint kernel_size,
    float *results)
{
  for (uint i = 0; i < n_filters; i++) {
    errors[i] = apply_activation_function_derivative(
      activation_function_id, errors[i], results[i]);

    for (uint j = 0; j < kernel_size; j++) {
      float potential = potentials[j + i * (kernel_size + 1)];
      gradients[j + i * (kernel_size + 1)] += errors[i] * potential;
    }
    // special case: bias neuron has potential := 1
    gradients[kernel_size + i * (kernel_size + 1)] += errors[i];
  }
}

void update_weights(uint n_weights, float *weights) {
  for (uint i=0; i < n_weights; i++) {
    weights[i] -= learning_rate * gradients[i];
  }

  if (!is_output_layer) {
    for (uint i=0; i < n_next_layer_weights; i++) {
      next_layer_weights[i] -= learning_rate * next_layer_gradients[i];
    }
  }
}

void reset_batch(uint n_weights) {
  batch_counter = 0;

  for (uint i=0; i < n_weights; i++) {
    gradients[i] = .0;
  }

  if (!is_output_layer) {
    for (uint i=0; i < n_next_layer_weights; i++) {
      next_layer_gradients[i] = .0;
    }
  }
}

void next_layer_weights_init(void) {
  next_layer_weights_sdram =
    data_specification_get_region(NEXT_LAYER_WEIGHTS, data_spec_meta);

  next_layer_weights = (float *)malloc(sizeof(float) * n_next_layer_weights);

  sark_mem_cpy(
    (void *)next_layer_weights,
    (void *)next_layer_weights_sdram,
    sizeof(float) * n_next_layer_weights
  );

  next_layer_gradients = (float *)malloc(sizeof(float) * n_next_layer_weights);
  next_layer_receive_counters = (uint *)malloc(sizeof(uint) * n_next_layer_weights);
}

void trainable_init(uint n_weights, uint n_filters) {
  trainable_params_sdram =
    data_specification_get_region(TRAINABLE_PARAMS, data_spec_meta);

  backward_key = trainable_params_sdram->backward_key;
  min_next_key = trainable_params_sdram->min_next_key;
  n_errors = trainable_params_sdram->n_errors;
  is_output_layer = trainable_params_sdram->is_output_layer;
  kernel_update_key = trainable_params_sdram->kernel_update_key;
  min_layer_key = trainable_params_sdram->min_layer_key;
  layer_size = trainable_params_sdram->layer_size;
  n_next_layer_weights = trainable_params_sdram->n_next_layer_weights;
  epochs = trainable_params_sdram->epochs;
  epoch_size = trainable_params_sdram->epoch_size;
  batch_size = trainable_params_sdram->batch_size;
  learning_rate = trainable_params_sdram->learning_rate;

  gradients = (float *)malloc(sizeof(float) * n_weights);

  errors = (float *)malloc(sizeof(float) * n_filters);

  if (!is_output_layer)
    next_layer_weights_init();

  // TODO: try removing if statement
  if (!layer_size == 0) // perceptron has layer_size of 0
    received_gradients_from_neuron_counter = (uint *)malloc(
      sizeof(uint) * layer_size);

  reset_batch(n_weights);
}
