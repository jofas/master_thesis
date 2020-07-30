#ifndef _SPIDNN_
#include "spiDNN.h"
#endif

#define BATCH_COMPLETE (batch_counter == batch_size) \
                       || (backward_passes_counter % epoch_size == 0)
#define FIT_COMPLETE (backward_passes_counter == epoch_size * epochs)
#define CONNECTION_OFFSET (n_next_layer_weights / n_next_layer_connections)

//! definitions of each element in the trainable_params region
typedef struct trainable_params_region {
  uint32_t backward_key;
  uint32_t min_next_key;
  uint32_t n_errors;
  uint32_t kernel_update_key; // Only used by Conv layers
  uint32_t min_layer_key;     // Only used by Conv layers
  uint32_t layer_size;        // Only used by Conv layers
  uint32_t n_next_layer_weights;
  uint32_t n_next_layer_connections;
  uint32_t id;
  uint32_t epochs;
  uint32_t epoch_size;
  uint32_t batch_size;
  float learning_rate;
} trainable_params_region_t;


/* global variables */

trainable_params_region_t *trainable_params_sdram;

uint backward_key;
uint min_next_key;
uint n_errors;

uint kernel_update_key;
uint min_layer_key;
uint layer_size;

uint n_next_layer_weights;
uint n_next_layer_connections;

uint epochs;
uint epoch_size;
uint batch_size;
float learning_rate;

float *neuron_gradients;
float *kernel_gradients;
float *next_layer_gradients;

uint *received_gradients_from_neuron_counter;

float *errors;

uint received_errors_counter = 0;
uint backward_passes_counter = 0;
uint received_gradients_counter = 0;
uint batch_counter = 0;

uint *received_errors;
uint id;


/* functions */

void reset_backward_receive(uint n_filters) {
  for (uint i = 0; i < n_filters; i++)
    errors[i] = .0;
}

void reset_gradient_receive(void) {
  for (uint i = 0; i < layer_size; i++)
    received_gradients_from_neuron_counter[i] = 0;
}

void reset_received_errors(void) {
  for (uint i = 0; i < n_next_layer_weights; i++)
    received_errors[i] = 0;
}

void receive_backward(
    uint key, float payload, uint n_filters, float *potentials)
{
  if (received_errors_counter == 0)
    reset_backward_receive(n_filters);

  uint idx = key - min_next_key;

  received_errors[idx]++;

  if (received_errors[idx] == (id + 1)) {
    for (uint i = 0; i < n_filters; i++)
      errors[i] += payload;
    received_errors_counter++;
  }
}

void receive_gradient(uint key, float payload) {
  if (received_gradients_counter == 0)
    reset_gradient_receive();

  uint idx = key - min_layer_key;

  kernel_gradients[received_gradients_from_neuron_counter[idx]] += payload;

  /*
  log_info("updating gradient: %d %d %f %f",
    key,
    received_gradients_from_neuron_counter[idx],
    payload,
    kernel_gradients[received_gradients_from_neuron_counter[idx]]
  );
  */

  received_gradients_from_neuron_counter[idx]++;
  received_gradients_counter++;
}

bool backward_pass_complete(void) {
  if (received_errors_counter == n_errors) {
    received_errors_counter = 0;
    reset_received_errors();
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

void update_neuron_gradients(
    uint activation_function_id, uint n_filters, uint n_kernel_elements,
    uint padding_offset, float *results)
{
  for (uint i = 0; i < n_filters; i++) {
    // as far as I know this works (at least for identity activation)
    errors[i] = apply_activation_function_derivative(
      activation_function_id, errors[i], results[i]);

    uint filter_offset = i * (n_kernel_elements + 1);

    for (uint j = 0; j < n_potentials; j++) {
      uint idx = j + filter_offset + padding_offset;
      // I only need to update gradients where I received a potential
      // (and bias), the other ones are zero anyways... so iter over
      // n_potentials (4) and index into gradients like in
      // generate_filter_results

      // potentials no bias me muy stupido
      // padding... goddamn
      // TODO: how to index potential ???
      // potentials : [      0, 1, 2, 3,          0,  1,  2,  3,   ]
      // gradients  : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
      neuron_gradients[idx] += errors[i] * potentials[j];
      //log_info("gradient at position: %f %f %d",
      //  neuron_gradients[idx], potentials[j], idx);
    }
    // special case: bias neuron has potential := 1
    neuron_gradients[n_kernel_elements + filter_offset] += errors[i];
  }
}

void update_neuron_weights(uint n_weights, float *weights) {
  for (uint i=0; i < n_weights; i++) {
    weights[i] -= learning_rate * neuron_gradients[i];
  }
}

void update_kernel_weights(uint n_weights, float *weights) {
  for (uint i=0; i < n_weights; i++) {
    weights[i] -= learning_rate * kernel_gradients[i];
  }
}

void reset_batch(uint n_weights) {
  batch_counter = 0;

  for (uint i=0; i < n_weights; i++) {
    neuron_gradients[i] = .0;
    kernel_gradients[i] = .0;
  }
}

void trainable_init(uint n_weights, uint n_filters) {
  trainable_params_sdram =
    data_specification_get_region(TRAINABLE_PARAMS, data_spec_meta);

  backward_key = trainable_params_sdram->backward_key;
  min_next_key = trainable_params_sdram->min_next_key;
  n_errors = trainable_params_sdram->n_errors;
  kernel_update_key = trainable_params_sdram->kernel_update_key;
  min_layer_key = trainable_params_sdram->min_layer_key;
  layer_size = trainable_params_sdram->layer_size;
  n_next_layer_weights = trainable_params_sdram->n_next_layer_weights;
  n_next_layer_connections = trainable_params_sdram->n_next_layer_connections;
  id = trainable_params_sdram->id;
  epochs = trainable_params_sdram->epochs;
  epoch_size = trainable_params_sdram->epoch_size;
  batch_size = trainable_params_sdram->batch_size;
  learning_rate = trainable_params_sdram->learning_rate;

  neuron_gradients = (float *)malloc(sizeof(float) * n_weights);
  kernel_gradients = (float *)malloc(sizeof(float) * n_weights);

  errors = (float *)malloc(sizeof(float) * n_filters);

  received_errors = (uint *)malloc(
    sizeof(uint) * n_next_layer_weights);
  reset_received_errors();

  // TODO: try removing if statement
  if (!layer_size == 0) // perceptron has layer_size of 0
    received_gradients_from_neuron_counter = (uint *)malloc(
      sizeof(uint) * layer_size);

  reset_batch(n_weights);
}
