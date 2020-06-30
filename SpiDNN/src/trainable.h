#include "perceptron.h"

#define BATCH_COMPLETE backward_passes == batch_size


/* global variables */

trainable_params_region_t *trainable_params_sdram;
float *next_layer_weights_sdram;

uint batch_size;
uint backward_key;
uint min_next_key;
uint n_errors;
uint is_output_layer;

float learning_rate;

float *next_layer_weights;

float *gradients;
float *next_layer_gradients;

float error;
float neuron_error;

uint received_errors_counter;
uint backward_passes;


/* functions */

void receive_backward(uint key, float payload) {
  if (received_errors_counter == 0) {
    error = .0;
  }

  if (is_output_layer) {
    error += payload;
  } else {
    error += payload * next_layer_weights[key - min_next_key];
    next_layer_gradients[key - min_next_key] += payload * potential;
  }

  received_errors_counter++;
}

bool backward_pass_complete() {
  if (received_errors_counter == n_errors) {
    received_errors_counter = 0;
    return true;
  }
  return false;
}

void update_gradients() {
  // TODO: depending on activation function
  neuron_error = error * potential * (1 - potential);

  // when all errors are received -> compute gradients for each
  // weight -> sum in *gradients
  for (uint i=0; i < N_POTENTIALS; i++) {
    gradients[i] += neuron_error * potentials[i];
  }
  // special case: bias neuron has potential := 1
  gradients[N_POTENTIALS] += neuron_error;
}

void update_weights() {
  for (uint i=0; i < n_weights; i++) {
    weights[i] -= learning_rate * gradients[i];
  }

  if (!is_output_layer) {
    for (uint i=0; i < n_errors; i++) {
      next_layer_weights[i] -= learning_rate * next_layer_gradients[i];
    }
  }
}

void reset_batch() {
  backward_passes = 0;

  for (uint i=0; i < n_weights; i++) {
    gradients[i] = .0;
  }

  if (!is_output_layer) {
    for (uint i=0; i < n_errors; i++) {
      next_layer_gradients[i] = .0;
    }
  }
}

void trainable_init() {
  trainable_params_sdram =
    data_specification_get_region(TRAINABLE_PARAMS, data_spec_meta);

  batch_size = trainable_params_sdram->batch_size;
  backward_key = trainable_params_sdram->backward_key;
  min_next_key = trainable_params_sdram->min_next_key;
  n_errors = trainable_params_sdram->n_errors;
  is_output_layer = trainable_params_sdram->is_output_layer;
  learning_rate = trainable_params_sdram->learning_rate;

  gradients = (float *)malloc(sizeof(float) * n_weights);

  if (!is_output_layer) {
    next_layer_weights_sdram =
      data_specification_get_region(NEXT_LAYER_WEIGHTS, data_spec_meta);

    next_layer_weights = (float *)malloc(sizeof(float) * n_errors);

    sark_mem_cpy(
      (void *)next_layer_weights,
      (void *)next_layer_weights_sdram,
      sizeof(float) * n_errors
    );

    next_layer_gradients = (float *)malloc(sizeof(float) * n_errors);
  }
}
