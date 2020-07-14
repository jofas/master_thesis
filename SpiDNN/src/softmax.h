#include "perceptron.h"


//! definitions of each element in the softmax_params region
typedef struct softmax_params_region {
  uint32_t key;
  uint32_t min_layer_key;
  uint32_t layer_size;
  uint32_t n_filters;
} softmax_params_region_t;


/* global variables */

softmax_params_region_t *softmax_params_sdram;

uint softmax_key;
uint softmax_min_layer_key;
uint softmax_layer_size;
uint softmax_n_filters;

float *softmax_denominators;
uint *softmax_filter_counters;
uint received_softmax_counter = 0;


/* functions */

void softmax_reset(void) {
  for (uint i = 0; i < softmax_n_filters; i++) {
    softmax_denominators[i] = .0;
  }

  for (uint i = 0; i < softmax_layer_size; i++) {
    softmax_filter_counters[i] = 0;
  }
}

void receive_softmax(uint key, float payload) {
  if (received_softmax_counter == 0)
    softmax_reset();

  uint idx = key - softmax_min_layer_key;

  softmax_denominators[softmax_filter_counters[idx]] += payload;

  softmax_filter_counters[idx]++;
  received_softmax_counter++;
}

bool softmax_pass_complete(void) {
  if (received_softmax_counter == softmax_layer_size * softmax_n_filters) {
    received_softmax_counter = 0;
    return true;
  }
  return false;
}

void softmax_init(void) {
  softmax_params_sdram =
    data_specification_get_region(SOFTMAX_PARAMS, data_spec_meta);

  softmax_key = softmax_params_sdram->key;
  softmax_min_layer_key = softmax_params_sdram->min_layer_key;
  softmax_layer_size = softmax_params_sdram->layer_size;
  softmax_n_filters = softmax_params_sdram->n_filters;

  softmax_denominators = (float *)malloc(sizeof(float) * softmax_n_filters);
  softmax_filter_counters = (uint *)malloc(sizeof(uint) * softmax_layer_size);
}
