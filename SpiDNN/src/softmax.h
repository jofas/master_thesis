#ifndef _SPIDNN_
#include "spiDNN.h"
#endif


//! definitions of each element in the softmax_params region
typedef struct softmax_params_region {
  uint32_t key;
  uint32_t layer_size;
} softmax_params_region_t;


/* global variables */

softmax_params_region_t *softmax_params_sdram;

uint softmax_key;
uint softmax_layer_size;

float softmax_denominator;
uint received_softmax_counter = 0;


/* functions */

void receive_softmax(float payload) {
  if (received_softmax_counter == 0)
    softmax_denominator = .0;

  softmax_denominator += payload;

  received_softmax_counter++;
}

bool softmax_pass_complete(void) {
  if (received_softmax_counter == softmax_layer_size) {
    received_softmax_counter = 0;
    return true;
  }
  return false;
}

void softmax_init(void) {
  softmax_params_sdram =
    data_specification_get_region(SOFTMAX_PARAMS, data_spec_meta);

  softmax_key = softmax_params_sdram->key;
  softmax_layer_size = softmax_params_sdram->layer_size;
}
