#include "spiDNN.h"

//! human readable definitions of each region in SDRAM
typedef enum regions_e {
    __SYSTEM_REGION,
    PARAMS,
    KEYS,
} regions_e;

//! human readable definitions of the loss functions
typedef enum loss_functions_e {
  MEAN_SQUARED_ERROR,
  CATEGORICAL_CROSSENTROPY,
  BINARY_CROSSENTROPY,
} loss_functions_e;

//! definitions of each element in the params region
typedef struct params_region {
  uint32_t extractor_key;
  uint32_t loss_function_id;
  uint32_t K;
  uint32_t min_pre_key;
  uint32_t min_y_key;
  uint32_t timer_offset;
  uint32_t epoch_size;
} params_region_t;


/* global variables */

uint extractor_key;

uint min_y_key;

uint loss_function_id;

uint K;

uint epoch_size;

uint *keys;

float *y;

uint received_y_counter = 0;

float error;

float loss;
float overall_loss = .0;
float average_loss;
uint N = 0;

params_region_t *params_sdram;
uint *keys_sdram;


/* functions */

void receive_y(uint key, float payload) {
  uint idx = key - min_y_key;

  y[idx] = payload;
  received_y_counter++;
}

void receive(uint key, float payload) {
  //log_info("received potential from %d: %f", key, payload);

  // This works because min_y_key is guaranteed to be bigger than
  // min_pre_key, because the forward partition is touched before the
  // y partition by the toolchain.
  if (key >= min_y_key) {
    receive_y(key, payload);
  } else {
    receive_forward(key, payload);
  }
}

void compute_error(uint i) {
  switch (loss_function_id) {
    case MEAN_SQUARED_ERROR:
      error = potentials[i] - y[i];
      break;

    case CATEGORICAL_CROSSENTROPY:
      error = -y[i] / potentials[i];
      break;

    case BINARY_CROSSENTROPY:
      error = -y[i] / potentials[i] + (1 - y[i]) / (1 - potentials[i]);
      break;

    default:
      log_error("Unknown loss function %d - exiting!",
        loss_function_id);
      rt_error(RTE_SWERR);
  }
}

void compute_mse(void) {
    loss = .0;
    for (uint i=0; i < K; i++) {
      loss += (y[i] - potentials[i]) * (y[i] - potentials[i]);
    }
    loss = loss / (float) K;

    overall_loss += loss;
    average_loss = overall_loss / (float) N;
}

void compute_categorical_crossentropy(void) {
  loss = .0;
  for (uint i=0; i < K; i++) {
    loss -= y[i] * log(potentials[i]);
  }

  overall_loss += loss;
  average_loss = overall_loss / (float) N;
}

void compute_binary_crossentropy(void) {
  loss = -y[0] * log(potentials[0]) + (1 - y[0]) * log(1 - potentials[0]);

  overall_loss += loss;
  average_loss = overall_loss / (float) N;
}

void compute_loss(void) {
  switch (loss_function_id) {
    case MEAN_SQUARED_ERROR:
      compute_mse();
      break;

    case CATEGORICAL_CROSSENTROPY:
      compute_categorical_crossentropy();
      break;

    case BINARY_CROSSENTROPY:
      compute_binary_crossentropy();
      break;

    default:
      log_error("Unknown loss function %d - exiting!",
        loss_function_id);
      rt_error(RTE_SWERR);
  }
}

void update(uint ticks, uint b) {
  use(b);
  use(ticks);

  spiDNN_time++;

  if ((spiDNN_received_potentials_counter == K) && (received_y_counter == K))
  {
    N++;

    compute_loss();
    send(extractor_key, (void *)&average_loss);

    for (uint i=0; i < K; i++) {
      compute_error(i);
      send(keys[i], (void *)&error);
    }

    if (N == epoch_size) {
      N = 0;
      overall_loss = .0;
    }

    received_y_counter = 0;
    spiDNN_received_potentials_counter = 0;
  }
}

void keys_and_y_init(void) {
  keys_sdram = data_specification_get_region(KEYS, data_spec_meta);

  keys = (uint *)malloc(sizeof(uint) * K);
  sark_mem_cpy((void *)keys, (void *)keys_sdram,
    sizeof(uint) * K);

  y = (float *)malloc(sizeof(float) * K);
}

void c_main(void) {
  base_init();

  keys_and_y_init();

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
  params_sdram = data_specification_get_region(PARAMS, data_spec_meta);

  extractor_key = params_sdram->extractor_key;
  loss_function_id = params_sdram->loss_function_id;
  K = params_sdram->K;
  min_y_key = params_sdram->min_y_key;
  epoch_size = params_sdram->epoch_size;

  *timer_offset = params_sdram->timer_offset;
  *n_potentials = K;
  *min_pre_key = params_sdram->min_pre_key;
}
