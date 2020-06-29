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
} loss_functions_e;

//! definitions of each element in the params region
typedef struct params_region {
    uint32_t loss_function_id;
    uint32_t K;
    uint32_t min_pre_key;
    uint32_t min_y_key;
    uint32_t timer_offset;
} params_region_t;

/* global variables */

uint min_y_key;

uint loss_function_id;

uint K;

uint *keys;

float *y;

uint received_y_counter;

params_region_t *params_sdram;
uint *keys_sdram;


/* functions */

void receive_y(uint key, float payload) {
  uint idx = key - min_y_key;

  y[idx] = payload;
  received_y_counter++;
}

void reset() {
  for (uint i=0; i < K; i++) {
    received_potentials[i] = false;
  }

  received_potentials_counter = 0;
  received_y_counter = 0;
}

void receive(uint key, float payload) {
  log_info("received potential from %d: %f", key, payload);

  // This works because min_y_key is guaranteed to be bigger than
  // min_pre_key, because the forward partition is touched before the
  // y partition by the toolchain.
  if (key >= min_y_key) {
    receive_y(key, payload);
  } else {
    receive_potential_from_pre_layer(key, payload);
  }
}

float compute_loss(uint i) {
  switch (loss_function_id) {
    case MEAN_SQUARED_ERROR:
      return potentials[i] - y[i];

    default:
      log_error("Unknown loss function %d - exiting!",
        loss_function_id);
      rt_error(RTE_SWERR);
  }
}

void update(uint ticks, uint b) {
  use(b);
  use(ticks);

  time++;

  if ((received_potentials_counter == K) &&
      (received_y_counter == K))
  {
    float loss_ = .0;
    for (uint i=0; i < K; i++) {
      float diff = y[i] - potentials[i];
      loss_ += diff * diff;
    }
    loss_ = loss_ / (float) K;
    log_info("loss: %f", loss_);

    float loss_i;
    for (uint i=0; i < K; i++) {
      loss_i = compute_loss(i);
      send(keys[i], (void *)&loss_i);
    }
    reset();
  }
}

void keys_and_y_init() {
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

  reset();

  log_info("\nStarting simulation\n");
  simulation_run();
}

/* function which has to be implemented by a machine vertex including
 * spiDNN.h */
void __init_base_params(
    uint32_t *timer_offset, uint32_t *n_potentials, uint32_t *min_pre_key)
{
  params_sdram = data_specification_get_region(PARAMS, data_spec_meta);

  loss_function_id = params_sdram->loss_function_id;
  K = params_sdram->K;
  min_y_key = params_sdram->min_y_key;

  *timer_offset = params_sdram->timer_offset;
  *n_potentials = K;
  *min_pre_key = params_sdram->min_pre_key;
}
