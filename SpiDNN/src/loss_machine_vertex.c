#include "spin1_api.h"
#include "common-typedefs.h"
#include <data_specification.h>
#include <simulation.h>
#include <sark.h>
#include <debug.h>

//! human readable definitions of each region in SDRAM
typedef enum regions_e { // {{{
    SYSTEM_REGION,
    PARAMS,
    KEYS,
} regions_e; // }}}

//! human readable definitions of the loss functions
typedef enum loss_functions_e { // {{{
  MEAN_SQUARED_ERROR = 0,
} loss_functions_e; // }}}

//! definitions of each element in the params region
typedef struct params_region { // {{{
    uint32_t loss_function_id;
    uint32_t K;
    uint32_t min_pre_key;
    uint32_t min_y_key;
    uint32_t timer_offset;
} params_region_t; // }}}

//! values for the priority for each callback
typedef enum callback_priorities { // {{{
    MC_PACKET = -1,
    SDP = 1,
    TIMER = 2,
    DMA = 3
} callback_priorities; // }}}


/* global variables */

uint min_pre_key;
uint min_y_key;

uint loss_function_id;

uint K;

uint *keys;

float *potentials;
bool *received_potentials;

float *y;
bool *received_y;

uint received_potentials_counter;
uint received_y_counter;

params_region_t *params_sdram;
uint *keys_sdram;

static uint32_t time;
data_specification_metadata_t *data = NULL;

// value for turning on and off interrupts
uint cpsr = 0;


/* functions */

void receive_potential_from_pre_layer(uint key, float payload) { // {{{
  uint idx = key - min_pre_key;

  if (received_potentials[idx]) {
    log_error("received potential too fast. Last input wasn't
               properly processed yet - exiting!");
    rt_error(RTE_SWERR);
  } else {
    potentials[idx] = payload;
    received_potentials[idx] = true;
    received_potentials_counter++;
  }
} // }}}

void receive_y(uint key, float payload) { // {{{
  uint idx = key - min_y_key;

  if (received_y[idx]) {
    log_error("received y too fast. Last input wasn't
               properly processed yet - exiting!");
    rt_error(RTE_SWERR);
  } else {
    y[idx] = payload;
    received_y[idx] = true;
    received_y_counter++;
  }
} // }}}

void reset() { // {{{
  for (uint i=0; i < K; i++) {
    received_potentials[i] = false;
    received_y[i] = false;
  }

  received_potentials_counter = 0;
  received_y_counter = 0;
} // }}}

void send(uint key, float payload) { // {{{
  uint send_bytes;
  sark_mem_cpy((void *)&send_bytes, &payload, sizeof(uint));

  log_info("sending value: %f with key: %d", payload, key);

  while (!spin1_send_mc_packet(key, send_bytes, WITH_PAYLOAD)) {
    spin1_delay_us(1);
  }
} // }}}

void __init_dtcm() { // {{{
  keys_sdram = data_specification_get_region(KEYS, data);

  keys = (uint *)malloc(sizeof(uint) * K);
  sark_mem_cpy((void *)keys, (void *)keys_sdram,
    sizeof(uint) * K);

  potentials = (float *)malloc(sizeof(float) * K);
  received_potentials = (bool *)malloc(sizeof(bool) * K);

  y = (float *)malloc(sizeof(float) * K);
  received_y = (bool *)malloc(sizeof(bool) * K);
} // }}}

static bool __init_simulation_and_data_spec(uint32_t *timer_period) { // {{{
  // Get the address this core's DTCM data starts at from SRAM
  data = data_specification_get_data_address();

  // Read the header
  if (!data_specification_read_header(data)) {
    log_error("failed to read the data spec header");
    return false;
  }

  // Get the timing details and set up the simulation interface
  if (!simulation_initialise(
        data_specification_get_region(SYSTEM_REGION, data),
        APPLICATION_NAME_HASH, timer_period, NULL,
        NULL, NULL, SDP, DMA)) {
    log_error("failed to set up the simulation interface");
    return false;
  }

  return true;
} // }}}

static bool __init_params(uint32_t *timer_offset) { // {{{
  params_sdram = data_specification_get_region(PARAMS, data);

  loss_function_id = params_sdram->loss_function_id;
  K = params_sdram->K;
  min_pre_key = params_sdram->min_pre_key;
  min_y_key = params_sdram->min_y_key;

  *timer_offset = params_sdram->timer_offset;

  return true;
} // }}}

void base_init() { // {{{
  uint32_t timer_period, timer_offset;

  // Start the time at "-1" so that the first tick will be 0
  time = UINT32_MAX;

  if (!__init_simulation_and_data_spec(&timer_period)) {
    log_error("Error in initializing simulation - exiting!");
    rt_error(RTE_SWERR);
  }

  if (!__init_params(&timer_offset)) {
    log_error("Error in initializing base parameters - exiting!");
    rt_error(RTE_SWERR);
  }

  spin1_set_timer_tick_and_phase(timer_period, timer_offset);

  __init_dtcm();
} // }}}

void receive_data_void(uint key, uint unknown) { // {{{
  use(key);
  use(unknown);
  log_error("this should never ever be done");
} // }}}

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

void update(uint ticks, uint b) { // {{{
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
      send(keys[i], loss_i);
    }
    reset();
  }
} // }}}

void c_main(void) { // {{{
  base_init();

  // register callbacks
  spin1_callback_on(MCPL_PACKET_RECEIVED, receive, MC_PACKET);
  spin1_callback_on(TIMER_TICK, update, TIMER);

  reset();

  // start execution
  log_info("\nStarting simulation\n");
  simulation_run();
} // }}}
