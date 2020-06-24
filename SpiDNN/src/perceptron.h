#include "spin1_api.h"
#include "common-typedefs.h"
#include <data_specification.h>
#include <simulation.h>
#include <sark.h>
#include <debug.h>
#include <math.h>

#define BIAS weights[n_weights - 1]
#define N_POTENTIALS n_weights - 1
#define FORWARD_PASS_COMPLETE received_potentials_counter == N_POTENTIALS
#define BACKWARD_PASS_COMPLETE received_errors_counter == n_errors

//! human readable definitions of each region in SDRAM
typedef enum regions_e { // {{{
    SYSTEM_REGION,
    BASE_PARAMS,
    WEIGHTS,
    INSTANCE_PARAMS,
    TRAINABLE_PARAMS,
    BACKWARD_KEYS,
} regions_e; // }}}

//! human readable definitions of the activation functions (except
//! softmax, which is handled by another type of perceptron)
typedef enum activations_e { // {{{
  IDENTITY = 0,
  RELU = 1,
  SIGMOID = 2,
  TANH = 3,
  //SOFTMAX,
} activations_e; // }}}

//! definitions of each element in the base_params region
typedef struct base_params_region { // {{{
  uint32_t has_key;
  uint32_t forward_key;
  uint32_t min_pre_key;
  uint32_t timer_offset;
  uint32_t n_weights;
} base_params_region_t; // }}}

//! definitions of each element in the trainable_params region
typedef struct trainable_params_region { // {{{
  uint32_t batch_size;
  uint32_t n_backward_keys;
  uint32_t min_next_key;
  uint32_t n_errors;
} trainable_params_region_t; // }}}

//! values for the priority for each callback
typedef enum callback_priorities { // {{{
    MC_PACKET = -1,
    SDP = 1,
    TIMER = 2,
    DMA = 3
} callback_priorities; // }}}


/* global variables */

uint forward_key;

uint min_pre_key;

uint n_weights;

float *weights;

float *potentials;
bool  *received_potentials;
uint received_potentials_counter;

float potential;

float *weights_sdram;
base_params_region_t *base_params_sdram;

#ifdef trainable
  trainable_params_region_t *trainable_params_sdram;
  uint *backward_keys_sdram;

  uint batch_size;
  uint n_backward_keys;
  uint min_next_key;
  uint n_errors;

  uint *backward_keys;

  float *gradients;
  float error;
  uint received_errors_counter;
#endif

static uint32_t time;
data_specification_metadata_t *data = NULL;

// value for turning on and off interrupts
uint cpsr = 0;


/* functions */

void generate_potential() { // {{{
  for (uint i = 0; i < N_POTENTIALS; i++) {
    potential += potentials[i] * weights[i];
  }

  potential += BIAS;
} // }}}

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

void reset() { // {{{
  potential = .0;

  for (uint i=0; i < N_POTENTIALS; i++) {
    received_potentials[i] = false;
  }

  received_potentials_counter = 0;

#ifdef trainable
  error = .0;
  received_errors_counter = 0;
#endif
} // }}}

void send(uint key) { // {{{
  uint send_bytes;
  sark_mem_cpy((void *)&send_bytes, &potential, sizeof(uint));

  while (!spin1_send_mc_packet(key, send_bytes, WITH_PAYLOAD)) {
    spin1_delay_us(1);
  }
} // }}}

void __init_dtcm() { // {{{
  weights_sdram = data_specification_get_region(WEIGHTS, data);

  weights = (float *)malloc(sizeof(float) * n_weights);

  sark_mem_cpy((void *)weights, (void *)weights_sdram,
    sizeof(float) * n_weights);

  potentials = (float *)malloc(sizeof(float) * N_POTENTIALS);

  received_potentials = (bool *)malloc(sizeof(bool) * N_POTENTIALS);
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

static bool __init_base_params(uint32_t *timer_offset) { // {{{
  base_params_sdram = data_specification_get_region(BASE_PARAMS, data);

  if (!base_params_sdram->has_key) {
    log_error(
      "this conways cell can't affect anything, deduced as an error,"
      "please fix the application fabric and try again");
    return false;
  }

  forward_key = base_params_sdram->forward_key;
  min_pre_key = base_params_sdram->min_pre_key;

  n_weights = base_params_sdram->n_weights;

  *timer_offset = base_params_sdram->timer_offset;

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

  if (!__init_base_params(&timer_offset)) {
    log_error("Error in initializing base parameters - exiting!");
    rt_error(RTE_SWERR);
  }

  spin1_set_timer_tick_and_phase(timer_period, timer_offset);

  __init_dtcm();
} // }}}

#ifdef trainable
  void on_exit_extract_weights() { // {{{
    // TODO: update weights one last time (incomplete batch)

    log_info("Extracting weights");

    sark_mem_cpy((void *)weights_sdram, (void *)weights,
      sizeof(float) * n_weights);
  } // }}}

  void trainable_init() { // {{{
    trainable_params_sdram =
      data_specification_get_region(TRAINABLE_PARAMS, data);

    batch_size = trainable_params_sdram->batch_size;
    n_backward_keys = trainable_params_sdram->n_backward_keys;
    min_next_key = trainable_params_sdram->min_next_key;
    n_errors = trainable_params_sdram->n_errors;

    backward_keys_sdram =
      data_specification_get_region(BACKWARD_KEYS, data);

    backward_keys = (uint *)malloc(sizeof(uint) * n_backward_keys);

    sark_mem_cpy((void *)backward_keys, (void *)backward_keys_sdram,
      sizeof(uint) * n_backward_keys);

    simulation_set_exit_function(on_exit_extract_weights);
  } // }}}
#endif

void receive_data_void(uint key, uint unknown) { // {{{
  use(key);
  use(unknown);
  log_error("this should never ever be done");
} // }}}
