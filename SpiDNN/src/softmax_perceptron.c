/*
 * Copyright (c) 2017-2019 The University of Manchester
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "spin1_api.h"
#include "common-typedefs.h"
#include <data_specification.h>
#include <simulation.h>
#include <sark.h>
#include <debug.h>
#include <circular_buffer.h>
#include <math.h>

#define BIAS weights[n_weights - 1]

/*! multicast routing keys to communicate with neighbours */
uint my_key;
uint softmax_key;

uint min_pre_key;
uint min_softmax_key;

uint softmax_layer_size;
uint n_weights;
uint n_potentials;

float *weights;

float *potentials;
bool  *received_potentials;
uint received_potentials_counter = 0;

float potential;

float softmax_denominator;
uint received_softmax_counter;

static uint32_t time;
data_specification_metadata_t *data = NULL;

// value for turning on and off interrupts
uint cpsr = 0;

//! human readable definitions of each region in SDRAM
typedef enum regions_e { // {{{
    SYSTEM_REGION,
    PARAMS,
    WEIGHTS,
} regions_e; // }}}

//! values for the priority for each callback
typedef enum callback_priorities { // {{{
    MC_PACKET = -1,
    SDP = 1,
    TIMER = 2,
    DMA = 3
} callback_priorities; // }}}

//! definitions of each element in the transmission region
typedef struct params_region { // {{{
    uint32_t has_key;
    uint32_t my_key;
    uint32_t softmax_key;
    uint32_t min_pre_key;
    uint32_t min_softmax_key;
    uint32_t timer_offset;
    uint32_t softmax_layer_size;
    uint32_t n_weights;
} params_region_t; // }}}

// pointer to sdram region containing the parameters of the conway
// cell
params_region_t *params_sdram;
float *weights_sdram;

void generate_potential() { // {{{
  for (uint i = 0; i < n_potentials; i++) {
    potential += potentials[i] * weights[i];
  }

  potential += BIAS;
} // }}}

void activate() { // {{{
  generate_potential();
  potential = exp(potential);
} // }}}

void reset() { // {{{
  potential = .0;

  softmax_denominator = .0;

  for (uint i=0; i < n_weights - 1; i++) {
    received_potentials[i] = false;
  }

  // 1 because we have already 'received' the potential of this per-
  // ceptron instance.
  received_softmax_counter = 1;
} // }}}

void receive_potential(uint key, float payload) { // {{{
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

// cause compiler warning because of type missmatch of payload but
// works just fine. C is awesome.
void receive_data(uint key, float payload) { // {{{
  //log_info("received payload: %f from: %d", payload, key);

  // min_pre_key will always be bigger than min_softmax_key, because
  // softmax partitions are explicitly allocated in the key space
  // beginning from 0.
  if (key >= min_softmax_key && key < min_pre_key) {
    softmax_denominator += payload;
    received_softmax_counter++;
  } else {
    receive_potential(key, payload);
  }
} // }}}

void send(uint key) { // {{{
    uint send_bytes;
    sark_mem_cpy((void *)&send_bytes, &potential, sizeof(uint));

    while (!spin1_send_mc_packet(key, send_bytes, WITH_PAYLOAD)) {
        spin1_delay_us(1);
    }
} // }}}

void update(uint ticks, uint b) { // {{{
  use(b);
  use(ticks);

  time++;

  if (received_softmax_counter == softmax_layer_size) {

    potential = potential / (softmax_denominator + potential);
    send(my_key);
    reset();

  } else if (received_potentials_counter == n_potentials) {
    //log_info("on tick %d I'm sending a potential", time);

    // presses potential through the activation function
    activate();
    send(softmax_key);

    // reset so data is not send twice for softmax
    received_potentials_counter = 0;
  }
  //log_info("sent potential %f\n", potential);
} // }}}

void receive_data_void(uint key, uint unknown) { // {{{
    use(key);
    use(unknown);
    log_error("this should never ever be done");
} // }}}

void initialize_dtcm() { // {{{
  weights = (float *)malloc(sizeof(float) * n_weights);

  sark_mem_cpy(
    (void *)weights, (void *)weights_sdram, sizeof(float) * n_weights
  );

  potentials = (float *)malloc(sizeof(float) * n_potentials);

  received_potentials = (bool *)malloc(sizeof(bool) * n_potentials);
} // }}}

static bool initialize(uint32_t *timer_period) { // {{{
  log_info("Initialise: started");

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

  // initialise transmission keys
  params_sdram = data_specification_get_region(PARAMS, data);

  if (!params_sdram->has_key) {
    log_error(
      "this conways cell can't affect anything, deduced as an error,"
      "please fix the application fabric and try again");
    return false;
  }

  my_key = params_sdram->my_key;
  softmax_key = params_sdram->softmax_key;

  min_pre_key = params_sdram->min_pre_key;
  min_softmax_key = params_sdram->min_softmax_key;

  softmax_layer_size = params_sdram->softmax_layer_size;
  n_weights = params_sdram->n_weights;
  n_potentials = n_weights - 1;

  //log_info("my key is %d", my_key);
  //log_info("my offset is %d", params_sdram->timer_offset);
  //log_info("my min_pre_key is %d", min_pre_key);
  //log_info("my n_weights is %d", n_weights);
  //log_info("my activation_function_id is %d", activation_function_id);
  //log_info("my pre_layer_activation_function_id is %d",
  //  pre_layer_activation_function_id);

  weights_sdram = data_specification_get_region(WEIGHTS, data);

  return true;
} // }}}

void c_main(void) { // {{{
    log_info("starting conway_cell");

    uint32_t timer_period;

    // initialise the model
    if (!initialize(&timer_period)) {
        log_error("Error in initialisation - exiting!");
        rt_error(RTE_SWERR);
    }

    log_info("setting timer to execute every %d microseconds with an
      offset of %d", timer_period, params_sdram->timer_offset);

    spin1_set_timer_tick_and_phase( timer_period
                                  , params_sdram->timer_offset );

    // register callbacks
    spin1_callback_on(MCPL_PACKET_RECEIVED, receive_data, MC_PACKET);
    spin1_callback_on(TIMER_TICK, update, TIMER);

    initialize_dtcm();

    reset();

    // Start the time at "-1" so that the first tick will be 0
    time = UINT32_MAX;

    // start execution
    log_info("\nStarting simulation\n");
    simulation_run();
} // }}}
