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

/*! multicast routing keys to communicate with neighbours */
uint my_key;
uint min_pre_key;
uint n_weights;
float *weights;

bool received_potential = false;
float potential;

/*! buffer used to store spikes */
//static circular_buffer input_buffer;
static uint32_t current_payload;

//! recorded data items
uint32_t size_written = 0;

static uint32_t time;
data_specification_metadata_t *data = NULL;

// value for turning on and off interrupts
uint cpsr = 0;

//! human readable definitions of each region in SDRAM
typedef enum regions_e {
    SYSTEM_REGION,
    PARAMS,
    WEIGHTS,
} regions_e;

//! values for the priority for each callback
typedef enum callback_priorities {
    MC_PACKET = -1,
    SDP = 1,
    TIMER = 2,
    DMA = 3
} callback_priorities;

//! values for the states
typedef enum states_values {
    DEAD = 0,
    ALIVE = 1
} states_values;

//! definitions of each element in the transmission region
typedef struct params_region {
    uint32_t has_key;
    uint32_t my_key;
    uint32_t min_pre_key;
    uint32_t timer_offset;
    uint32_t n_weights;
} params_region_t;

// pointer to sdram region containing the parameters of the conway
// cell
params_region_t *params_sdram;
float *weights_sdram;

void reset_potential() {
  potential = weights[n_weights - 1];
}

// currently only sigmoid
float activate() {
  potential = exp(potential) / (exp(potential) + 1.);
}

void receive_data(uint key, float payload) { // {{{
    use(key);

    // TODO: receive all before received_potential
    //       is set to true (with a counter)
    log_info("received payload: %f", payload);

    received_potential = true;

    potential += weights[key - min_pre_key] * payload;

} // }}}

void send_potential(void) { // {{{
    // presses potential through the activation function
    activate();

    uint send_bytes;
    sark_mem_cpy((void *)&send_bytes, &potential, sizeof(float));

    while (!spin1_send_mc_packet(my_key, send_bytes, WITH_PAYLOAD)) {
        spin1_delay_us(1);
    }

    log_info("sent potential %f", potential);

    reset_potential();

} // }}}

void next_state(void) { // {{{
    // calculate new state from the total received so far
    /*
    if (my_state == ALIVE) {
        if (alive_states_recieved_this_tick <= 1) {
            my_state = DEAD;
        }
        if ((alive_states_recieved_this_tick == 2) |
                (alive_states_recieved_this_tick == 3)) {
            my_state = ALIVE;
        }
        if (alive_states_recieved_this_tick >= 4) {
            my_state = DEAD;
        }
    } else if (alive_states_recieved_this_tick == 3) {
        my_state = ALIVE;
    }
    */
} // }}}

void update(uint ticks, uint b) { // {{{
  use(b);
  use(ticks);

  time++;

  if (received_potential) {
    log_info("on tick %d I'm sending a potential", time);
    send_potential();
    received_potential = false;
  }

  /*
    if (time == 0) {
      log_info("Send my first state!");

      //next_state();
      send_state();

    } else {
      read_input_buffer();

      // find my next state
      next_state();

      // do a safety check on number of states. Not like we can fix it
      // if we've missed events
      do_safety_check();

      send_state();
    }
  */
} // }}}

void receive_data_void(uint key, uint unknown) { // {{{
    use(key);
    use(unknown);
    log_error("this should never ever be done");
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
    min_pre_key = params_sdram->min_pre_key;
    n_weights = params_sdram->n_weights;

    //log_info("my key is %d", my_key);
    //log_info("my offset is %d", params_sdram->timer_offset);
    //log_info("my min_pre_key is %d", min_pre_key);
    //log_info("my n_weights is %d", n_weights);

    weights_sdram = data_specification_get_region(WEIGHTS, data);

    /*
    // initialise my input_buffer for receiving packets
    input_buffer = circular_buffer_initialize(256);
    if (input_buffer == 0) {
        log_error("failed initializing receiving buffer");
        return false;
    }
    log_info("input_buffer initialised");
    */

    return true;
} // }}}

void copy_sdram_weights_to_dtcm() { // {{{
  weights = (float *)malloc(n_weights);

  for(uint i=0; i<n_weights; i++) {
    weights[i] = weights_sdram[i];
    //log_info("weight at %d: %f", i, weights[i]);
  }
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

    copy_sdram_weights_to_dtcm();

    reset_potential();

    // Start the time at "-1" so that the first tick will be 0
    time = UINT32_MAX;

    // start execution
    log_info("\nStarting simulation\n");
    simulation_run();
} // }}}
