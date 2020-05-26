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
#include <debug.h>
#include <circular_buffer.h>

#define STEPS 50

/*! multicast routing keys to communicate with neighbours */
uint my_key;
uint32_t my_state = 0;

/*! buffer used to store spikes */
static circular_buffer input_buffer;
static uint32_t current_payload;

int alive_states_recieved_this_tick = 0;
int dead_states_recieved_this_tick = 0;

//! recorded data items
uint32_t size_written = 0;

//! control value, which says how many timer ticks to run for before exiting
static uint32_t simulation_ticks = 0;

static uint32_t time = 0;
data_specification_metadata_t *data = NULL;

// value for turning on and off interrupts
uint cpsr = 0;

//! human readable definitions of each region in SDRAM
typedef enum regions_e {
    SYSTEM_REGION,
    PARAMS,
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
    uint32_t timer_offset;
    uint32_t my_state;
} params_region_t;

// pointer to sdram region containing the parameters of the conway
// cell
params_region_t *params_sdram;

void receive_data(uint key, uint payload) { // {{{
    use(key);
    //log_info("the key i've received is %d\n", key);
    //log_info("the payload i've received is %d\n", payload);
    // If there was space to add spike to incoming spike queue
    if (!circular_buffer_add(input_buffer, payload)) {
        log_info("Could not add state");
    }
} // }}}

void do_safety_check(void) { // {{{
    // do a safety check on number of states. Not like we can fix it
    // if we've missed events
    cpsr = spin1_int_disable();
    int total = alive_states_recieved_this_tick +
	    dead_states_recieved_this_tick;
    if (total != 8){
         log_error("didn't receive the correct number of states");
         log_error("only received %d states", total);
    }
    log_debug("only received %d alive states",
	    alive_states_recieved_this_tick);
    log_debug("only received %d dead states",
	    dead_states_recieved_this_tick);
    spin1_mode_restore(cpsr);
} // }}}

void read_input_buffer(void) { // {{{
    cpsr = spin1_int_disable();
    circular_buffer_print_buffer(input_buffer);

    // pull payloads from input_buffer. Filter for alive and dead states
    for (uint32_t counter = 0; counter < 8; counter++) {
        bool success = circular_buffer_get_next(input_buffer, &current_payload);
        if (success) {
            if (current_payload == DEAD) {
                 dead_states_recieved_this_tick += 1;
            } else if (current_payload == ALIVE) {
                 alive_states_recieved_this_tick += 1;
            } else {
                 log_error("Not recognised payload");
            }
        } else {
            log_debug("couldn't read state from my neighbours.");
        }

    }
    spin1_mode_restore(cpsr);
} // }}}

void send_state(void) { // {{{
    // reset for next iteration
    alive_states_recieved_this_tick = 0;
    dead_states_recieved_this_tick = 0;

    // send my new state to the simulation neighbours
    log_info("sending my state of %d via multicast with key %d",
	    my_state, my_key);

    while (!spin1_send_mc_packet(my_key, my_state, WITH_PAYLOAD)) {
        spin1_delay_us(1);
    }

    log_info("sent my state via multicast");
} // }}}

void next_state(void) { // {{{
    // calculate new state from the total received so far
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
} // }}}

void update(uint ticks, uint b) { // {{{
    use(b);
    use(ticks);

    time++;

    log_info("on tick %d of %d", time, simulation_ticks);

    // check that the run time hasn't already elapsed and thus needs to be
    // killed
    if (time >= STEPS) {
        // fall into the pause resume mode of operating
        simulation_handle_pause_resume(NULL);
        log_info("Simulation complete.");
        // switch to state where host is ready to read
        simulation_ready_to_read();

        //sark_cpu_state(CPU_STATE_WAIT);
        return;
    }

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
            APPLICATION_NAME_HASH, timer_period, NULL,//&simulation_ticks,
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
    my_state = params_sdram->my_state;

    log_info("my key is %d", my_key);
    log_info("my offset is %d", params_sdram->timer_offset);
    log_info("my initial state is %d", my_state);

    // initialise my input_buffer for receiving packets
    input_buffer = circular_buffer_initialize(256);
    if (input_buffer == 0) {
        log_error("failed initializing receiving buffer");
        return false;
    }
    log_info("input_buffer initialised");

    return true;
} // }}}

void c_main(void) { // {{{
    log_info("starting conway_cell");

    // Load DTCM data
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

    // start execution
    log_info("Starting\n");

    // Start the time at "-1" so that the first tick will be 0
    time = UINT32_MAX;

    simulation_run();
} // }}}
