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

#include "perceptron.h"

typedef struct softmax_params_region { // {{{
  uint32_t key;
  uint32_t min_layer_key;
  uint32_t layer_size;
} softmax_params_region_t; // }}}

uint softmax_key;
uint min_softmax_key;
uint softmax_layer_size;

float softmax_denominator;
uint received_softmax_counter;

softmax_params_region_t *softmax_params_sdram;

void softmax_reset() { // {{{
  reset();

  softmax_denominator = .0;

  // 1 because we have already 'received' the potential of this per-
  // ceptron instance.
  received_softmax_counter = 1;
} // }}}

void activate() { // {{{
  generate_potential();
  potential = exp(potential);
} // }}}

void receive_data(uint key, float payload) { // {{{
  //log_info("received payload: %f from: %d", payload, key);

  // min_pre_key will always be bigger than min_softmax_key, because
  // softmax partitions are explicitly allocated in the key space
  // beginning from 0.
  if (key >= min_softmax_key && key < min_pre_key) {
    softmax_denominator += payload;
    received_softmax_counter++;
  } else {
    receive_potential_from_pre_layer(key, payload);
  }
} // }}}

void update(uint ticks, uint b) { // {{{
  use(b);
  use(ticks);

  time++;

  if (received_softmax_counter == softmax_layer_size) {

    potential = potential / (softmax_denominator + potential);
    send(my_key);
    softmax_reset();

  } else if (received_potentials_counter == N_POTENTIALS) {
    activate();
    send(softmax_key);

    // reset so data is not send twice for softmax
    received_potentials_counter = 0;
  }
} // }}}

void c_main(void) { // {{{
  base_init();

  softmax_params_sdram =
    data_specification_get_region(INSTANCE_PARAMS, data);

  softmax_key = softmax_params_sdram->key;
  min_softmax_key = softmax_params_sdram->min_layer_key;
  softmax_layer_size = softmax_params_sdram->layer_size;

  // register callbacks
  spin1_callback_on(MCPL_PACKET_RECEIVED, receive_data, MC_PACKET);
  spin1_callback_on(TIMER_TICK, update, TIMER);

  softmax_reset();

  // start execution
  log_info("\nStarting simulation\n");
  simulation_run();
} // }}}
