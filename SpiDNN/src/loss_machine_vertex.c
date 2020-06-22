#include "loss_machine_vertex.h"

void receive(uint key, float payload) {
  // This works because min_y_key is guaranteed to be bigger than
  // min_pre_key, because the forward partition has a higher priority
  // than the y partition.
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
