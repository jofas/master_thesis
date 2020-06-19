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

void update(uint ticks, uint b) { // {{{
  use(b);
  use(ticks);

  time++;

  if ((received_potentials_counter == K) &&
      (received_y_counter == K))
  {
    compute_loss();
    // motherfucker
    // a partition for each output neuron
    // jeeeeezzzus
    // K keys.... goddamn, how do i do that with my partition manager
    // TODO: continue here after I've changed python code to have a
    //       memory region with the K keys and changed the backward
    //       pass graph
    send(my_key);
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
