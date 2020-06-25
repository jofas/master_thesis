#include "perceptron.h"

void activate() { // {{{
  generate_potential();
#ifdef softmax
  potential = exp(potential);
#else
  switch (activation_function_id) {
    case IDENTITY:
      break;

    case RELU:
      potential = potential > .0 ? potential : .0;
      break;

    case SIGMOID:
      potential = 1. / (1. + exp(-potential));
      break;

    case TANH:
      potential = tanh(potential);
      break;

    default:
      log_error("Unknown activation function %d - exiting!",
        activation_function_id);
      rt_error(RTE_SWERR);
  }
#endif
} // }}}

void receive(uint key, float payload) {
#ifdef softmax
  // min_pre_key will always be bigger than min_softmax_key, because
  // softmax partitions are touched by the toolchain before forward
  // and backward partitions
  if (key < min_pre_key) {
    softmax_denominator += payload;
    received_softmax_counter++;
  } else
#elif defined trainable
  // min_next_key will always be bigger than min_pre_key, because
  // the forward partition is touched by the toolchain before the
  // backward partition
  if (key >= min_next_key) {
    // TODO: backward stuff
  } else
#endif
  {
    receive_potential_from_pre_layer(key, payload);
  }
}

void update(uint ticks, uint b) { // {{{
  use(b);
  use(ticks);

  time++;

#ifdef softmax
  if (SOFTMAX_PASS_COMPLETE) {

    potential = potential / (softmax_denominator + potential);
    send(forward_key, potential);
    reset();
    return;
  }
#endif

  if (FORWARD_PASS_COMPLETE) {
    activate();
#ifdef softmax
    send(softmax_key, potential);
    // reset so data is not send twice for softmax (update being
    // executed before SOFTMAX_PASS_COMPLETE
    received_potentials_counter = 0;
#elif !defined trainable
    send(forward_key, potential);
    // only reset when a normal perceptron (not softmax) and not
    // trainable
    reset();
#else
    send(forward_key, potential);
#endif
    return;
  }

#ifdef trainable
  if (BACKWARD_PASS_COMPLETE) {
    // when all errors are received -> compute gradients for each
    // weight -> sum in *gradients

    // pass error backwards with backward key
    //
    // TODO
    /*
    for (uint i=0; i < n_backward_keys; i++) {
      send(backward_keys[i]);
    }
    */

    // if batch_size full -> update weights with learning_rate * gradient

    reset();
  }
#endif
} // }}}

void c_main(void) { // {{{
  base_init();

  instance_init();

#ifdef trainable
  trainable_init();
#endif

  // register callbacks
  spin1_callback_on(MCPL_PACKET_RECEIVED, receive, MC_PACKET);
  spin1_callback_on(TIMER_TICK, update, TIMER);

  reset();

  // start execution
  log_info("\nStarting simulation\n");
  simulation_run();
} // }}}
