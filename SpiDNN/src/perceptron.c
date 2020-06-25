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
  // softmax partitions are explicitly allocated in the key space
  // beginning from 0.
  if (key >= min_softmax_key && key < min_pre_key) {
    softmax_denominator += payload;
    received_softmax_counter++;
  } else {
    receive_potential_from_pre_layer(key, payload);
  }
#else
  receive_potential_from_pre_layer(key, payload);
#endif

/*
#ifdef trainable
  if (key > min_next_key) {
    // TODO: do backward stuff
  } else {
    receive_potential_from_pre_layer(key, payload);
  }
#else
  receive_potential_from_pre_layer(key, payload);
#endif
*/
}

void update(uint ticks, uint b) { // {{{
  use(b);
  use(ticks);

  time++;

#ifdef softmax
  if (received_softmax_counter == softmax_layer_size) {

    potential = potential / (softmax_denominator + potential);
    send(forward_key, potential);
    reset();

  } else if (FORWARD_PASS_COMPLETE) {
    activate();
    send(softmax_key, potential);

    // reset so data is not send twice for softmax
    received_potentials_counter = 0;
  }
#else
  if (FORWARD_PASS_COMPLETE) {
    activate();
    send(forward_key, potential);
#ifndef trainable
    reset();
#endif
  }
#ifdef trainable
  else if (BACKWARD_PASS_COMPLETE) {
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
#endif
} // }}}

void c_main(void) { // {{{
  base_init();

#ifdef softmax
  log_info("HELLO FROM SOFTMAX");
  softmax_params_sdram =
    data_specification_get_region(INSTANCE_PARAMS, data);

  softmax_key = softmax_params_sdram->key;
  min_softmax_key = softmax_params_sdram->min_layer_key;
  softmax_layer_size = softmax_params_sdram->layer_size;
#else
  perceptron_params_sdram =
    data_specification_get_region(INSTANCE_PARAMS, data);

  activation_function_id =
    perceptron_params_sdram->activation_function_id;
#endif

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
