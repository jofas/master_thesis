#include "perceptron.h"

typedef struct perceptron_params_region { // {{{
  uint32_t activation_function_id;
} perceptron_params_region_t; // }}}

perceptron_params_region_t *perceptron_params_sdram;

uint activation_function_id;


void activate() { // {{{
  generate_potential();

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
} // }}}

void receive(uint key, float payload) {
#ifdef trainable
  if (key == backward_key) {
    // E_i -> delta E_i / delta out -> sum in error
    //
    //receive_error_from_next_layer(key, payload);
  } else if (key == forward_key) {
    receive_potential_from_pre_layer(key, payload);
  }
#else
  receive_potential_from_pre_layer(key, payload);
#endif
}

void update(uint ticks, uint b) { // {{{
  use(b);
  use(ticks);

  time++;

  if (FORWARD_PASS_COMPLETE) {
    //received_potentials_counter == N_POTENTIALS) {
    activate();
    send(forward_key);
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


    // if batch_size full -> update weights with learning_rate * gradient

    reset();
  }
#endif
} // }}}

void c_main(void) { // {{{
  base_init();

#ifdef trainable
  trainable_init();
#endif

  perceptron_params_sdram =
    data_specification_get_region(INSTANCE_PARAMS, data);

  activation_function_id =
    perceptron_params_sdram->activation_function_id;

  // register callbacks
  spin1_callback_on(MCPL_PACKET_RECEIVED, receive, MC_PACKET);
  spin1_callback_on(TIMER_TICK, update, TIMER);

  reset();

  // start execution
  log_info("\nStarting simulation\n");
  simulation_run();
} // }}}
