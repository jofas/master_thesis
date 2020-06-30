#ifdef trainable
#include "trainable.h"
#else
#include "perceptron.h"
#endif

void activate() {
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
}

void receive(uint key, float payload) {
  //log_info("received potential from %d: %f", key, payload);

#ifdef softmax
  // min_pre_key will always be bigger than min_softmax_key, because
  // softmax partitions are touched by the toolchain before forward
  // and backward partitions
  if (key < min_pre_key) {
    receive_softmax(payload);
  } else
#elif defined trainable
  // min_next_key will always be bigger than min_pre_key, because
  // the forward partition is touched by the toolchain before the
  // backward partition
  if (key >= min_next_key) {
    receive_backward(key, payload);
  } else
#endif
  {
    if (received_potentials_counter == 0) {
      potential = .0;
    }
    receive_forward(key, payload);
#ifndef softmax
  //log_error("received shit from softmax: %f", payload);
  //rt_error(RTE_SWERR);
#endif
  }
}

void update(uint ticks, uint b) {
  use(b);
  use(ticks);

  spiDNN_time++;

#ifdef softmax
  if (softmax_pass_complete()) {
    potential = potential / softmax_denominator;
    send(forward_key, (void *)&potential);
    //log_error("sending shit: %f", potential);
    //rt_error(RTE_SWERR);
  }
#endif

  if (forward_pass_complete()) {
    activate();
#ifdef softmax
    send(softmax_key, (void *)&potential);
#else
    send(forward_key, (void *)&potential);
#endif
  }

#ifdef trainable
  if (backward_pass_complete()) {
    backward_passes++;
    update_gradients();

    if (BATCH_COMPLETE) {
      update_weights();
      reset_batch();
      // TODO: get rid of this call (simulation_exit not possible so
      // get information from spiDNN side (epochs, epochs_len,...))
      sark_mem_cpy((void *)weights_sdram, (void *)weights,
        sizeof(float) * n_weights);
    }

    send(backward_key, (void *)&neuron_error);
  }
#endif
}

void c_main(void) {
  base_init();

  instance_init();

  weights_init();

#ifdef trainable
  trainable_init();
  reset_batch();
#endif

  // register callbacks
  spin1_callback_on(MCPL_PACKET_RECEIVED, receive, MC_PACKET);
  spin1_callback_on(TIMER_TICK, update, TIMER);

  log_info("\nStarting simulation\n");
  simulation_run();
}
