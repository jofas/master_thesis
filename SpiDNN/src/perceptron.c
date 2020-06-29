#ifdef trainable
#include "trainable.h"
#else
#include "perceptron.h"
#endif

// instance variables
#ifdef softmax
  softmax_params_region_t *softmax_params_sdram;

  uint softmax_key;
  uint min_softmax_key;
  uint softmax_layer_size;

  float softmax_denominator;
  uint received_softmax_counter;
#else
  perceptron_params_region_t *perceptron_params_sdram;

  uint activation_function_id;
#endif

void reset() {
  potential = .0;
  received_potentials_counter = 0;

  for (uint i=0; i < N_POTENTIALS; i++) {
    received_potentials[i] = false;
  }

#ifdef softmax
  softmax_denominator = .0;
  received_softmax_counter = 0;
#endif

#ifdef trainable
  error = .0;
  received_errors_counter = 0;
#endif
}

void instance_init() {
#ifdef softmax
  softmax_params_sdram =
    data_specification_get_region(INSTANCE_PARAMS, data_spec_meta);

  softmax_key = softmax_params_sdram->key;
  min_softmax_key = softmax_params_sdram->min_layer_key;
  softmax_layer_size = softmax_params_sdram->layer_size;
#else
  perceptron_params_sdram =
    data_specification_get_region(INSTANCE_PARAMS, data_spec_meta);

  activation_function_id =
    perceptron_params_sdram->activation_function_id;
#endif
}

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
    softmax_denominator += payload;
    received_softmax_counter++;
  } else
#elif defined trainable
  // min_next_key will always be bigger than min_pre_key, because
  // the forward partition is touched by the toolchain before the
  // backward partition
  if (key >= min_next_key) {
    if (is_output_layer) {
      error += payload;
    } else {
      error += payload * next_layer_weights[key - min_next_key];
      next_layer_gradients[key - min_next_key] += payload * potential;
    }
    received_errors_counter++;
  } else
#endif
  {
    receive_potential_from_pre_layer(key, payload);
  }
}

void update(uint ticks, uint b) {
  use(b);
  use(ticks);

  time++;

#ifdef softmax
  // TODO: current implementation does not support single neuron
  //       softmax layer ... change to sending potential to self as
  //       well
  if (SOFTMAX_PASS_COMPLETE) {
    potential = potential / softmax_denominator;
    send(forward_key, (void *)&potential);
#ifdef trainable
    received_softmax_counter = 0;
#else
    reset();
#endif
    return;
  }
#endif

  if (FORWARD_PASS_COMPLETE) {
    activate();
#ifdef softmax
    send(softmax_key, (void *)&potential);
    // reset so data is not send twice for softmax (update being
    // executed again before SOFTMAX_PASS_COMPLETE)
    received_potentials_counter = 0;
#elif !defined trainable
    send(forward_key, (void *)&potential);
    // only reset when a normal perceptron (not softmax) and not
    // trainable
    reset();
#else
    send(forward_key, (void *)&potential);
    // reset so data is not send twice during forward pass
    received_potentials_counter = 0;
#endif
    return;
  }

#ifdef trainable
  if (BACKWARD_PASS_COMPLETE) {
    log_info("backward_pass_complete. Error is: %f", error);

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

    reset();
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

  reset();

  log_info("\nStarting simulation\n");
  simulation_run();
}
