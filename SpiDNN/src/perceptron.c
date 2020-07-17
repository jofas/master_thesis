#include "perceptron.h"

#ifdef trainable
#include "trainable.h"
#endif

#ifdef softmax
#include "softmax.h"
#endif

void receive(uint key, float payload) {
  //log_info("received potential from %d: %f", key, payload);

#ifdef softmax
  // min_pre_key will always be bigger than min_softmax_key, because
  // softmax partitions are touched by the toolchain before forward
  // and backward partitions
  if (key < min_pre_key) {
    receive_softmax(payload);
    return;
  }
#endif

#ifdef trainable
  // min_next_key will always be bigger than min_pre_key, because
  // the forward partition is touched by the toolchain before the
  // backward partition
  if (key >= min_next_key) {
    receive_backward(key, payload, 1, &potential);
    return;
  }
#endif

  if (spiDNN_received_potentials_counter == 0)
    potential = .0;

  receive_forward(key, payload);
}

void update(uint ticks, uint b) {
  use(b);
  use(ticks);

  spiDNN_time++;

#ifdef softmax
  if (softmax_pass_complete()) {
    potential = potential / softmax_denominator;
    send(forward_key, (void *)&potential);
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
    backward_passes_counter++;
    batch_counter++;

    update_gradients(activation_function_id, 1, n_potentials, &potential);

    if (BATCH_COMPLETE) {
      update_weights(n_weights, weights);
      if (FIT_COMPLETE) {
        sark_mem_cpy((void *)weights_sdram, (void *)weights,
          sizeof(float) * n_weights);
      }
      reset_batch(n_weights);
    }

    send(backward_key, (void *)errors);
  }
#endif
}

void c_main(void) {
  base_init();

  weights_init();

#ifdef softmax
  softmax_init();
#endif

#ifdef trainable
  trainable_init(n_weights, 1);
#endif

  // register callbacks
  spin1_callback_on(MCPL_PACKET_RECEIVED, receive, MC_PACKET);
  spin1_callback_on(TIMER_TICK, update, TIMER);

  log_info("\nStarting simulation\n");
  simulation_run();
}
