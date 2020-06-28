#include "spin1_api.h"
#include "common-typedefs.h"
#include <data_specification.h>
#include <simulation.h>
#include <sark.h>
#include <debug.h>
#include <math.h>

#define BIAS weights[n_weights - 1]
#define N_POTENTIALS n_weights - 1
#define FORWARD_PASS_COMPLETE received_potentials_counter == N_POTENTIALS
#define SOFTMAX_PASS_COMPLETE received_softmax_counter == softmax_layer_size
#define BACKWARD_PASS_COMPLETE received_errors_counter == n_errors


/* structs and enums */

//! human readable definitions of each region in SDRAM
typedef enum regions_e { // {{{
    SYSTEM_REGION,
    BASE_PARAMS,
    WEIGHTS,
    INSTANCE_PARAMS,
    TRAINABLE_PARAMS,
    NEXT_LAYER_WEIGHTS,
} regions_e; // }}}

//! human readable definitions of the activation functions (except
//! softmax, which is handled by another type of perceptron)
typedef enum activations_e { // {{{
  IDENTITY = 0,
  RELU = 1,
  SIGMOID = 2,
  TANH = 3,
  //SOFTMAX,
} activations_e; // }}}

//! definitions of each element in the base_params region
typedef struct base_params_region { // {{{
  uint32_t has_key;
  uint32_t forward_key;
  uint32_t min_pre_key;
  uint32_t timer_offset;
  uint32_t n_weights;
} base_params_region_t; // }}}

//! definitions of each element in the instance_params region, when
//! perceptron has other activation function than softmax
typedef struct perceptron_params_region { // {{{
  uint32_t activation_function_id;
} perceptron_params_region_t; // }}}

//! definitions of each element in the instance_params region, when
//! perceptron has softmax as its activation
typedef struct softmax_params_region { // {{{
  uint32_t key;
  uint32_t min_layer_key;
  uint32_t layer_size;
} softmax_params_region_t; // }}}

//! definitions of each element in the trainable_params region
typedef struct trainable_params_region { // {{{
  uint32_t batch_size;
  uint32_t backward_key;
  uint32_t min_next_key;
  uint32_t n_errors;
  uint32_t is_output_layer;
} trainable_params_region_t; // }}}

//! values for the priority for each callback
typedef enum callback_priorities { // {{{
    MC_PACKET = -1,
    SDP = 1,
    TIMER = 2,
    DMA = 3
} callback_priorities; // }}}


/* global variables */

uint forward_key;

uint min_pre_key;

uint n_weights;

float *weights;

float *potentials;
bool  *received_potentials;
uint received_potentials_counter;

float potential;

float *weights_sdram;
base_params_region_t *base_params_sdram;

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

// additional trainable variables
#ifdef trainable
  trainable_params_region_t *trainable_params_sdram;
  float *next_layer_weights_sdram;

  uint batch_size;
  uint backward_key;
  uint min_next_key;
  uint n_errors;
  uint is_output_layer;

  float *next_layer_weights;

  float *gradients;
  float *next_layer_gradients;

  float error;

  uint received_errors_counter;
  uint backward_passes;
#endif

static uint32_t time;
data_specification_metadata_t *data = NULL;

// value for turning on and off interrupts
uint cpsr = 0;


/* functions */

void generate_potential() { // {{{
  for (uint i = 0; i < N_POTENTIALS; i++) {
    potential += potentials[i] * weights[i];
  }

  potential += BIAS;
} // }}}

void receive_potential_from_pre_layer(uint key, float payload) { // {{{
  uint idx = key - min_pre_key;

  if (received_potentials[idx]) {
    log_error("received potential too fast. Last input wasn't
               properly processed yet - exiting!");
    rt_error(RTE_SWERR);
  } else {
    potentials[idx] = payload;
    received_potentials[idx] = true;
    received_potentials_counter++;
  }
} // }}}

void reset() { // {{{
  potential = .0;
  received_potentials_counter = 0;

  for (uint i=0; i < N_POTENTIALS; i++) {
    received_potentials[i] = false;
  }
#ifdef softmax
  softmax_denominator = .0;

  // 1 because we have already 'received' the potential of this per-
  // ceptron instance.
  received_softmax_counter = 1;
#endif

#ifdef trainable
  error = .0;
  received_errors_counter = 0;
#endif
} // }}}

void send(uint key, float payload) { // {{{
  uint send_bytes;
  sark_mem_cpy((void *)&send_bytes, &payload, sizeof(uint));

  log_info("sending value: %f with key: %d", payload, key);

  while (!spin1_send_mc_packet(key, send_bytes, WITH_PAYLOAD)) {
    spin1_delay_us(1);
  }
} // }}}

void __init_dtcm() { // {{{
  weights_sdram = data_specification_get_region(WEIGHTS, data);

  weights = (float *)malloc(sizeof(float) * n_weights);

  sark_mem_cpy((void *)weights, (void *)weights_sdram,
    sizeof(float) * n_weights);

  potentials = (float *)malloc(sizeof(float) * N_POTENTIALS);

  received_potentials = (bool *)malloc(sizeof(bool) * N_POTENTIALS);
} // }}}

static bool __init_simulation_and_data_spec(uint32_t *timer_period) { // {{{
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
        APPLICATION_NAME_HASH, timer_period, NULL,
        NULL, NULL, SDP, DMA)) {
    log_error("failed to set up the simulation interface");
    return false;
  }

  return true;
} // }}}

void __init_base_params(uint32_t *timer_offset) { // {{{
  base_params_sdram = data_specification_get_region(BASE_PARAMS, data);
  /* TODO: remove has_key from base region
  if (!base_params_sdram->has_key) {
    log_error(
      "this conways cell can't affect anything, deduced as an error,"
      "please fix the application fabric and try again");
    return false;
  }
  */
  forward_key = base_params_sdram->forward_key;
  min_pre_key = base_params_sdram->min_pre_key;

  n_weights = base_params_sdram->n_weights;

  *timer_offset = base_params_sdram->timer_offset;
} // }}}

void base_init() { // {{{
  uint32_t timer_period, timer_offset;

  // Start the time at "-1" so that the first tick will be 0
  time = UINT32_MAX;

  if (!__init_simulation_and_data_spec(&timer_period)) {
    log_error("Error in initializing simulation - exiting!");
    rt_error(RTE_SWERR);
  }

  __init_base_params(&timer_offset);

  __init_dtcm();

  spin1_set_timer_tick_and_phase(timer_period, timer_offset);
} // }}}

void instance_init() { // {{{
#ifdef softmax
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
} // }}}

#ifdef trainable
  void reset_batch() {
    backward_passes = 0;

    for (uint i=0; i < n_weights; i++) {
      gradients[i] = .0;
    }

    if (!is_output_layer) {
      for (uint i=0; i < n_errors; i++) {
        next_layer_gradients[i] = .0;
      }
    }
  }

  void on_exit_extract_weights() { // {{{
    // TODO: update weights one last time (incomplete batch)

    log_info("Extracting weights");

    /*
    for (uint i=0; i < n_weights; i++) {
      weights_sdram[i] = weights[i];
      if (i == N_POTENTIALS) {
        log_info("COPYING BIAS TO SDRAM: %f", weights_sdram[i]);
      }
    }*/

    sark_mem_cpy((void *)weights_sdram, (void *)weights,
      sizeof(float) * n_weights);

    //log_info("COPYING BIAS TO SDRAM: %f", weights_sdram[N_POTENTIALS]);
    //log_info("done extracting weights");
  } // }}}

  void trainable_init() { // {{{
    trainable_params_sdram =
      data_specification_get_region(TRAINABLE_PARAMS, data);

    batch_size = trainable_params_sdram->batch_size;
    backward_key = trainable_params_sdram->backward_key;
    min_next_key = trainable_params_sdram->min_next_key;
    n_errors = trainable_params_sdram->n_errors;
    is_output_layer = trainable_params_sdram->is_output_layer;

    gradients = (float *)malloc(sizeof(float) * n_weights);

    if (!is_output_layer) {
      next_layer_weights_sdram =
        data_specification_get_region(NEXT_LAYER_WEIGHTS, data);

      next_layer_weights = (float *)malloc(sizeof(float) * n_errors);

      sark_mem_cpy(
        (void *)next_layer_weights,
        (void *)next_layer_weights_sdram,
        sizeof(float) * n_errors
      );

      next_layer_gradients = (float *)malloc(sizeof(float) * n_errors);
    }

    simulation_set_exit_function(on_exit_extract_weights);
  } // }}}
#endif

void receive_data_void(uint key, uint unknown) { // {{{
  use(key);
  use(unknown);
  log_error("this should never ever be done");
} // }}}

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

void receive(uint key, float payload) { // {{{
  log_info("received potential from %d: %f", key, payload);

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
} // }}}

void update(uint ticks, uint b) { // {{{
  use(b);
  use(ticks);

  time++;

#ifdef softmax
  // TODO: current implementation does not support single neuron
  //       softmax layer ... change to sending potential to self as
  //       well
  if (SOFTMAX_PASS_COMPLETE) {
    potential = potential / (softmax_denominator + potential);
    send(forward_key, potential);
#ifdef trainable
    received_softmax_counter = 1;
#else
    reset();
#endif
    return;
  }
#endif

  if (FORWARD_PASS_COMPLETE) {
    activate();
#ifdef softmax
    send(softmax_key, potential);
    // reset so data is not send twice for softmax (update being
    // executed again before SOFTMAX_PASS_COMPLETE)
    received_potentials_counter = 0;
#elif !defined trainable
    send(forward_key, potential);
    // only reset when a normal perceptron (not softmax) and not
    // trainable
    reset();
#else
    send(forward_key, potential);
    // reset so data is not send twice during forward pass
    received_potentials_counter = 0;
#endif
    return;
  }

#ifdef trainable
  if (BACKWARD_PASS_COMPLETE) {
    log_info("backward_pass_complete. Error is: %f", error);

    backward_passes++;

    // TODO: depending on activation function
    float neuron_error = error * potential * (1 - potential);

    // when all errors are received -> compute gradients for each
    // weight -> sum in *gradients
    for (uint i=0; i < N_POTENTIALS; i++) {
      gradients[i] += neuron_error * potentials[i];
    }
    // special case: bias neuron has potential := 1
    gradients[N_POTENTIALS] += neuron_error;

    log_info("gradient for bias: %f", gradients[N_POTENTIALS]);

    // if batch_size full -> update weights with learning_rate * gradient
    // also update next_layer_weights if applicable
    if (backward_passes == batch_size) {
      for (uint i=0; i < n_weights; i++) {
        weights[i] -= 0.01 * gradients[i];
      }

      if (!is_output_layer) {
        for (uint i=0; i < n_errors; i++) {
          next_layer_weights[i] -= 0.01 * next_layer_gradients[i];
        }
      }

      reset_batch();
      // WHY TF NOT WORKING in on exit??
      on_exit_extract_weights();
    }

    send(backward_key, neuron_error);

    reset();
  }
#endif
} // }}}

void c_main(void) { // {{{
  base_init();

  instance_init();

#ifdef trainable
  trainable_init();
  reset_batch();
#endif

  // register callbacks
  spin1_callback_on(MCPL_PACKET_RECEIVED, receive, MC_PACKET);
  spin1_callback_on(TIMER_TICK, update, TIMER);

  reset();

  // start execution
  log_info("\nStarting simulation\n");
  simulation_run();
} // }}}
