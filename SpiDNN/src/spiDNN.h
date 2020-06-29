#include "spin1_api.h"
#include "common-typedefs.h"
#include <data_specification.h>
#include <simulation.h>
#include <sark.h>
#include <debug.h>
#include <math.h>


/* function which has to be implemented by a machine vertex including
 * spiDNN.h */
void __init_base_params(
    uint32_t *timer_offset, uint32_t *n_potentials, uint32_t *min_pre_key);


//! values for the priority for each callback
typedef enum callback_priorities {
    MC_PACKET = -1,
    SDP = 1,
    TIMER = 2,
    DMA = 3
} callback_priorities;


const uint SYSTEM_REGION=0;

static uint32_t time = UINT32_MAX;
data_specification_metadata_t *data_spec_meta = NULL;

// value for turning on and off interrupts
uint cpsr = 0;


float *potentials;
bool *received_potentials;
uint received_potentials_counter;
uint min_pre_key;

void send(uint key, void *payload) {
  uint send_bytes;
  sark_mem_cpy((void *)&send_bytes, payload, sizeof(uint));

  //log_info("sending value: %f with key: %d", payload, key);

  while (!spin1_send_mc_packet(key, send_bytes, WITH_PAYLOAD)) {
    spin1_delay_us(1);
  }
}

void receive_potential_from_pre_layer(uint key, float payload) {
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
}

static bool init_simulation_and_data_spec(uint32_t *timer_period) {
  // Get the address this core's DTCM data starts at from SRAM
  data_spec_meta = data_specification_get_data_address();

  // Read the header
  if (!data_specification_read_header(data_spec_meta)) {
    log_error("failed to read the data spec header");
    return false;
  }

  // Get the timing details and set up the simulation interface
  if (!simulation_initialise(
        data_specification_get_region(SYSTEM_REGION, data_spec_meta),
        APPLICATION_NAME_HASH, timer_period, NULL,
        NULL, NULL, SDP, DMA)) {
    log_error("failed to set up the simulation interface");
    return false;
  }

  return true;
}

void base_init() {
  uint32_t timer_period, timer_offset, n_potentials;

  if (!init_simulation_and_data_spec(&timer_period)) {
    log_error("Error in initializing simulation - exiting!");
    rt_error(RTE_SWERR);
  }

  __init_base_params(&timer_offset, &n_potentials, &min_pre_key);

  potentials = (float *)malloc(sizeof(float) * n_potentials);
  received_potentials = (bool *)malloc(sizeof(bool) * n_potentials);

  spin1_set_timer_tick_and_phase(timer_period, timer_offset);
}

void receive_data_void(uint key, uint unknown) {
  use(key);
  use(unknown);
  log_error("this should never ever be done");
}
