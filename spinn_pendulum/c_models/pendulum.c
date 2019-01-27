//
//  bkout.c
//  BreakOut
//
//  Created by Steve Furber on 26/08/2016.
//  Copyright © 2016 Steve Furber. All rights reserved.
//
// Standard includes
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Spin 1 API includes
#include <spin1_api.h>

// Common includes
#include <debug.h>

// Front end common includes
#include <data_specification.h>
#include <simulation.h>
#include "random.h"
#include <math.h>
//#include <chrono>
//using namespace std::chrono;

#include <recording.h>

//----------------------------------------------------------------------------
// Macros
//----------------------------------------------------------------------------

// Frame delay (ms)
//#define time_increment 200 //14//20
/*
    number of bins for current angle of the pole
    number of bins for the force to be applied or number of spikes per tick equals a force
    mass of the cart
    mass of the pole
    initial starting angle
    velocity of the cart
    velocity of the pendulum
    base rate for the neurons to fire in each bin
    each spike equals a change in force to be applied (what is that amount)
    receptive field of each bin
    update model on each timer tick and on each spike received, or number of spikes per tick equals a force

    add option to rate (increased poisson P()) code and rank code
*/

//----------------------------------------------------------------------------
// Enumerations
//----------------------------------------------------------------------------
typedef enum
{
  REGION_SYSTEM,
  REGION_PENDULUM,
  REGION_RECORDING,
  REGION_DATA,
} region_t;

typedef enum
{
  SPECIAL_EVENT_ANGLE,
  SPECIAL_EVENT_ANGLE_V,
  SPECIAL_EVENT_CART,
  SPECIAL_EVENT_CART_V,
} special_event_t;

typedef enum // forward will be considered positive motion
{
  BACKWARD_MOTOR  = 0x0,
  FORWARD_MOTOR  = 0x1,
} arm_key_t;

//----------------------------------------------------------------------------
// Globals
//----------------------------------------------------------------------------

static uint32_t _time;

//! Should simulation run for ever? 0 if not
static uint32_t infinite_run;

mars_kiss64_seed_t kiss_seed;

int32_t current_score = 0;
int32_t reward_based = 1;

// experimental constraints and variables
float current_time = 0;
float max_motor_force = 10; // N
float min_motor_force = -10; // N
float motor_force = 0;
float force_increment = 100;
float track_length = 4.8; // m
float cart_position = 0; // m
float cart_velocity = 0;  // m/s
float cart_acceleration = 0;  // m/s^2
float highend_cart_v = 5; // used to calculate firing rate and bins
int max_pole_angle = 36;
int min_pole_angle = -36;
float pole_angle = 1;
float pole_velocity = 0; // angular/s
float pole_acceleration = 0; // angular/s^2
float highend_pole_v = 10; // used to calculate firing rate and bins

int max_firing_rate = 20;
float max_firing_prob = 0;
int encoding_scheme = 0; // 0: rate, 1: time, 2: rank (replace with type def
int number_of_bins = 20;
int central = 1; // if it's central that mean perfectly central on the track and angle is the lowest rate, else half

// experimental parameters
float half_pole_length = 0.5; // m
float gravity = 9.8; // m/s^2
float mass_cart = 1; // kg
float mass_pole = 0.1; // kg
float friction_cart_on_track = 0.0005; // coefficient of friction
float friction_pole_hinge = 0.000002; // coefficient of friction

float max_balance_time = 0;

float current_state[2];
bool in_bounds = true;

uint32_t time_increment;

//! How many ticks until next frame
static uint32_t tick_in_frame = 0;

//! The upper bits of the key value that model should transmit with
static uint32_t key;

//! the number of timer ticks that this model should run for before exiting.
uint32_t simulation_ticks = 0;
uint32_t score_change_count=0;

//----------------------------------------------------------------------------
// Inline functions
//----------------------------------------------------------------------------
static inline void spike_angle()
{
  spin1_send_mc_packet(key | (SPECIAL_EVENT_ANGLE), 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "Got a reward\n");
}

static inline void spike_angle_v()
{
  spin1_send_mc_packet(key | (SPECIAL_EVENT_ANGLE_V), 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "Got a reward\n");
}

static inline void spike_cart()
{
  spin1_send_mc_packet(key | (SPECIAL_EVENT_CART), 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "Got a reward\n");
}

static inline void spike_cart_v()
{
  spin1_send_mc_packet(key | (SPECIAL_EVENT_CART_V), 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "Got a reward\n");
}

void resume_callback() {
    recording_reset();
}

//void add_event(int i, int j, colour_t col, bool bricked)
//{
//  const uint32_t colour_bit = (col == COLOUR_BACKGROUND) ? 0 : 1;
//  const uint32_t spike_key = key | (SPECIAL_EVENT_MAX + (i << 10) + (j << 2) + (bricked<<1) + colour_bit);
//
//  spin1_send_mc_packet(spike_key, 0, NO_PAYLOAD);
//  io_printf(IO_BUF, "%d, %d, %u, %08x\n", i, j, col, spike_key);
//}

static bool initialize(uint32_t *timer_period)
{
    io_printf(IO_BUF, "Initialise bandit: started\n");

    // Get the address this core's DTCM data starts at from SRAM
    address_t address = data_specification_get_data_address();

    // Read the header
    if (!data_specification_read_header(address))
    {
      return false;
    }
    /*
    simulation_initialise(
        address_t address, uint32_t expected_app_magic_number,
        uint32_t* timer_period, uint32_t *simulation_ticks_pointer,
        uint32_t *infinite_run_pointer, int sdp_packet_callback_priority,
        int dma_transfer_done_callback_priority)
    */
    // Get the timing details and set up thse simulation interface
    if (!simulation_initialise(data_specification_get_region(REGION_SYSTEM, address),
    APPLICATION_NAME_HASH, timer_period, &simulation_ticks,
    &infinite_run, 1, NULL))
    {
      return false;
    }
    io_printf(IO_BUF, "simulation time = %u\n", simulation_ticks);


    // Read breakout region
    address_t breakout_region = data_specification_get_region(REGION_PENDULUM, address);
    key = breakout_region[0];
    io_printf(IO_BUF, "\tKey=%08x\n", key);
    io_printf(IO_BUF, "\tTimer period=%d\n", *timer_period);

    //get recording region
    address_t recording_address = data_specification_get_region(
                                       REGION_RECORDING,address);
    // Setup recording
    uint32_t recording_flags = 0;
    if (!recording_initialize(recording_address, &recording_flags))
    {
       rt_error(RTE_SWERR);
       return false;
    }

    cart_position = track_length / 2;

    address_t pend_region = data_specification_get_region(REGION_DATA, address);
//    encoding_scheme = pend_region[0]; // 0 rate
    encoding_scheme = pend_region[0];
    time_increment = pend_region[1];
    half_pole_length = pend_region[2] / 2;
    pole_angle = pend_region[3];
    reward_based = pend_region[4];
    force_increment = pend_region[5];
    max_firing_rate = pend_region[6];
    number_of_bins = pend_region[7];

    max_firing_prob = max_firing_rate / 1000;

    kiss_seed[0] = pend_region[8];
    kiss_seed[1] = pend_region[9];
    kiss_seed[2] = pend_region[10];
    kiss_seed[3] = pend_region[11];
    validate_mars_kiss64_seed(kiss_seed);

    force_increment = (float)((max_motor_force - min_motor_force) / (float)force_increment);
    
    //TODO check this prints right, ybug read the address
    io_printf(IO_BUF, "r1 %d\n", (uint32_t *)pend_region[0]);
    io_printf(IO_BUF, "r2 %d\n", (uint32_t *)pend_region[1]);
    io_printf(IO_BUF, "rand3. %d\n", (uint32_t *)pend_region[2]);
    io_printf(IO_BUF, "rand3 0x%x\n", (uint32_t *)pend_region[3]);
    io_printf(IO_BUF, "r4 0x%x\n", pend_region[3]);
    io_printf(IO_BUF, "r5 0x%x\n", pend_region);
    io_printf(IO_BUF, "r6 %u\n", pend_region[0]);
    io_printf(IO_BUF, "r6d %d\n", pend_region[0]);
    io_printf(IO_BUF, "r7 %u\n", pend_region[1]);
    io_printf(IO_BUF, "r7d %d\n", pend_region[1]);
    io_printf(IO_BUF, "r2 %u\n", pend_region[2]);
    io_printf(IO_BUF, "r2d %d\n", pend_region[2]);
    io_printf(IO_BUF, "r3 %u\n", pend_region[3]);
    io_printf(IO_BUF, "r3d %d\n", pend_region[3]);
    io_printf(IO_BUF, "r4 %u\n", pend_region[4]);
    io_printf(IO_BUF, "r4d %d\n", pend_region[4]);
    io_printf(IO_BUF, "r5 %u\n", pend_region[5]);
    io_printf(IO_BUF, "r5d %d\n", pend_region[5]);
    io_printf(IO_BUF, "r6 %u\n", pend_region[6]);
    io_printf(IO_BUF, "r6d %d\n", pend_region[6]);
    io_printf(IO_BUF, "r7 %u\n", pend_region[7]);
    io_printf(IO_BUF, "r7d %d\n", pend_region[7]);
    io_printf(IO_BUF, "re %d\n", reward_based);
//    io_printf(IO_BUF, "r6 0x%x\n", *pend_region);
//    io_printf(IO_BUF, "r6 0x%x\n", &pend_region);

    io_printf(IO_BUF, "Initialise: completed successfully\n");

//    auto start = chrono::steady_clock::now();
    return true;
}

// updates the current state of the pendulum
bool update_state(float time_step){
    float effective_force_pole_on_cart = 0;
    float pole_angle_force = (mass_pole * half_pole_length * pole_velocity * pole_velocity * sin(pole_angle));
    float angle_scalar = ((3.0f / 4.0f) * mass_pole * cos(pole_angle));
    float friction_and_gravity = (((friction_pole_hinge * pole_velocity) / (mass_pole * half_pole_length)) +
                        (gravity * sin(pole_angle)));
    float effective_pole_mass = mass_pole * (1.0f - ((3.0f / 4.0f) * cos(pole_angle) * cos(pole_angle)));

    effective_force_pole_on_cart = pole_angle_force + (angle_scalar * friction_and_gravity);
    if (cart_velocity > 0){
        cart_acceleration = (motor_force - friction_cart_on_track + effective_force_pole_on_cart) /
                                (mass_cart + effective_pole_mass);
    }
    else{
        cart_acceleration = (motor_force + friction_cart_on_track + effective_force_pole_on_cart) /
                                (mass_cart + effective_pole_mass);
    }

    float length_scalar = (-3.0f / 4.0f * half_pole_length);
    float cart_acceleration_effect = cart_acceleration * cos(pole_angle);
    float gravity_effect = gravity * sin(pole_angle);
    float friction_effect = (friction_pole_hinge * pole_velocity) / (mass_pole * half_pole_length);
    pole_acceleration = length_scalar * (cart_acceleration_effect + gravity_effect + friction_effect);

    cart_velocity = (cart_acceleration * time_step) + cart_velocity;
    cart_position = (cart_velocity * time_step) + cart_position;

    pole_velocity = (pole_acceleration * time_step) + pole_velocity;
    pole_angle = (pole_velocity * time_step) + pole_angle;

    io_printf(IO_BUF, "pole (d,v,a):(%d, %d, %d) and cart (d,v,a):(%d, %d, %d)\n", pole_angle, pole_velocity,
                        pole_acceleration, cart_position, pole_velocity, pole_acceleration);

    if (abs(cart_position) > (track_length / 2.0f) || abs(pole_angle) > max_pole_angle) {
        return false;
    }
    else{
        return true;
    }
}

void mc_packet_received_callback(uint key, uint payload)
{
    uint32_t compare;
    compare = key & 0x7;
//    io_printf(IO_BUF, "compare = %x\n", compare);
//    io_printf(IO_BUF, "key = %x\n", key);
//    io_printf(IO_BUF, "payload = %x\n", payload);
    use(payload);
    if(compare == BACKWARD_MOTOR){
        motor_force = motor_force - force_increment;
        if (motor_force < min_motor_force){
            motor_force = min_motor_force;
        }
    }
    else if(compare == FORWARD_MOTOR){
        motor_force = motor_force + force_increment;
        if (motor_force > max_motor_force){
            motor_force = max_motor_force;
        }
    }
}

void send_status(){
    if (encoding_scheme == 0){
        float relative_cart = 0;
        float relative_angle = 0;
        float relative_cart_velocity = 0;
        float relative_angular_velocity = 0;
        if (central){
            relative_angle = abs(pole_angle) / max_pole_angle;
            relative_angular_velocity = abs(pole_velocity) / highend_pole_v;
            relative_cart = abs(cart_position) / (track_length / 2);
            relative_cart_velocity = abs(cart_velocity) / (highend_cart_v);
        }
        else{
            relative_angle = (pole_angle - min_pole_angle) / (max_pole_angle - min_pole_angle);
            relative_angular_velocity = (pole_velocity + highend_pole_v) / (highend_pole_v * 2);
            relative_cart = cart_position / track_length;
            relative_cart_velocity = (cart_velocity + highend_cart_v) / (highend_cart_v * 2);
        }
        io_printf(IO_BUF, "relative angle, v %d, %d and cart, v %d, %d\n", relative_angle, relative_angular_velocity,
                            relative_cart, relative_cart_velocity);
        float angle_roll = 0;
        float angle_roll_v = 0;
        float cart_roll = 0;
        float cart_roll_v = 0;
        angle_roll = (float)(mars_kiss64_seed(kiss_seed) / 0xffffffff);
        angle_roll_v = (float)(mars_kiss64_seed(kiss_seed) / 0xffffffff);
        cart_roll = (float)(mars_kiss64_seed(kiss_seed) / 0xffffffff);
        cart_roll_v = (float)(mars_kiss64_seed(kiss_seed) / 0xffffffff);
        io_printf(IO_BUF, "roll angle, v %d, %d and cart, v %d, %d\n", angle_roll, angle_roll_v, cart_roll, cart_roll_v);
        if (angle_roll > max_firing_prob){
            spike_angle();
        }
        if (angle_roll_v > max_firing_prob){
            spike_angle_v();
        }
        if (cart_roll > max_firing_prob){
            spike_cart();
        }
        if (cart_roll_v > max_firing_prob){
            spike_cart_v();
        }
    }
    else{
        io_printf(IO_BUF, "some stuff with bins here\n");
    }
}

void timer_callback(uint unused, uint dummy)
{
    use(unused);
    use(dummy);

    _time++;
    score_change_count++;

    if (!infinite_run && _time >= simulation_ticks)
    {
        //spin1_pause();
        recording_finalise();
        // go into pause and resume state to avoid another tick
        simulation_handle_pause_resume(resume_callback);
        //    spin1_callback_off(MC_PACKET_RECEIVED);

        io_printf(IO_BUF, "infinite_run %d; time %d\n",infinite_run, _time);
        io_printf(IO_BUF, "simulation_ticks %d\n",simulation_ticks);
        //    io_printf(IO_BUF, "key count Left %u\n", left_key_count);
        //    io_printf(IO_BUF, "key count Right %u\n", right_key_count);

        io_printf(IO_BUF, "Exiting on timer.\n");
//        simulation_handle_pause_resume(NULL);
        simulation_ready_to_read();

        _time -= 1;
        return;
    }
    // Otherwise
    else
    {
        send_status();
        if (tick_in_frame == 0){
            update_state(0);
            // possibly use this to allow updating of time whenever
//            auto start = chrono::steady_clock::now();
        }
        // Increment ticks in frame counter and if this has reached frame delay
        tick_in_frame++;
        if(tick_in_frame == time_increment)
        {
            if (in_bounds){
                max_balance_time = (float)_time;
                in_bounds = update_state((float)time_increment / 1000.f);
            }
            // Reset ticks in frame and update frame
            tick_in_frame = 0;
//            update_frame();
            // Update recorded score every 1s
            if(score_change_count>=1000){
                current_state[0] = cart_position;
                current_state[1] = pole_angle;
                if(reward_based == 0){
                    recording_record(0, &current_state, 4);
                }
                else{
                    recording_record(0, &max_balance_time, 4);
                }
                score_change_count=0;
            }
        }
    }
//    io_printf(IO_BUF, "time %u\n", ticks);
//    io_printf(IO_BUF, "time %u\n", _time);
}
//-------------------------------------------------------------------------------

INT_HANDLER sark_int_han (void);


//-------------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Entry point
//----------------------------------------------------------------------------
void c_main(void)
{
  // Load DTCM data
  uint32_t timer_period;
  if (!initialize(&timer_period))
  {
    io_printf(IO_BUF,"Error in initialisation - exiting!\n");
    rt_error(RTE_SWERR);
    return;
  }

  tick_in_frame = 0;

  // Set timer tick (in microseconds)
  io_printf(IO_BUF, "setting timer tick callback for %d microseconds\n",
              timer_period);
  spin1_set_timer_tick(timer_period);

  io_printf(IO_BUF, "simulation_ticks %d\n",simulation_ticks);

  // Register callback
  spin1_callback_on(TIMER_TICK, timer_callback, 2);
  spin1_callback_on(MC_PACKET_RECEIVED, mc_packet_received_callback, -1);

  _time = UINT32_MAX;

  simulation_run();




}
