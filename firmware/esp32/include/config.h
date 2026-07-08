/**
 * @file config.h
 * @brief ESP32 Firmware Configuration for LEGO Sorter V2
 * 
 * Hardware constants and motion parameters derived from:
 * - SM-DES-005 Electrical Design (pin assignments)
 * - SM-DES-007 Motion Control Design (motion parameters)
 * - SM-DES-003 UART Protocol Spec (communication)
 * 
 * @version 1.0
 * @date 2026-03-10
 */

#ifndef CONFIG_H
#define CONFIG_H

// ============================================================================
// UART CONFIGURATION (SM-DES-003 §2)
// ============================================================================
#define UART_BAUD           115200      // Baud rate for Pi communication
#define UART_MAX_LINE       96          // Max line length in bytes

// ============================================================================
// PIN ASSIGNMENTS (SM-DES-005 §2)
// ============================================================================
// X-Axis Stepper
#define PIN_X_STEP          26          // GPIO 26 - X STEP output
#define PIN_X_DIR           27          // GPIO 27 - X DIR output
#define PIN_X_ENDSTOP       34          // GPIO 34 - X Endstop (input-only, NO pull-up)

// Y-Axis Stepper
#define PIN_Y_STEP          14          // GPIO 14 - Y STEP output
#define PIN_Y_DIR           12          // GPIO 12 - Y DIR output
#define PIN_Y_ENDSTOP       35          // GPIO 35 - Y Endstop (input-only, NO pull-up)

// Shared Control
#define PIN_ENABLE          25          // GPIO 25 - Stepper enable (active LOW)

// Servo Gate
#define PIN_SERVO           13          // GPIO 13 - Servo PWM signal

// Endstop logic: LOW = triggered (external pull-up to 3.3V, switch to GND)
#define ENDSTOP_TRIGGERED   LOW
#define ENDSTOP_OPEN        HIGH

// ============================================================================
// MOTION PARAMETERS (SM-DES-007 §4.4)
// ============================================================================
#define STEPS_PER_MM        80          // NEMA17, 1/16 µstep, GT2-20T pulley

// Velocity & Acceleration Limits
#define V_MAX_MM_S          50.0f       // Max velocity: 50 mm/s = 3000 mm/min
#define A_MAX_MM_S2         500.0f      // Max acceleration: 500 mm/s²
#define J_MAX_MM_S3         2000.0f     // Max jerk: 2000 mm/s³

// Derived: max step frequency
#define STEP_FREQ_MAX       (V_MAX_MM_S * STEPS_PER_MM)  // 4000 steps/s

// ============================================================================
// HOMING PARAMETERS (SM-DES-007 §7)
// ============================================================================
#define HOMING_FAST_MM_S    20.0f       // Fast approach speed: 20 mm/s
#define HOMING_SLOW_MM_S    2.0f        // Slow re-approach speed: 2 mm/s
#define HOMING_BACKOFF_MM   5.0f        // Back-off distance after trigger: 5 mm
#define HOMING_TIMEOUT_MS   30000       // Max time for homing: 30 seconds

// ============================================================================
// GANTRY TRAVEL LIMITS (SM-DES-004 Mechanical Design)
// ============================================================================
#define X_MAX_MM            355.0f      // X-axis travel: 355 mm
#define Y_MAX_MM            160.0f      // Y-axis travel: 160 mm

// ============================================================================
// ENDSTOP DEBOUNCE (SM-DES-007 §8.1)
// ============================================================================
#define ENDSTOP_DEBOUNCE_MS 5           // Debounce time: 5 ms

// ============================================================================
// SERVO GATE PARAMETERS (SM-DES-007 §10)
// ============================================================================
#define SERVO_FREQ_HZ       50          // Servo PWM frequency: 50 Hz
#define SERVO_CLOSE_US      500         // Gate closed position: ~0°
#define SERVO_OPEN_US       2500        // Gate open position: ~180°
#define GATE_HOLD_MS        300         // Hold time for part drop: 300 ms

// ============================================================================
// COMMAND QUEUE (SM-DES-003 §7)
// ============================================================================
#define CMD_QUEUE_SIZE      4           // 4-slot circular buffer for G1 commands

// ============================================================================
// WATCHDOG TIMER (SM-DES-007 §9.2)
// ============================================================================
#define WATCHDOG_TIMEOUT_MS 5000        // Watchdog timeout: 5 seconds

// ============================================================================
// S-CURVE MOTION PROFILE (SM-DES-007 §4.3)
// ============================================================================
// Minimum distance for 3-segment S-curve: d_min = v_max² / j_max
#define S_CURVE_D_MIN_MM    1.25f       // 50² / 2000 = 1.25 mm

// ============================================================================
// ERROR CODES (SM-DES-003 §4)
// ============================================================================
enum ErrorCode {
    ERR_UNKNOWN_CMD      = 1,           // Unknown G/M code
    ERR_INVALID_PARAM    = 2,           // Missing or out-of-range parameter
    ERR_OUT_OF_BOUNDS    = 3,           // Target position exceeds limits
    ERR_NOT_HOMED        = 4,           // Movement before homing
    ERR_ESTOP_ACTIVE     = 5,           // Emergency stop active (SM-DES-003 §4)
    ERR_BUSY             = 6            // Command buffer full, retry later (SM-DES-003 §4)
};

// ============================================================================
// NOTIFICATION STRINGS (SM-DES-003 §5)
// ============================================================================
#define NOTIFY_READY        "!READY"
#define NOTIFY_HOMED        "!HOMED"
#define NOTIFY_MOVE_DONE    "!MOVE_DONE"
#define NOTIFY_GATE_DONE    "!GATE_DONE"
#define NOTIFY_ESTOP        "!ESTOP"

// ============================================================================
// RESPONSE STRINGS (SM-DES-003 §4)
// ============================================================================
#define RESP_OK             "ok"
#define RESP_OK_BUSY        "ok BUSY"
#define RESP_ERROR_PREFIX   "error:"

#endif // CONFIG_H