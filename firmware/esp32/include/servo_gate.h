/**
 * @file servo_gate.h
 * @brief Servo Gate Control for LEGO Sorter V2
 * 
 * PWM control for SG90 servo to open/close the sorting gate.
 * Per SM-DES-007 §10 and SM-DES-005 §6.3.
 * 
 * PWM: 50 Hz (20 ms period)
 * Open:  2500 µs pulse (~180°)
 * Close: 500 µs pulse (~0°)
 * Hold:  300 ms after open/close
 * 
 * @version 1.0
 * @date 2026-03-10
 */

#ifndef SERVO_GATE_H
#define SERVO_GATE_H

#include <Arduino.h>
#include "config.h"

/**
 * @brief Gate state
 */
enum class GateState : uint8_t {
    CLOSED,         // Gate is closed
    OPENING,        // Gate is opening
    OPEN,           // Gate is open
    CLOSING,        // Gate is closing
    HOLDING         // Holding position (waiting for GATE_HOLD_MS)
};

/**
 * @brief Servo gate controller
 */
class ServoGate {
public:
    /**
     * @brief Get singleton instance
     */
    static ServoGate& getInstance();
    
    /**
     * @brief Initialize servo PWM
     */
    void begin();
    
    /**
     * @brief Open the gate (non-blocking)
     * Sends "ok" immediately, "!GATE_DONE" after hold time
     */
    void open();
    
    /**
     * @brief Close the gate (non-blocking)
     * Sends "ok" immediately, "!GATE_DONE" after hold time
     */
    void close();
    
    /**
     * @brief Update gate state (call every loop)
     * Handles timing for hold and notification
     * @return true if gate operation just completed (for !GATE_DONE)
     */
    bool update();
    
    /**
     * @brief Check if gate is busy (opening/closing/holding)
     */
    bool isBusy() const;
    
    /**
     * @brief Check if gate is open
     */
    bool isOpen() const;
    
    /**
     * @brief Get current gate state
     */
    GateState getState() const;
    
    /**
     * @brief Set PWM directly (for calibration)
     * @param pulseUs Pulse width in microseconds
     */
    void setPulse(uint16_t pulseUs);

private:
    ServoGate();
    ~ServoGate() = default;
    
    // Prevent copying
    ServoGate(const ServoGate&) = delete;
    ServoGate& operator=(const ServoGate&) = delete;
    
    /**
     * @brief Set servo PWM duty cycle
     * @param pulseUs Pulse width in microseconds
     */
    void setServoPwm(uint16_t pulseUs);
    
    GateState state;
    uint32_t holdStartMs;
    bool busy;
    bool gateOpen;
    
    // LEDC channel
    uint8_t pwmChannel;
    uint16_t currentPulse;
};

#endif // SERVO_GATE_H