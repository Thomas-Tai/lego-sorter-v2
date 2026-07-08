/**
 * @file endstop.h
 * @brief Endstop Management for LEGO Sorter V2
 * 
 * Debounced endstop reading per SM-DES-007 §8.
 * Uses polling (not interrupts) to avoid bounce issues.
 * 
 * Note: GPIO 34 and 35 are input-only with no internal pull-ups.
 * External 10kΩ pull-ups to 3.3V are required.
 * Endstop triggered = LOW (switch connects to GND)
 * 
 * @version 1.0
 * @date 2026-03-10
 */

#ifndef ENDSTOP_H
#define ENDSTOP_H

#include <Arduino.h>
#include "config.h"

/**
 * @brief Endstop state with debounce
 */
struct EndstopState {
    bool xTriggered;        // X endstop is currently triggered
    bool yTriggered;        // Y endstop is currently triggered
    bool xChanged;          // X endstop state changed this cycle
    bool yChanged;          // Y endstop state changed this cycle
    uint32_t lastChangeMs;  // Time of last state change
};

/**
 * @brief Endstop manager class
 */
class EndstopManager {
public:
    /**
     * @brief Get singleton instance
     */
    static EndstopManager& getInstance();
    
    /**
     * @brief Initialize endstop pins
     */
    void begin();
    
    /**
     * @brief Poll endstops and update debounced state
     * Call this every loop iteration
     */
    void update();
    
    /**
     * @brief Check if X endstop is triggered (debounced)
     */
    bool isXTriggered() const;
    
    /**
     * @brief Check if Y endstop is triggered (debounced)
     */
    bool isYTriggered() const;
    
    /**
     * @brief Get raw (un-debounced) X endstop state
     */
    bool getRawX() const;
    
    /**
     * @brief Get raw (un-debounced) Y endstop state
     */
    bool getRawY() const;
    
    /**
     * @brief Check if either endstop was just triggered (edge detect)
     */
    bool wasTriggered() const;
    
    /**
     * @brief Clear the triggered flag
     */
    void clearTriggered();
    
    /**
     * @brief Get full endstop state
     */
    EndstopState getState() const;

private:
    EndstopManager();
    ~EndstopManager() = default;
    
    // Prevent copying
    EndstopManager(const EndstopManager&) = delete;
    EndstopManager& operator=(const EndstopManager&) = delete;
    
    /**
     * @brief Debounce a single endstop
     * @param pin GPIO pin number
     * @param currentState Current stable state
     * @param lastRaw Last raw reading
     * @param debounceStart Time when debounce started
     * @return New stable state
     */
    bool debounce(int pin, bool currentState, bool& lastRaw, uint32_t& debounceStart);
    
    // Debounced states
    bool xTriggeredState;
    bool yTriggeredState;
    
    // Raw states for debounce
    bool xLastRaw;
    bool yLastRaw;
    
    // Debounce timers
    uint32_t xDebounceStart;
    uint32_t yDebounceStart;
    
    // Edge detection
    bool triggeredFlag;
    bool xPrevState;
    bool yPrevState;
    
    // Last change timestamp
    uint32_t lastChangeMs;
};

#endif // ENDSTOP_H