/**
 * @file endstop.cpp
 * @brief Endstop Management Implementation
 * 
 * Implements debounced endstop reading per SM-DES-007 §8.
 * 
 * Endstop wiring (SM-DES-005 §6.2):
 *   - GPIO 34 (X_ENDSTOP) and GPIO 35 (Y_ENDSTOP) are input-only
 *   - External 10kΩ pull-up to 3.3V
 *   - Switch connects to GND when triggered
 *   - LOW = triggered, HIGH = open
 */

#include "endstop.h"

EndstopManager& EndstopManager::getInstance() {
    static EndstopManager manager;
    return manager;
}

EndstopManager::EndstopManager()
    : xTriggeredState(false)
    , yTriggeredState(false)
    , xLastRaw(HIGH)
    , yLastRaw(HIGH)
    , xDebounceStart(0)
    , yDebounceStart(0)
    , triggeredFlag(false)
    , xPrevState(false)
    , yPrevState(false)
    , lastChangeMs(0)
{
}

void EndstopManager::begin() {
    // Configure pins as input (no pull-up - external pull-ups required)
    pinMode(PIN_X_ENDSTOP, INPUT);
    pinMode(PIN_Y_ENDSTOP, INPUT);
    
    // Initialize states from current readings
    xLastRaw = digitalRead(PIN_X_ENDSTOP);
    yLastRaw = digitalRead(PIN_Y_ENDSTOP);
    
    // Endstop triggered = LOW
    xTriggeredState = (xLastRaw == ENDSTOP_TRIGGERED);
    yTriggeredState = (yLastRaw == ENDSTOP_TRIGGERED);
    
    xPrevState = xTriggeredState;
    yPrevState = yTriggeredState;
    
    xDebounceStart = millis();
    yDebounceStart = millis();
}

void EndstopManager::update() {
    uint32_t now = millis();
    
    // Read current raw states
    bool xRaw = digitalRead(PIN_X_ENDSTOP);
    bool yRaw = digitalRead(PIN_Y_ENDSTOP);
    
    // Debounce X endstop
    bool xNewState = debounce(PIN_X_ENDSTOP, xTriggeredState, xLastRaw, xDebounceStart);
    
    // Debounce Y endstop
    bool yNewState = debounce(PIN_Y_ENDSTOP, yTriggeredState, yLastRaw, yDebounceStart);
    
    // Check for state changes (edge detection)
    if (xNewState != xPrevState || yNewState != yPrevState) {
        lastChangeMs = now;
        
        // Detect trigger edge (not triggered → triggered)
        if (!xPrevState && xNewState) {
            triggeredFlag = true;
        }
        if (!yPrevState && yNewState) {
            triggeredFlag = true;
        }
    }
    
    xPrevState = xNewState;
    yPrevState = yNewState;
    xTriggeredState = xNewState;
    yTriggeredState = yNewState;
}

bool EndstopManager::debounce(int pin, bool currentState, bool& lastRaw, uint32_t& debounceStart) {
    bool raw = digitalRead(pin);
    uint32_t now = millis();
    
    // Check if raw reading changed
    if (raw != lastRaw) {
        debounceStart = now;
        lastRaw = raw;
    }
    
    // Check if debounce period has elapsed
    if ((now - debounceStart) >= ENDSTOP_DEBOUNCE_MS) {
        // Update stable state
        // Endstop triggered = LOW (ENDSTOP_TRIGGERED)
        return (raw == ENDSTOP_TRIGGERED);
    }
    
    // Return current stable state during debounce
    return currentState;
}

bool EndstopManager::isXTriggered() const {
    return xTriggeredState;
}

bool EndstopManager::isYTriggered() const {
    return yTriggeredState;
}

bool EndstopManager::getRawX() const {
    return digitalRead(PIN_X_ENDSTOP) == ENDSTOP_TRIGGERED;
}

bool EndstopManager::getRawY() const {
    return digitalRead(PIN_Y_ENDSTOP) == ENDSTOP_TRIGGERED;
}

bool EndstopManager::wasTriggered() const {
    return triggeredFlag;
}

void EndstopManager::clearTriggered() {
    triggeredFlag = false;
}

EndstopState EndstopManager::getState() const {
    EndstopState state;
    state.xTriggered = xTriggeredState;
    state.yTriggered = yTriggeredState;
    state.xChanged = (xTriggeredState != xPrevState);
    state.yChanged = (yTriggeredState != yPrevState);
    state.lastChangeMs = lastChangeMs;
    return state;
}