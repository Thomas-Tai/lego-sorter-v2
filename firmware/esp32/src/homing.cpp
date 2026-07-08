/**
 * @file homing.cpp
 * @brief Homing Sequence Implementation
 * 
 * Implements G28 homing per SM-DES-007 §7.
 * Note: Homing is BLOCKING - only runs from IDLE_NOT_HOMED or READY state.
 */

#include "homing.h"
#include "stepper_driver.h"
#include "endstop.h"
#include "motion_planner.h"

HomingManager& HomingManager::getInstance() {
    static HomingManager manager;
    return manager;
}

HomingManager::HomingManager()
    : phase(HomingPhase::IDLE)
    , homingActive(false)
    , homeXRequested(false)
    , homeYRequested(false)
    , homingStartMs(0)
    , xStepMask(1 << PIN_X_STEP)
    , yStepMask(1 << PIN_Y_STEP)
    , xDirMask(1 << PIN_X_DIR)
    , yDirMask(1 << PIN_Y_DIR)
{
}

void HomingManager::begin() {
    phase = HomingPhase::IDLE;
    homingActive = false;
}

bool HomingManager::start(bool homeX, bool homeY) {
    if (homingActive) {
        return false;  // Already homing
    }
    
    homeXRequested = homeX;
    homeYRequested = homeY;
    homingActive = true;
    homingStartMs = millis();
    phase = HomingPhase::IDLE;
    
    // Enable steppers
    StepperDriver::getInstance().enable();
    
    // Determine starting phase
    if (homeXRequested) {
        phase = HomingPhase::X_FAST_APPROACH;
    } else if (homeYRequested) {
        phase = HomingPhase::Y_FAST_APPROACH;
    } else {
        // Nothing to home
        phase = HomingPhase::COMPLETE;
        homingActive = false;
        return true;
    }
    
    return true;
}

bool HomingManager::update() {
    if (!homingActive || phase == HomingPhase::IDLE) {
        return false;
    }
    
    // Check timeout
    if (millis() - homingStartMs > HOMING_TIMEOUT_MS) {
        phase = HomingPhase::ERROR;
        homingActive = false;
        return false;
    }
    
    EndstopManager& endstops = EndstopManager::getInstance();
    StepperDriver& stepper = StepperDriver::getInstance();
    
    switch (phase) {
        case HomingPhase::X_FAST_APPROACH: {
            // Check if endstop already triggered
            if (endstops.isXTriggered()) {
                // Back off first
                phase = HomingPhase::X_BACKOFF;
                startBackoffMove('X', HOMING_BACKOFF_MM);
                return false;
            }
            
            // Start fast approach
            startHomeMove('X', HOMING_FAST_MM_S * 60.0f);
            
            // Wait for endstop trigger
            while (stepper.isMoving()) {
                endstops.update();
                if (endstops.isXTriggered()) {
                    stepper.stop();
                    break;
                }
                delay(1);
            }
            
            // Move to backoff phase
            phase = HomingPhase::X_BACKOFF;
            startBackoffMove('X', HOMING_BACKOFF_MM);
            return false;
        }
        
        case HomingPhase::X_BACKOFF: {
            // Wait for backoff to complete
            if (!stepper.isMoving()) {
                delay(100);  // Brief pause
                phase = HomingPhase::X_SLOW_APPROACH;
                startHomeMove('X', HOMING_SLOW_MM_S * 60.0f);
            }
            return false;
        }
        
        case HomingPhase::X_SLOW_APPROACH: {
            // Wait for endstop trigger at slow speed
            while (stepper.isMoving()) {
                endstops.update();
                if (endstops.isXTriggered()) {
                    stepper.stop();
                    break;
                }
                delay(1);
            }
            
            // X is homed
            phase = HomingPhase::X_DONE;
            
            // Check if Y should be homed
            if (homeYRequested) {
                phase = HomingPhase::Y_FAST_APPROACH;
            } else {
                phase = HomingPhase::COMPLETE;
                homingActive = false;
                stepper.resetPosition();
                return true;
            }
            return false;
        }
        
        case HomingPhase::X_DONE:
            // Transition to Y if needed
            if (homeYRequested) {
                phase = HomingPhase::Y_FAST_APPROACH;
            } else {
                phase = HomingPhase::COMPLETE;
                homingActive = false;
                stepper.resetPosition();
                return true;
            }
            return false;
        
        case HomingPhase::Y_FAST_APPROACH: {
            // Check if endstop already triggered
            if (endstops.isYTriggered()) {
                phase = HomingPhase::Y_BACKOFF;
                startBackoffMove('Y', HOMING_BACKOFF_MM);
                return false;
            }
            
            // Start fast approach
            startHomeMove('Y', HOMING_FAST_MM_S * 60.0f);
            
            // Wait for endstop trigger
            while (stepper.isMoving()) {
                endstops.update();
                if (endstops.isYTriggered()) {
                    stepper.stop();
                    break;
                }
                delay(1);
            }
            
            phase = HomingPhase::Y_BACKOFF;
            startBackoffMove('Y', HOMING_BACKOFF_MM);
            return false;
        }
        
        case HomingPhase::Y_BACKOFF: {
            if (!stepper.isMoving()) {
                delay(100);
                phase = HomingPhase::Y_SLOW_APPROACH;
                startHomeMove('Y', HOMING_SLOW_MM_S * 60.0f);
            }
            return false;
        }
        
        case HomingPhase::Y_SLOW_APPROACH: {
            while (stepper.isMoving()) {
                endstops.update();
                if (endstops.isYTriggered()) {
                    stepper.stop();
                    break;
                }
                delay(1);
            }
            
            // Homing complete
            phase = HomingPhase::COMPLETE;
            homingActive = false;
            stepper.resetPosition();
            return true;
        }
        
        case HomingPhase::COMPLETE:
            homingActive = false;
            return true;
            
        case HomingPhase::ERROR:
            homingActive = false;
            return false;
            
        default:
            return false;
    }
}

bool HomingManager::isHoming() const {
    return homingActive;
}

bool HomingManager::isComplete() const {
    return phase == HomingPhase::COMPLETE;
}

bool HomingManager::hasError() const {
    return phase == HomingPhase::ERROR;
}

HomingPhase HomingManager::getPhase() const {
    return phase;
}

void HomingManager::cancel() {
    StepperDriver::getInstance().stop();
    homingActive = false;
    phase = HomingPhase::ERROR;
}

void HomingManager::stepX(int32_t steps, bool positive) {
    StepperDriver& stepper = StepperDriver::getInstance();
    digitalWrite(PIN_X_DIR, positive ? HIGH : LOW);
    
    for (int32_t i = 0; i < steps; i++) {
        digitalWrite(PIN_X_STEP, HIGH);
        delayMicroseconds(2);
        digitalWrite(PIN_X_STEP, LOW);
        delayMicroseconds(500);  // ~2000 steps/s = 25 mm/s
    }
}

void HomingManager::stepY(int32_t steps, bool positive) {
    digitalWrite(PIN_Y_DIR, positive ? HIGH : LOW);
    
    for (int32_t i = 0; i < steps; i++) {
        digitalWrite(PIN_Y_STEP, HIGH);
        delayMicroseconds(2);
        digitalWrite(PIN_Y_STEP, LOW);
        delayMicroseconds(500);
    }
}

void HomingManager::startHomeMove(char axis, float feedrate) {
    // Home is in negative direction (toward 0)
    StepperDriver& stepper = StepperDriver::getInstance();
    
    if (axis == 'X') {
        stepper.startMove(-X_MAX_MM, 0, feedrate);
    } else {
        stepper.startMove(0, -Y_MAX_MM, feedrate);
    }
}

void HomingManager::startBackoffMove(char axis, float distance) {
    // Backoff is in positive direction (away from endstop)
    StepperDriver& stepper = StepperDriver::getInstance();
    
    if (axis == 'X') {
        stepper.startMove(distance, 0, HOMING_FAST_MM_S * 60.0f);
    } else {
        stepper.startMove(0, distance, HOMING_FAST_MM_S * 60.0f);
    }
    
    // Wait for backoff to complete
    while (stepper.isMoving()) {
        delay(1);
    }
}

void HomingManager::waitMoveComplete() {
    while (StepperDriver::getInstance().isMoving()) {
        delay(1);
    }
}