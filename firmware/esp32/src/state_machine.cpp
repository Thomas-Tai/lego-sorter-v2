/**
 * @file state_machine.cpp
 * @brief ESP32 State Machine Implementation
 * 
 * State transitions per SM-DES-007 §3:
 *   BOOT → IDLE_NOT_HOMED (setup complete)
 *   IDLE_NOT_HOMED → READY (G28 complete)
 *   READY → MOVING (G1 accepted)
 *   MOVING → READY (move complete)
 *   ANY → ESTOPPED (M112 or endstop during move)
 *   ESTOPPED → IDLE_NOT_HOMED (M999)
 */

#include "state_machine.h"

StateMachine::StateMachine() 
    : currentState(State::BOOT)
    , homed(false) {
}

State StateMachine::getState() const {
    return currentState;
}

bool StateMachine::transitionTo(State newState) {
    // Validate state transitions
    switch (currentState) {
        case State::BOOT:
            // BOOT can only go to IDLE_NOT_HOMED
            if (newState != State::IDLE_NOT_HOMED) {
                return false;
            }
            break;
            
        case State::IDLE_NOT_HOMED:
            // IDLE_NOT_HOMED can go to READY or ESTOPPED
            if (newState != State::READY && newState != State::ESTOPPED) {
                return false;
            }
            break;
            
        case State::READY:
            // READY can go to MOVING or ESTOPPED
            if (newState != State::MOVING && newState != State::ESTOPPED) {
                return false;
            }
            break;
            
        case State::MOVING:
            // MOVING can go to READY or ESTOPPED
            if (newState != State::READY && newState != State::ESTOPPED) {
                return false;
            }
            break;
            
        case State::ESTOPPED:
            // ESTOPPED can only go to IDLE_NOT_HOMED (via M999)
            if (newState != State::IDLE_NOT_HOMED) {
                return false;
            }
            break;
    }
    
    currentState = newState;
    
    // Update homed flag on transition
    if (newState == State::IDLE_NOT_HOMED) {
        homed = false;
    } else if (newState == State::READY) {
        homed = true;
    }
    
    return true;
}

bool StateMachine::canAccept(CommandType cmd) const {
    switch (currentState) {
        case State::BOOT:
            // No commands accepted during BOOT
            return false;
            
        case State::IDLE_NOT_HOMED:
            return canAcceptInIdleNotHomed(cmd);
            
        case State::READY:
            return canAcceptInReady(cmd);
            
        case State::MOVING:
            return canAcceptInMoving(cmd);
            
        case State::ESTOPPED:
            return canAcceptInEstopped(cmd);
    }
    return false;
}

const char* StateMachine::getStateName() const {
    switch (currentState) {
        case State::BOOT:           return "BOOT";
        case State::IDLE_NOT_HOMED: return "IDLE_NOT_HOMED";
        case State::READY:          return "READY";
        case State::MOVING:         return "MOVING";
        case State::ESTOPPED:       return "ESTOPPED";
    }
    return "UNKNOWN";
}

bool StateMachine::isHomed() const {
    return homed;
}

void StateMachine::setHomed(bool h) {
    homed = h;
}

bool StateMachine::isEstopActive() const {
    return currentState == State::ESTOPPED;
}

void StateMachine::triggerEstop() {
    currentState = State::ESTOPPED;
}

bool StateMachine::canAcceptInIdleNotHomed(CommandType cmd) const {
    // Per SM-DES-003 §6: G28, M17, M18, M112, M114, M119
    // M999 only accepted in ESTOPPED state
    switch (cmd) {
        case CommandType::G28:
        case CommandType::M17:
        case CommandType::M18:
        case CommandType::M112:
        case CommandType::M114:
        case CommandType::M119:
            return true;
        default:
            return false;
    }
}

bool StateMachine::canAcceptInReady(CommandType cmd) const {
    // Per SM-DES-003 §6: All commands accepted
    // G1, G28, G90, G91, M3, M5, M17, M18, M112, M114, M119, M400, M999
    switch (cmd) {
        case CommandType::G1:
        case CommandType::G28:
        case CommandType::G90:
        case CommandType::G91:
        case CommandType::M3:
        case CommandType::M5:
        case CommandType::M17:
        case CommandType::M18:
        case CommandType::M112:
        case CommandType::M114:
        case CommandType::M119:
        case CommandType::M400:
        case CommandType::M999:
            return true;
        default:
            return false;
    }
}

bool StateMachine::canAcceptInMoving(CommandType cmd) const {
    // Per SM-DES-003 §6: M3, M5, M17, M18, M112, M114, M119, M400 (immediate)
    // G1 can be queued
    switch (cmd) {
        case CommandType::G1:       // Queued
        case CommandType::M3:
        case CommandType::M5:
        case CommandType::M17:
        case CommandType::M18:
        case CommandType::M112:
        case CommandType::M114:
        case CommandType::M119:
        case CommandType::M400:
            return true;
        default:
            return false;
    }
}

bool StateMachine::canAcceptInEstopped(CommandType cmd) const {
    // Per SM-DES-003 §6: M114, M999 only
    switch (cmd) {
        case CommandType::M114:
        case CommandType::M999:
            return true;
        default:
            return false;
    }
}