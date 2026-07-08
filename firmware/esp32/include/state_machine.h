/**
 * @file state_machine.h
 * @brief ESP32 State Machine for LEGO Sorter V2
 * 
 * Implements the state machine defined in SM-DES-007 §3 and SM-DES-003 §6.
 * 
 * States:
 *   BOOT → IDLE_NOT_HOMED → READY ↔ MOVING → (via M112) → ESTOPPED
 * 
 * @version 1.0
 * @date 2026-03-10
 */

#ifndef STATE_MACHINE_H
#define STATE_MACHINE_H

#include <Arduino.h>

/**
 * @brief System states for ESP32 motion controller
 */
enum class State : uint8_t {
    BOOT,               // Initial state during setup()
    IDLE_NOT_HOMED,     // System ready but not homed
    READY,              // System homed and idle
    MOVING,             // Gantry in motion (G1 executing)
    ESTOPPED            // Emergency stop active
};

/**
 * @brief Command types that can be received via UART
 */
enum class CommandType : uint8_t {
    NONE = 0,
    // Motion commands
    G1,                 // Linear move
    G28,                // Home axes
    G90,                // Set absolute mode
    G91,                // Set relative mode
    // Actuator commands
    M3,                 // Open gate
    M5,                 // Close gate
    M17,                // Enable steppers
    M18,                // Disable steppers
    // Control commands
    M112,               // Emergency stop
    M114,               // Report position
    M119,               // Report endstop status
    M400,               // Wait for queue empty
    M999                // Reset from ESTOPPED
};

/**
 * @brief Class managing ESP32 state machine
 */
class StateMachine {
public:
    StateMachine();
    
    /**
     * @brief Get current state
     */
    State getState() const;
    
    /**
     * @brief Transition to a new state
     * @param newState Target state
     * @return true if transition was valid
     */
    bool transitionTo(State newState);
    
    /**
     * @brief Check if a command can be accepted in current state
     * @param cmd Command type to check
     * @return true if command is allowed
     */
    bool canAccept(CommandType cmd) const;
    
    /**
     * @brief Get state name as string (for debugging/logging)
     */
    const char* getStateName() const;
    
    /**
     * @brief Check if system is homed
     */
    bool isHomed() const;
    
    /**
     * @brief Set homed flag (called after successful G28)
     */
    void setHomed(bool homed);
    
    /**
     * @brief Check if E-stop is active
     */
    bool isEstopActive() const;
    
    /**
     * @brief Trigger E-stop from external source (endstop hit, watchdog)
     */
    void triggerEstop();

private:
    State currentState;
    bool homed;
    
    /**
     * @brief Check if command is valid in IDLE_NOT_HOMED state
     */
    bool canAcceptInIdleNotHomed(CommandType cmd) const;
    
    /**
     * @brief Check if command is valid in READY state
     */
    bool canAcceptInReady(CommandType cmd) const;
    
    /**
     * @brief Check if command is valid in MOVING state
     */
    bool canAcceptInMoving(CommandType cmd) const;
    
    /**
     * @brief Check if command is valid in ESTOPPED state
     */
    bool canAcceptInEstopped(CommandType cmd) const;
};

#endif // STATE_MACHINE_H