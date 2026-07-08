/**
 * @file homing.h
 * @brief Homing Sequence Manager for LEGO Sorter V2
 * 
 * Implements G28 homing sequence per SM-DES-007 §7.
 * 
 * Sequence:
 * 1. Move toward endstop at v_home_fast (20 mm/s)
 * 2. Back off 5 mm at v_home_fast
 * 3. Re-approach at v_home_slow (2 mm/s)
 * 4. Set position to 0
 * 
 * @version 1.0
 * @date 2026-03-10
 */

#ifndef HOMING_H
#define HOMING_H

#include <Arduino.h>
#include "config.h"

// Forward declarations
class StepperDriver;
class EndstopManager;
class MotionPlanner;

/**
 * @brief Homing phase
 */
enum class HomingPhase : uint8_t {
    IDLE,               // Not homing
    X_FAST_APPROACH,    // X fast approach to endstop
    X_BACKOFF,          // X back off from endstop
    X_SLOW_APPROACH,    // X slow approach to endstop
    X_DONE,             // X axis homed
    Y_FAST_APPROACH,    // Y fast approach to endstop
    Y_BACKOFF,          // Y back off from endstop
    Y_SLOW_APPROACH,    // Y slow approach to endstop
    COMPLETE,           // All axes homed
    ERROR               // Homing error (timeout, etc.)
};

/**
 * @brief Homing manager
 */
class HomingManager {
public:
    /**
     * @brief Get singleton instance
     */
    static HomingManager& getInstance();
    
    /**
     * @brief Initialize homing manager
     */
    void begin();
    
    /**
     * @brief Start homing sequence
     * @param homeX Home X axis
     * @param homeY Home Y axis
     * @return true if homing started
     */
    bool start(bool homeX = true, bool homeY = true);
    
    /**
     * @brief Update homing state machine (call every loop)
     * @return true if homing just completed
     */
    bool update();
    
    /**
     * @brief Check if homing is in progress
     */
    bool isHoming() const;
    
    /**
     * @brief Check if homing completed successfully
     */
    bool isComplete() const;
    
    /**
     * @brief Check if there was a homing error
     */
    bool hasError() const;
    
    /**
     * @brief Get current homing phase
     */
    HomingPhase getPhase() const;
    
    /**
     * @brief Cancel homing (E-stop)
     */
    void cancel();

private:
    HomingManager();
    ~HomingManager() = default;
    
    // Prevent copying
    HomingManager(const HomingManager&) = delete;
    HomingManager& operator=(const HomingManager&) = delete;
    
    /**
     * @brief Step X axis in specified direction
     * @param steps Number of steps
     * @param positive Direction (true = toward max)
     */
    void stepX(int32_t steps, bool positive);
    
    /**
     * @brief Step Y axis in specified direction
     * @param steps Number of steps
     * @param positive Direction (true = toward max)
     */
    void stepY(int32_t steps, bool positive);
    
    /**
     * @brief Start move toward home (negative direction)
     * @param axis 'X' or 'Y'
     * @param feedrate Feedrate in mm/min
     */
    void startHomeMove(char axis, float feedrate);
    
    /**
     * @brief Start backoff move (positive direction)
     * @param axis 'X' or 'Y'
     * @param distance Distance in mm
     */
    void startBackoffMove(char axis, float distance);
    
    /**
     * @brief Wait for move to complete (blocking during homing)
     */
    void waitMoveComplete();
    
    HomingPhase phase;
    bool homingActive;
    bool homeXRequested;
    bool homeYRequested;
    uint32_t homingStartMs;
    
    // GPIO masks for direct step control
    uint32_t xStepMask;
    uint32_t yStepMask;
    uint32_t xDirMask;
    uint32_t yDirMask;
};

#endif // HOMING_H