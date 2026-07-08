/**
 * @file stepper_driver.h
 * @brief Stepper Motor Driver for LEGO Sorter V2
 * 
 * Timer ISR-driven step generation for 2-axis XY gantry.
 * Uses hardware Timer0 for precise step timing.
 * 
 * Design: SM-DES-007 §6 (Step Generation)
 * 
 * @version 1.0
 * @date 2026-03-10
 */

#ifndef STEPPER_DRIVER_H
#define STEPPER_DRIVER_H

#include <Arduino.h>
#include "config.h"

/**
 * @brief Motion segment for ISR execution
 */
struct MotionSegment {
    volatile int32_t stepsX;        // Remaining steps for X axis
    volatile int32_t stepsY;        // Remaining steps for Y axis
    volatile int32_t totalSteps;    // Total steps in segment
    volatile float currentVelocity; // Current velocity (mm/s)
    volatile bool active;           // Segment is being executed
};

/**
 * @brief Stepper driver class - singleton for ISR access
 */
class StepperDriver {
public:
    /**
     * @brief Get singleton instance
     */
    static StepperDriver& getInstance();
    
    /**
     * @brief Initialize GPIO and timer
     */
    void begin();
    
    /**
     * @brief Enable steppers (active LOW)
     */
    void enable();
    
    /**
     * @brief Disable steppers (release motors)
     */
    void disable();
    
    /**
     * @brief Check if steppers are enabled
     */
    bool isEnabled() const;
    
    /**
     * @brief Start a coordinated XY move
     * @param dx X distance in mm (signed)
     * @param dy Y distance in mm (signed)
     * @param feedrate Feedrate in mm/min
     * @return true if move started successfully
     */
    bool startMove(float dx, float dy, float feedrate);
    
    /**
     * @brief Check if motion is in progress
     */
    bool isMoving() const;
    
    /**
     * @brief Stop motion immediately (E-stop)
     */
    void stop();
    
    /**
     * @brief Get current X position in mm
     */
    float getPositionX() const;
    
    /**
     * @brief Get current Y position in mm
     */
    float getPositionY() const;
    
    /**
     * @brief Reset position to origin (after homing)
     */
    void resetPosition();
    
    /**
     * @brief Timer ISR - called from hardware timer
     * Note: This is public for the ISR callback, do not call directly
     */
    void handleISR();
    /**
     * @brief Check if last move completed (for !MOVE_DONE notification)
     */
    bool isMoveComplete() const;
    
    /**
     * @brief Clear move complete flag
     */
    void clearMoveComplete();

private:
    StepperDriver();
    ~StepperDriver() = default;
    
    // Prevent copying
    StepperDriver(const StepperDriver&) = delete;
    StepperDriver& operator=(const StepperDriver&) = delete;
    
    /**
     * @brief Step the X axis
     * @param direction Direction (true = positive)
     */
    void stepX(bool direction);
    
    /**
     * @brief Step the Y axis
     * @param direction Direction (true = positive)
     */
    void stepY(bool direction);
    
    /**
     * @brief Configure hardware timer
     */
    void setupTimer();
    
    /**
     * @brief Set next step interval from velocity
     */
    void setStepInterval(float velocity_mm_s);
    
    // Member variables
    volatile bool enabled;
    volatile bool moving;
    volatile bool moveComplete;
    
    // Position tracking (mm)
    volatile float posX;
    volatile float posY;
    
    // Current motion segment
    MotionSegment segment;
    
    // Bresenham error accumulator
    volatile float bresenhamError;
    volatile float bresenhamRatio;   // minor_steps / major_steps
    
    // Direction flags
    volatile bool dirX;
    volatile bool dirY;
    
    // Which axis is major (more steps)
    volatile bool xIsMajor;
    
    // Hardware timer
    hw_timer_t* timer;
    volatile bool timerRunning;
    
    // Step pulse state
    volatile bool stepPinHigh;
    volatile uint32_t stepPinMaskX;
    volatile uint32_t stepPinMaskY;
};

// Global ISR function for timer callback
void IRAM_ATTR stepperTimerISR();

#endif // STEPPER_DRIVER_H