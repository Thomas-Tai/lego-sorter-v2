/**
 * @file motion_planner.h
 * @brief Motion Planner for LEGO Sorter V2
 * 
 * Coordinates XY motion with S-curve profiles and Bresenham line algorithm.
 * Per SM-DES-007 §5 (XY Coordinated Motion).
 * 
 * @version 1.0
 * @date 2026-03-10
 */

#ifndef MOTION_PLANNER_H
#define MOTION_PLANNER_H

#include <Arduino.h>
#include "config.h"
#include "scurve.h"
#include "stepper_driver.h"

/**
 * @brief Planned motion segment
 */
struct PlannedMove {
    float startX;           // Start X position (mm)
    float startY;           // Start Y position (mm)
    float endX;             // Target X position (mm)
    float endY;             // Target Y position (mm)
    float distance;         // Total path distance (mm)
    float feedrate;         // Feedrate (mm/min)
    
    // Step counts
    int32_t stepsX;         // Steps for X axis
    int32_t stepsY;         // Steps for Y axis
    int32_t totalSteps;     // Total steps (major axis)
    
    // Direction
    bool dirXPositive;      // X direction
    bool dirYPositive;      // Y direction
    
    // Bresenham
    bool xIsMajor;          // True if X has more steps
    float bresenhamRatio;   // minor_steps / major_steps
    
    // S-curve profile
    SCurveProfile scurve;
    
    bool valid;             // Move is valid
};

/**
 * @brief Motion planner class
 */
class MotionPlanner {
public:
    /**
     * @brief Get singleton instance
     */
    static MotionPlanner& getInstance();
    
    /**
     * @brief Initialize planner
     */
    void begin();
    
    /**
     * @brief Plan a move to absolute coordinates
     * @param x Target X position (mm)
     * @param y Target Y position (mm)
     * @param feedrate Feedrate (mm/min)
     * @return Planned move structure
     */
    PlannedMove planAbsolute(float x, float y, float feedrate);
    
    /**
     * @brief Plan a relative move
     * @param dx X distance (mm)
     * @param dy Y distance (mm)
     * @param feedrate Feedrate (mm/min)
     * @return Planned move structure
     */
    PlannedMove planRelative(float dx, float dy, float feedrate);
    
    /**
     * @brief Execute a planned move (non-blocking)
     * @param move Planned move from planAbsolute() or planRelative()
     * @return true if move started
     */
    bool execute(const PlannedMove& move);
    
    /**
     * @brief Wait for current move to complete (blocking)
     * Note: Only use for homing, not in main loop
     */
    void waitComplete();
    
    /**
     * @brief Get current X position
     */
    float getX() const;
    
    /**
     * @brief Get current Y position
     */
    float getY() const;
    
    /**
     * @brief Set current position (after homing)
     */
    void setPosition(float x, float y);
    
    /**
     * @brief Check if currently moving
     */
    bool isMoving() const;
    
    /**
     * @brief Stop motion immediately
     */
    void stop();

private:
    MotionPlanner();
    ~MotionPlanner() = default;
    
    // Prevent copying
    MotionPlanner(const MotionPlanner&) = delete;
    MotionPlanner& operator=(const MotionPlanner&) = delete;
    
    /**
     * @brief Plan move from current position
     */
    PlannedMove planMove(float targetX, float targetY, float feedrate);
    
    /**
     * @brief Validate move bounds
     */
    bool validateBounds(float x, float y) const;
    
    // Current position
    float currentX;
    float currentY;
    
    // Current planned move
    PlannedMove currentMove;
    
    // S-curve generator
    SCurveGenerator scurveGen;
    
    // Stepper driver reference
    StepperDriver& stepper;
};

#endif // MOTION_PLANNER_H