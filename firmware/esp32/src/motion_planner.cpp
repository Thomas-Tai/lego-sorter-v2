/**
 * @file motion_planner.cpp
 * @brief Motion Planner Implementation
 * 
 * Coordinates XY motion with S-curve profiles per SM-DES-007 §5.
 */

#include "motion_planner.h"
#include <math.h>

MotionPlanner& MotionPlanner::getInstance() {
    static MotionPlanner planner;
    return planner;
}

MotionPlanner::MotionPlanner()
    : currentX(0.0f)
    , currentY(0.0f)
    , stepper(StepperDriver::getInstance())
{
    currentMove.valid = false;
}

void MotionPlanner::begin() {
    stepper.begin();
    currentX = 0.0f;
    currentY = 0.0f;
}

bool MotionPlanner::validateBounds(float x, float y) const {
    return (x >= 0.0f && x <= X_MAX_MM && 
            y >= 0.0f && y <= Y_MAX_MM);
}

PlannedMove MotionPlanner::planAbsolute(float x, float y, float feedrate) {
    return planMove(x, y, feedrate);
}

PlannedMove MotionPlanner::planRelative(float dx, float dy, float feedrate) {
    return planMove(currentX + dx, currentY + dy, feedrate);
}

PlannedMove MotionPlanner::planMove(float targetX, float targetY, float feedrate) {
    PlannedMove move;
    move.valid = false;
    
    // Validate bounds
    if (!validateBounds(targetX, targetY)) {
        return move;
    }
    
    // Clamp feedrate
    if (feedrate <= 0) {
        feedrate = V_MAX_MM_S * 60.0f;  // Default to max
    }
    if (feedrate > V_MAX_MM_S * 60.0f) {
        feedrate = V_MAX_MM_S * 60.0f;
    }
    
    // Store positions
    move.startX = currentX;
    move.startY = currentY;
    move.endX = targetX;
    move.endY = targetY;
    move.feedrate = feedrate;
    
    // Calculate deltas
    float dx = targetX - currentX;
    float dy = targetY - currentY;
    
    // Calculate total path distance
    move.distance = sqrtf(dx * dx + dy * dy);
    
    if (move.distance < 0.001f) {
        // No movement needed
        move.valid = true;
        move.stepsX = 0;
        move.stepsY = 0;
        move.totalSteps = 0;
        return move;
    }
    
    // Calculate step counts
    move.stepsX = abs((int32_t)(dx * STEPS_PER_MM));
    move.stepsY = abs((int32_t)(dy * STEPS_PER_MM));
    move.totalSteps = max(move.stepsX, move.stepsY);
    
    // Set directions
    move.dirXPositive = (dx >= 0);
    move.dirYPositive = (dy >= 0);
    
    // Determine major axis for Bresenham
    if (move.stepsX >= move.stepsY) {
        move.xIsMajor = true;
        move.bresenhamRatio = (move.stepsY > 0) ? 
            (float)move.stepsY / (float)move.stepsX : 0.0f;
    } else {
        move.xIsMajor = false;
        move.bresenhamRatio = (move.stepsX > 0) ? 
            (float)move.stepsX / (float)move.stepsY : 0.0f;
    }
    
    // Plan S-curve profile
    float vTarget = feedrate / 60.0f;  // mm/min to mm/s
    move.scurve = scurveGen.plan(move.distance, vTarget);
    
    move.valid = true;
    return move;
}

bool MotionPlanner::execute(const PlannedMove& move) {
    if (!move.valid) {
        return false;
    }
    
    if (move.totalSteps == 0) {
        // No movement needed, update position
        currentX = move.endX;
        currentY = move.endY;
        return true;
    }
    
    // Store current move
    currentMove = move;
    
    // Calculate deltas for stepper driver
    float dx = move.endX - move.startX;
    float dy = move.endY - move.startY;
    
    // Start move via stepper driver
    if (!stepper.startMove(dx, dy, move.feedrate)) {
        return false;
    }
    
    // Update position tracking
    currentX = move.endX;
    currentY = move.endY;
    
    return true;
}

void MotionPlanner::waitComplete() {
    while (stepper.isMoving()) {
        delay(1);
    }
}

float MotionPlanner::getX() const {
    return currentX;
}

float MotionPlanner::getY() const {
    return currentY;
}

void MotionPlanner::setPosition(float x, float y) {
    currentX = x;
    currentY = y;
    stepper.resetPosition();
}

bool MotionPlanner::isMoving() const {
    return stepper.isMoving();
}

void MotionPlanner::stop() {
    stepper.stop();
}