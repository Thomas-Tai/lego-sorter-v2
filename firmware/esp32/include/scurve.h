/**
 * @file scurve.h
 * @brief S-Curve Motion Profile for LEGO Sorter V2
 * 
 * Implements 3-segment S-curve velocity profile per SM-DES-007 §4.
 * 
 * Profile: P1 (jerk ramp-up) → P4 (cruise) → P7 (jerk ramp-down)
 * 
 * Key formulas:
 *   t_j = min(a_max / j_max, sqrt(v_target / j_max))
 *   d_min = v_max² / j_max = 1.25 mm
 * 
 * @version 1.0
 * @date 2026-03-10
 */

#ifndef SCURVE_H
#define SCURVE_H

#include <Arduino.h>
#include "config.h"

/**
 * @brief S-Curve phase
 */
enum class SCurvePhase : uint8_t {
    IDLE,           // No active motion
    ACCELERATING,   // Jerk ramp-up phase (P1)
    CRUISING,       // Constant velocity phase (P4)
    DECELERATING,   // Jerk ramp-down phase (P7)
    COMPLETE        // Motion complete
};

/**
 * @brief S-Curve profile parameters
 */
struct SCurveProfile {
    float distance;             // Total move distance (mm)
    float vStart;               // Starting velocity (mm/s), usually 0
    float vCruise;              // Cruise velocity (mm/s)
    float vEnd;                 // End velocity (mm/s), usually 0
    
    // Timing
    float tAccel;               // Acceleration time (s)
    float tCruise;              // Cruise time (s)
    float tDecel;               // Deceleration time (s)
    float tTotal;               // Total time (s)
    
    // Distance per phase
    float dAccel;               // Distance during acceleration
    float dCruise;              // Distance during cruise
    float dDecel;               // Distance during deceleration
    
    // Jerk phase duration
    float tJ;                   // Jerk phase time (s)
    
    SCurvePhase phase;
};

/**
 * @brief S-Curve motion generator
 */
class SCurveGenerator {
public:
    SCurveGenerator();
    
    /**
     * @brief Plan a new S-curve profile
     * @param distance Total move distance (mm)
     * @param vTarget Target velocity (mm/s)
     * @return Profile parameters
     */
    SCurveProfile plan(float distance, float vTarget);
    
    /**
     * @brief Get velocity at time t in the profile
     * @param profile Profile parameters
     * @param t Time since start (s)
     * @return Velocity at time t (mm/s)
     */
    float getVelocity(const SCurveProfile& profile, float t);
    
    /**
     * @brief Get position at time t in the profile
     * @param profile Profile parameters
     * @param t Time since start (s)
     * @return Position at time t (mm)
     */
    float getPosition(const SCurveProfile& profile, float t);
    
    /**
     * @brief Get current phase at time t
     * @param profile Profile parameters
     * @param t Time since start (s)
     * @return Current phase
     */
    SCurvePhase getPhase(const SCurveProfile& profile, float t);

private:
    /**
     * @brief Calculate minimum distance for full S-curve
     */
    float calculateMinDistance(float vMax) const;
    
    /**
     * @brief Calculate jerk phase time
     */
    float calculateJerkTime(float vTarget) const;
    
    /**
     * @brief Calculate velocity during jerk phase
     * @param t Time since start of jerk phase
     * @param jMax Maximum jerk
     * @param tJ Jerk phase duration
     * @return Velocity (mm/s)
     */
    float jerkVelocity(float t, float jMax, float tJ) const;
    
    /**
     * @brief Calculate position during jerk phase
     */
    float jerkPosition(float t, float jMax, float tJ) const;
};

#endif // SCURVE_H