/**
 * @file scurve.cpp
 * @brief S-Curve Motion Profile Implementation
 * 
 * Implements 3-segment S-curve profile per SM-DES-007 §4.3.
 * 
 * MVP S-curve: P1 (jerk ramp-up) → P4 (cruise) → P7 (jerk ramp-down)
 * 
 * For short moves (< d_min), reduces peak velocity:
 *   v_peak = sqrt(j_max × d)
 */

#include "scurve.h"
#include <math.h>

SCurveGenerator::SCurveGenerator() {
}

float SCurveGenerator::calculateMinDistance(float vMax) const {
    // d_min = v_max² / j_max
    return (vMax * vMax) / J_MAX_MM_S3;
}

float SCurveGenerator::calculateJerkTime(float vTarget) const {
    // t_j = min(a_max / j_max, sqrt(v_target / j_max))
    float t1 = A_MAX_MM_S2 / J_MAX_MM_S3;
    float t2 = sqrtf(vTarget / J_MAX_MM_S3);
    return (t1 < t2) ? t1 : t2;
}

SCurveProfile SCurveGenerator::plan(float distance, float vTarget) {
    SCurveProfile profile;
    profile.distance = fabsf(distance);
    profile.vStart = 0.0f;
    profile.vEnd = 0.0f;
    profile.phase = SCurvePhase::IDLE;
    
    // Clamp target velocity
    if (vTarget > V_MAX_MM_S) {
        vTarget = V_MAX_MM_S;
    }
    if (vTarget <= 0) {
        vTarget = V_MAX_MM_S;
    }
    
    // Calculate minimum distance for full S-curve
    float dMin = calculateMinDistance(vTarget);
    
    // Calculate jerk phase time
    float tJ = calculateJerkTime(vTarget);
    
    // Calculate distance during jerk phase (ramp up to peak velocity)
    // During jerk phase: a(t) = j × t, v(t) = j × t² / 2
    // d_jerk = j × t_j³ / 6
    float dJerk = J_MAX_MM_S3 * tJ * tJ * tJ / 6.0f;
    
    // Total acceleration distance (two jerk phases: accel + decel)
    float dAccelDecel = 2.0f * dJerk;
    
    if (profile.distance < dMin) {
        // Short move: reduce peak velocity
        // v_peak = sqrt(j_max × d)
        profile.vCruise = sqrtf(J_MAX_MM_S3 * profile.distance);
        profile.tJ = calculateJerkTime(profile.vCruise);
        profile.tAccel = profile.tJ;
        profile.tDecel = profile.tJ;
        profile.tCruise = 0.0f;
        profile.dAccel = J_MAX_MM_S3 * profile.tJ * profile.tJ * profile.tJ / 6.0f;
        profile.dDecel = profile.dAccel;
        profile.dCruise = 0.0f;
    } else {
        // Full S-curve with cruise phase
        profile.vCruise = vTarget;
        profile.tJ = tJ;
        profile.tAccel = tJ;
        profile.tDecel = tJ;
        profile.dAccel = dJerk;
        profile.dDecel = dJerk;
        profile.dCruise = profile.distance - dAccelDecel;
        profile.tCruise = profile.dCruise / profile.vCruise;
    }
    
    profile.tTotal = profile.tAccel + profile.tCruise + profile.tDecel;
    
    return profile;
}

float SCurveGenerator::jerkVelocity(float t, float jMax, float tJ) const {
    // v(t) = j × t² / 2 (integrating a(t) = j × t)
    return jMax * t * t / 2.0f;
}

float SCurveGenerator::jerkPosition(float t, float jMax, float tJ) const {
    // d(t) = j × t³ / 6 (integrating v(t))
    return jMax * t * t * t / 6.0f;
}

float SCurveGenerator::getVelocity(const SCurveProfile& profile, float t) {
    if (t < 0 || t >= profile.tTotal) {
        return 0.0f;
    }
    
    if (t < profile.tAccel) {
        // Acceleration phase (jerk ramp-up)
        return jerkVelocity(t, J_MAX_MM_S3, profile.tJ);
    } 
    else if (t < profile.tAccel + profile.tCruise) {
        // Cruise phase
        return profile.vCruise;
    }
    else {
        // Deceleration phase (jerk ramp-down)
        float tDecel = t - profile.tAccel - profile.tCruise;
        float tRemaining = profile.tDecel - tDecel;
        // Velocity decreases: v = v_cruise - jerk_velocity(tDecel)
        return profile.vCruise - jerkVelocity(tDecel, J_MAX_MM_S3, profile.tJ);
    }
}

float SCurveGenerator::getPosition(const SCurveProfile& profile, float t) {
    if (t < 0) {
        return 0.0f;
    }
    if (t >= profile.tTotal) {
        return profile.distance;
    }
    
    float pos = 0.0f;
    
    if (t < profile.tAccel) {
        // Acceleration phase
        return jerkPosition(t, J_MAX_MM_S3, profile.tJ);
    }
    
    // Add full acceleration distance
    pos += profile.dAccel;
    
    if (t < profile.tAccel + profile.tCruise) {
        // Cruise phase
        float tCruise = t - profile.tAccel;
        return pos + profile.vCruise * tCruise;
    }
    
    // Add full cruise distance
    pos += profile.dCruise;
    
    // Deceleration phase
    float tDecel = t - profile.tAccel - profile.tCruise;
    pos += jerkPosition(tDecel, J_MAX_MM_S3, profile.tJ);
    
    return pos;
}

SCurvePhase SCurveGenerator::getPhase(const SCurveProfile& profile, float t) {
    if (t < 0) {
        return SCurvePhase::IDLE;
    }
    if (t >= profile.tTotal) {
        return SCurvePhase::COMPLETE;
    }
    
    if (t < profile.tAccel) {
        return SCurvePhase::ACCELERATING;
    }
    if (t < profile.tAccel + profile.tCruise) {
        return SCurvePhase::CRUISING;
    }
    return SCurvePhase::DECELERATING;
}