/**
 * @file scurve.cpp
 * @brief S-Curve Motion Profile Implementation
 *
 * Implements 3-segment S-curve profile per SM-DES-007 §4.3.
 *
 * MVP S-curve: P1 (jerk ramp-up) → P4 (cruise) → P7 (jerk ramp-down)
 *
 * Each ramp is a single constant-jerk segment (no constant-accel phase):
 *   ramp-up:   v(t) = j × t² / 2, reaches v_cruise at t_j = sqrt(2v / j)
 *   distances: d_accel = v × t_j / 3,  d_decel = 2 × v × t_j / 3
 *              (a single-jerk ramp is not distance-symmetric)
 *   d_min     = d_accel + d_decel = v × t_j
 *
 * For short moves (< d_min), reduces peak velocity so the phase
 * distances sum exactly to d:
 *   v_peak = cbrt(j_max × d² / 2)
 */

#include "scurve.h"
#include <math.h>

SCurveGenerator::SCurveGenerator() {
}

float SCurveGenerator::calculateMinDistance(float vMax) const {
    // Shortest move that still reaches vMax: both ramps, no cruise.
    // d_min = d_accel + d_decel = v × t_j
    return vMax * calculateJerkTime(vMax);
}

float SCurveGenerator::calculateJerkTime(float vTarget) const {
    // Single constant-jerk ramp reaches v = j × t_j² / 2 at
    // t_j = sqrt(2 × v_target / j_max)  (SM-DES-007 §4.5).
    // The a_max/j_max cap cannot bind while v ≤ a_max²/(2·j_max) = 62.5 mm/s,
    // which plan()'s V_MAX clamp guarantees.
    float t1 = A_MAX_MM_S2 / J_MAX_MM_S3;
    float t2 = sqrtf(2.0f * vTarget / J_MAX_MM_S3);
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

    if (profile.distance < dMin) {
        // Short move: no cruise. Solve d = v_peak × t_j(v_peak)
        // = sqrt(2 × v_peak³ / j_max)  =>  v_peak = cbrt(j_max × d² / 2)
        profile.vCruise = cbrtf(
            J_MAX_MM_S3 * profile.distance * profile.distance / 2.0f);
    } else {
        // Full S-curve with cruise phase
        profile.vCruise = vTarget;
    }

    profile.tJ = calculateJerkTime(profile.vCruise);
    profile.tAccel = profile.tJ;
    profile.tDecel = profile.tJ;

    // Ramp-up covers j × t_j³ / 6 = v × t_j / 3; ramp-down covers the
    // mirror integral v × t_j − j × t_j³ / 6 = 2 × v × t_j / 3.
    profile.dAccel = J_MAX_MM_S3 * profile.tJ * profile.tJ * profile.tJ / 6.0f;
    profile.dDecel = 2.0f * profile.dAccel;

    profile.dCruise = profile.distance - profile.dAccel - profile.dDecel;
    if (profile.dCruise < 0.0f) {
        profile.dCruise = 0.0f;  // float round-off on short moves
    }
    profile.tCruise =
        (profile.vCruise > 0.0f) ? profile.dCruise / profile.vCruise : 0.0f;

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

    // Deceleration phase: v(τ) = v_cruise − j × τ² / 2, so
    // d(τ) = v_cruise × τ − j × τ³ / 6
    float tDecel = t - profile.tAccel - profile.tCruise;
    pos += profile.vCruise * tDecel - jerkPosition(tDecel, J_MAX_MM_S3, profile.tJ);

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
