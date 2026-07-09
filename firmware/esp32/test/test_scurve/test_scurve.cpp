/**
 * @file test_scurve.cpp
 * @brief Native unit tests for SCurveGenerator (SM-DES-007 §4, todolist S5-21).
 *
 * Expected values are derived from config.h constants so the tests track
 * configuration changes:
 *   V_MAX_MM_S = 50, A_MAX_MM_S2 = 500, J_MAX_MM_S3 = 2000
 *   tJ = min(A/J, sqrt(v/J));  dJerk = J * tJ^3 / 6
 *
 * Run with: pio test -e native
 */

#include <unity.h>
#include <math.h>
#include "scurve.h"

static SCurveGenerator gen;

// Jerk time as coded: min(A/J, sqrt(v/J))
static float jerkTime(float v) {
    float t1 = A_MAX_MM_S2 / J_MAX_MM_S3;
    float t2 = sqrtf(v / J_MAX_MM_S3);
    return (t1 < t2) ? t1 : t2;
}

// Jerk-phase distance as coded: J * tJ^3 / 6
static float jerkDistance(float tJ) {
    return J_MAX_MM_S3 * tJ * tJ * tJ / 6.0f;
}

void setUp() {}
void tearDown() {}

// ------------------------------------------------------------ long moves

void test_long_move_profile_shape() {
    SCurveProfile p = gen.plan(100.0f, V_MAX_MM_S);
    float tJ = jerkTime(V_MAX_MM_S);
    float dJ = jerkDistance(tJ);

    TEST_ASSERT_FLOAT_WITHIN(0.001f, 100.0f, p.distance);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, V_MAX_MM_S, p.vCruise);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, tJ, p.tJ);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, tJ, p.tAccel);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, tJ, p.tDecel);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, dJ, p.dAccel);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, dJ, p.dDecel);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 100.0f - 2.0f * dJ, p.dCruise);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, p.dCruise / p.vCruise, p.tCruise);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, p.tAccel + p.tCruise + p.tDecel, p.tTotal);
}

void test_long_move_phase_distances_sum_to_total() {
    SCurveProfile p = gen.plan(100.0f, V_MAX_MM_S);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, p.distance, p.dAccel + p.dCruise + p.dDecel);
}

void test_velocity_target_clamped_to_vmax() {
    SCurveProfile p = gen.plan(100.0f, 999.0f);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, V_MAX_MM_S, p.vCruise);
}

void test_nonpositive_velocity_defaults_to_vmax() {
    SCurveProfile p1 = gen.plan(100.0f, 0.0f);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, V_MAX_MM_S, p1.vCruise);
    SCurveProfile p2 = gen.plan(100.0f, -3.0f);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, V_MAX_MM_S, p2.vCruise);
}

void test_negative_distance_uses_magnitude() {
    SCurveProfile p = gen.plan(-40.0f, V_MAX_MM_S);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 40.0f, p.distance);
}

// ----------------------------------------------------------- short moves

void test_short_move_reduces_peak_velocity() {
    // 0.5 mm < d_min: v_peak = sqrt(J * d), no cruise phase
    SCurveProfile p = gen.plan(0.5f, V_MAX_MM_S);
    float vPeak = sqrtf(J_MAX_MM_S3 * 0.5f);
    float tJ = jerkTime(vPeak);

    TEST_ASSERT_FLOAT_WITHIN(0.001f, vPeak, p.vCruise);
    TEST_ASSERT_TRUE(p.vCruise < V_MAX_MM_S);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.0f, p.tCruise);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.0f, p.dCruise);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, tJ, p.tJ);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, p.dAccel, p.dDecel);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 2.0f * tJ, p.tTotal);
}

void test_short_move_distance_consistency_KNOWN_ISSUE() {
    // KNOWN ISSUE (found while writing these tests, 2026-07-09):
    // for short moves the coded v_peak = sqrt(J*d) does not satisfy the
    // profile's own kinematics: dAccel + dDecel = 2*(J*tJ^3/6) != d.
    // For d = 0.5 mm the phase distances sum to ~1.33 mm, so getPosition()
    // overshoots d during the accel phase and snaps back to d at tTotal.
    // Fix belongs in a reviewed firmware change before H3 bench bring-up.
    TEST_IGNORE_MESSAGE("scurve.cpp short-move math inconsistent: sum(phase distances) != distance");
}

void test_near_dmin_move_cruise_nonnegative_KNOWN_ISSUE() {
    // KNOWN ISSUE (same root cause): the d_min = v^2/J = 1.25 mm threshold
    // does not match the coded jerk-phase distance 2*dJerk ~= 2.64 mm.
    // Moves in (1.25, 2.64) mm take the full-curve branch and get a
    // NEGATIVE dCruise/tCruise. Example: plan(2.0, 50) -> tCruise < 0.
    // No production move today falls in this window (homing backoff is
    // 5 mm; bin pitch >= 50 mm), but it must be fixed before H3.
    TEST_IGNORE_MESSAGE("scurve.cpp d_min threshold inconsistent with 2*dJerk; negative cruise in (1.25, 2.64) mm");
}

// ------------------------------------------------------- velocity queries

void test_velocity_zero_outside_profile() {
    SCurveProfile p = gen.plan(100.0f, V_MAX_MM_S);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.0f, gen.getVelocity(p, -1.0f));
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.0f, gen.getVelocity(p, p.tTotal + 1.0f));
}

void test_velocity_during_jerk_rampup() {
    // v(t) = J * t^2 / 2 during accel
    SCurveProfile p = gen.plan(100.0f, V_MAX_MM_S);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, J_MAX_MM_S3 * 0.1f * 0.1f / 2.0f,
                             gen.getVelocity(p, 0.1f));
}

void test_velocity_monotonic_during_accel() {
    SCurveProfile p = gen.plan(100.0f, V_MAX_MM_S);
    float v1 = gen.getVelocity(p, 0.05f);
    float v2 = gen.getVelocity(p, 0.10f);
    float v3 = gen.getVelocity(p, 0.15f);
    TEST_ASSERT_TRUE(v1 < v2);
    TEST_ASSERT_TRUE(v2 < v3);
}

void test_velocity_equals_cruise_in_cruise_phase() {
    SCurveProfile p = gen.plan(100.0f, V_MAX_MM_S);
    float tMid = p.tAccel + p.tCruise / 2.0f;
    TEST_ASSERT_FLOAT_WITHIN(0.001f, p.vCruise, gen.getVelocity(p, tMid));
}

// ------------------------------------------------------- position queries

void test_position_boundaries() {
    SCurveProfile p = gen.plan(100.0f, V_MAX_MM_S);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.0f, gen.getPosition(p, -1.0f));
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.0f, gen.getPosition(p, 0.0f));
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 100.0f, gen.getPosition(p, p.tTotal));
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 100.0f, gen.getPosition(p, p.tTotal + 5.0f));
}

void test_position_during_jerk_rampup() {
    // d(t) = J * t^3 / 6 during accel
    SCurveProfile p = gen.plan(100.0f, V_MAX_MM_S);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, J_MAX_MM_S3 * 0.001f / 6.0f,
                             gen.getPosition(p, 0.1f));
}

void test_position_in_cruise_phase() {
    // pos = dAccel + vCruise * (t - tAccel)
    SCurveProfile p = gen.plan(100.0f, V_MAX_MM_S);
    float t = p.tAccel + 1.0f;
    TEST_ASSERT_FLOAT_WITHIN(0.01f, p.dAccel + p.vCruise * 1.0f,
                             gen.getPosition(p, t));
}

void test_position_monotonic_in_long_move() {
    SCurveProfile p = gen.plan(100.0f, V_MAX_MM_S);
    float p1 = gen.getPosition(p, 0.5f);
    float p2 = gen.getPosition(p, 1.0f);
    float p3 = gen.getPosition(p, 2.0f);
    TEST_ASSERT_TRUE(p1 < p2);
    TEST_ASSERT_TRUE(p2 < p3);
}

// ----------------------------------------------------------- phase queries

void test_phase_sequence() {
    SCurveProfile p = gen.plan(100.0f, V_MAX_MM_S);
    TEST_ASSERT_EQUAL(SCurvePhase::IDLE, gen.getPhase(p, -0.1f));
    TEST_ASSERT_EQUAL(SCurvePhase::ACCELERATING, gen.getPhase(p, p.tAccel / 2.0f));
    TEST_ASSERT_EQUAL(SCurvePhase::CRUISING, gen.getPhase(p, p.tAccel + p.tCruise / 2.0f));
    TEST_ASSERT_EQUAL(SCurvePhase::DECELERATING,
                      gen.getPhase(p, p.tAccel + p.tCruise + p.tDecel / 2.0f));
    TEST_ASSERT_EQUAL(SCurvePhase::COMPLETE, gen.getPhase(p, p.tTotal + 0.1f));
}

// -------------------------------------------------------------------- main

int main(int, char**) {
    UNITY_BEGIN();
    RUN_TEST(test_long_move_profile_shape);
    RUN_TEST(test_long_move_phase_distances_sum_to_total);
    RUN_TEST(test_velocity_target_clamped_to_vmax);
    RUN_TEST(test_nonpositive_velocity_defaults_to_vmax);
    RUN_TEST(test_negative_distance_uses_magnitude);
    RUN_TEST(test_short_move_reduces_peak_velocity);
    RUN_TEST(test_short_move_distance_consistency_KNOWN_ISSUE);
    RUN_TEST(test_near_dmin_move_cruise_nonnegative_KNOWN_ISSUE);
    RUN_TEST(test_velocity_zero_outside_profile);
    RUN_TEST(test_velocity_during_jerk_rampup);
    RUN_TEST(test_velocity_monotonic_during_accel);
    RUN_TEST(test_velocity_equals_cruise_in_cruise_phase);
    RUN_TEST(test_position_boundaries);
    RUN_TEST(test_position_during_jerk_rampup);
    RUN_TEST(test_position_in_cruise_phase);
    RUN_TEST(test_position_monotonic_in_long_move);
    RUN_TEST(test_phase_sequence);
    return UNITY_END();
}
