/**
 * @file test_scurve.cpp
 * @brief Native unit tests for SCurveGenerator (SM-DES-007 §4, todolist S5-21).
 *
 * Expected values are derived from config.h constants so the tests track
 * configuration changes:
 *   V_MAX_MM_S = 50, A_MAX_MM_S2 = 500, J_MAX_MM_S3 = 2000
 *   tJ = sqrt(2v/J) (single constant-jerk ramp reaching v; A/J cap
 *   cannot bind for v <= A^2/(2J) = 62.5 mm/s)
 *   dAccel = J * tJ^3 / 6 = v*tJ/3;  dDecel = 2*dAccel;  dMin = v*tJ
 *   short moves (d < dMin): vPeak = cbrt(J * d^2 / 2)
 *
 * Run with: pio test -e native
 */

#include <unity.h>
#include <math.h>
#include "scurve.h"

static SCurveGenerator gen;

// Ramp time: single constant-jerk ramp reaches v = J*tJ^2/2
static float jerkTime(float v) {
    float t1 = A_MAX_MM_S2 / J_MAX_MM_S3;
    float t2 = sqrtf(2.0f * v / J_MAX_MM_S3);
    return (t1 < t2) ? t1 : t2;
}

// Ramp-up distance: J * tJ^3 / 6 (= v*tJ/3); ramp-down covers twice this
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
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 2.0f * dJ, p.dDecel);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 100.0f - 3.0f * dJ, p.dCruise);
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
    // 0.5 mm < d_min: v_peak = cbrt(J * d^2 / 2), no cruise phase
    SCurveProfile p = gen.plan(0.5f, V_MAX_MM_S);
    float vPeak = cbrtf(J_MAX_MM_S3 * 0.5f * 0.5f / 2.0f);
    float tJ = jerkTime(vPeak);

    TEST_ASSERT_FLOAT_WITHIN(0.001f, vPeak, p.vCruise);
    TEST_ASSERT_TRUE(p.vCruise < V_MAX_MM_S);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.0f, p.tCruise);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.0f, p.dCruise);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, tJ, p.tJ);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 2.0f * p.dAccel, p.dDecel);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 2.0f * tJ, p.tTotal);
}

void test_short_move_phase_distances_sum_to_total() {
    // Was TEST_IGNORE (KNOWN ISSUE, 2026-07-09): v_peak = sqrt(J*d) left the
    // phase distances summing to ~1.33 mm for a 0.5 mm move. Fixed 2026-07-10:
    // v_peak = cbrt(J*d^2/2) makes d_accel + d_decel = d exactly.
    SCurveProfile p = gen.plan(0.5f, V_MAX_MM_S);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.5f, p.dAccel + p.dCruise + p.dDecel);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.5f, gen.getPosition(p, p.tTotal));
}

void test_cruise_nonnegative_across_distance_sweep() {
    // Was TEST_IGNORE (KNOWN ISSUE, 2026-07-09): d_min = v^2/J = 1.25 mm
    // undercut the real ramp distance, so moves in (1.25, 2.64) mm got a
    // NEGATIVE dCruise/tCruise. Fixed 2026-07-10: d_min = v*tJ is the same
    // expression the full branch subtracts, so cruise >= 0 by construction.
    // Sweep covers the old dead zone and both branch sides of the new d_min.
    const float distances[] = {0.3f, 0.5f, 1.25f, 2.0f, 2.64f,
                               5.0f, 11.0f, 11.2f, 20.0f, 100.0f};
    for (unsigned i = 0; i < sizeof(distances) / sizeof(distances[0]); i++) {
        SCurveProfile p = gen.plan(distances[i], V_MAX_MM_S);
        TEST_ASSERT_TRUE(p.tCruise >= 0.0f);
        TEST_ASSERT_TRUE(p.dCruise >= 0.0f);
        TEST_ASSERT_FLOAT_WITHIN(0.01f, distances[i],
                                 p.dAccel + p.dCruise + p.dDecel);
    }
}

// -------------------------------------------------------- ramp kinematics

void test_ramp_reaches_cruise_velocity() {
    // Velocity continuity at cruise entry: the ramp must END at vCruise
    // (J*tJ^2/2 == vCruise). The pre-fix tJ = sqrt(v/J) ended the ramp at
    // vCruise/2 — a commanded 25 mm/s velocity step at v = 50. Stall risk.
    SCurveProfile pLong = gen.plan(100.0f, V_MAX_MM_S);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, pLong.vCruise,
                             J_MAX_MM_S3 * pLong.tJ * pLong.tJ / 2.0f);
    SCurveProfile pShort = gen.plan(0.5f, V_MAX_MM_S);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, pShort.vCruise,
                             J_MAX_MM_S3 * pShort.tJ * pShort.tJ / 2.0f);
}

void test_peak_acceleration_within_amax() {
    // Peak accel during the ramp is J*tJ; must respect A_MAX
    // (at v_max = 50: sqrt(2*50*2000) ~= 447 mm/s^2 < 500)
    SCurveProfile p = gen.plan(100.0f, V_MAX_MM_S);
    TEST_ASSERT_TRUE(J_MAX_MM_S3 * p.tJ <= A_MAX_MM_S2 + 0.01f);
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

void test_position_during_decel_ramp() {
    // pos = dAccel + dCruise + vCruise*tau - J*tau^3/6 (integral of the
    // decel velocity v_cruise - J*tau^2/2, NOT the accel integral)
    SCurveProfile p = gen.plan(100.0f, V_MAX_MM_S);
    float tau = p.tDecel / 2.0f;
    float expected = p.dAccel + p.dCruise + p.vCruise * tau -
                     J_MAX_MM_S3 * tau * tau * tau / 6.0f;
    TEST_ASSERT_FLOAT_WITHIN(0.01f, expected,
                             gen.getPosition(p, p.tAccel + p.tCruise + tau));
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
    RUN_TEST(test_short_move_phase_distances_sum_to_total);
    RUN_TEST(test_cruise_nonnegative_across_distance_sweep);
    RUN_TEST(test_ramp_reaches_cruise_velocity);
    RUN_TEST(test_peak_acceleration_within_amax);
    RUN_TEST(test_velocity_zero_outside_profile);
    RUN_TEST(test_velocity_during_jerk_rampup);
    RUN_TEST(test_velocity_monotonic_during_accel);
    RUN_TEST(test_velocity_equals_cruise_in_cruise_phase);
    RUN_TEST(test_position_boundaries);
    RUN_TEST(test_position_during_jerk_rampup);
    RUN_TEST(test_position_in_cruise_phase);
    RUN_TEST(test_position_during_decel_ramp);
    RUN_TEST(test_position_monotonic_in_long_move);
    RUN_TEST(test_phase_sequence);
    return UNITY_END();
}
