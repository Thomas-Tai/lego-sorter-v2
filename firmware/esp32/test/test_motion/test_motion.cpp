/**
 * @file test_motion.cpp
 * @brief Native unit tests for MotionPlanner (SM-DES-007 §5, todolist S5-24).
 *
 * Uses the StepperDriver fake in test/stubs/stepper_stub.cpp so the XY
 * Bresenham planning math runs without ESP32 timer hardware.
 *
 * Run with: pio test -e native
 */

#include <unity.h>
#include <math.h>
#include "motion_planner.h"
#include "stepper_stub.h"

static MotionPlanner& planner = MotionPlanner::getInstance();

void setUp() {
    // Reset planner position first (calls the stub), then zero stub counters
    planner.setPosition(0.0f, 0.0f);
    stepper_stub::reset();
}

void tearDown() {}

// ------------------------------------------------------------- planning

void test_plan_absolute_basic() {
    PlannedMove m = planner.planAbsolute(100.0f, 50.0f, 3000.0f);
    TEST_ASSERT_TRUE(m.valid);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, sqrtf(100.0f * 100.0f + 50.0f * 50.0f), m.distance);
    TEST_ASSERT_EQUAL_INT32(100 * STEPS_PER_MM, m.stepsX);
    TEST_ASSERT_EQUAL_INT32(50 * STEPS_PER_MM, m.stepsY);
    TEST_ASSERT_EQUAL_INT32(100 * STEPS_PER_MM, m.totalSteps);
    TEST_ASSERT_TRUE(m.dirXPositive);
    TEST_ASSERT_TRUE(m.dirYPositive);
}

void test_plan_bresenham_x_major() {
    PlannedMove m = planner.planAbsolute(100.0f, 50.0f, 3000.0f);
    TEST_ASSERT_TRUE(m.xIsMajor);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.5f, m.bresenhamRatio);
}

void test_plan_bresenham_y_major() {
    PlannedMove m = planner.planAbsolute(10.0f, 40.0f, 3000.0f);
    TEST_ASSERT_TRUE(m.valid);
    TEST_ASSERT_FALSE(m.xIsMajor);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.25f, m.bresenhamRatio);
    TEST_ASSERT_EQUAL_INT32(40 * STEPS_PER_MM, m.totalSteps);
}

void test_plan_bresenham_equal_steps() {
    // stepsX == stepsY: X wins the major-axis tie, ratio = 1
    PlannedMove m = planner.planAbsolute(40.0f, 40.0f, 3000.0f);
    TEST_ASSERT_TRUE(m.xIsMajor);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 1.0f, m.bresenhamRatio);
}

void test_plan_pure_x_move() {
    PlannedMove m = planner.planAbsolute(50.0f, 0.0f, 3000.0f);
    TEST_ASSERT_TRUE(m.valid);
    TEST_ASSERT_TRUE(m.xIsMajor);
    TEST_ASSERT_EQUAL_INT32(0, m.stepsY);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.0f, m.bresenhamRatio);
}

void test_plan_pure_y_move() {
    PlannedMove m = planner.planAbsolute(0.0f, 50.0f, 3000.0f);
    TEST_ASSERT_TRUE(m.valid);
    TEST_ASSERT_FALSE(m.xIsMajor);
    TEST_ASSERT_EQUAL_INT32(0, m.stepsX);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.0f, m.bresenhamRatio);
}

void test_plan_relative_directions() {
    planner.setPosition(10.0f, 10.0f);
    PlannedMove m = planner.planRelative(-5.0f, 20.0f, 3000.0f);
    TEST_ASSERT_TRUE(m.valid);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 5.0f, m.endX);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 30.0f, m.endY);
    TEST_ASSERT_FALSE(m.dirXPositive);
    TEST_ASSERT_TRUE(m.dirYPositive);
    TEST_ASSERT_EQUAL_INT32(5 * STEPS_PER_MM, m.stepsX);
    TEST_ASSERT_EQUAL_INT32(20 * STEPS_PER_MM, m.stepsY);
}

void test_plan_zero_distance_move() {
    planner.setPosition(30.0f, 40.0f);
    PlannedMove m = planner.planAbsolute(30.0f, 40.0f, 3000.0f);
    TEST_ASSERT_TRUE(m.valid);
    TEST_ASSERT_EQUAL_INT32(0, m.totalSteps);
}

void test_plan_scurve_integration() {
    // feedrate mm/min is converted to mm/s for the S-curve
    PlannedMove m = planner.planAbsolute(100.0f, 0.0f, 3000.0f);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 3000.0f / 60.0f, m.scurve.vCruise);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 100.0f, m.scurve.distance);
}

// ------------------------------------------------------------ validation

void test_plan_bounds_max_corner_ok() {
    PlannedMove m = planner.planAbsolute(X_MAX_MM, Y_MAX_MM, 3000.0f);
    TEST_ASSERT_TRUE(m.valid);
}

void test_plan_out_of_bounds_rejected() {
    TEST_ASSERT_FALSE(planner.planAbsolute(X_MAX_MM + 0.1f, 0.0f, 3000.0f).valid);
    TEST_ASSERT_FALSE(planner.planAbsolute(0.0f, Y_MAX_MM + 0.1f, 3000.0f).valid);
    TEST_ASSERT_FALSE(planner.planAbsolute(-0.1f, 0.0f, 3000.0f).valid);
    TEST_ASSERT_FALSE(planner.planAbsolute(0.0f, -0.1f, 3000.0f).valid);
}

void test_plan_feedrate_clamped() {
    float maxFeedrate = V_MAX_MM_S * 60.0f;
    PlannedMove m1 = planner.planAbsolute(100.0f, 0.0f, 10000.0f);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, maxFeedrate, m1.feedrate);
    PlannedMove m2 = planner.planAbsolute(100.0f, 0.0f, 0.0f);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, maxFeedrate, m2.feedrate);
    PlannedMove m3 = planner.planAbsolute(100.0f, 0.0f, -50.0f);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, maxFeedrate, m3.feedrate);
}

// ------------------------------------------------------------- execution

void test_execute_dispatches_to_stepper() {
    PlannedMove m = planner.planAbsolute(50.0f, 20.0f, 1500.0f);
    TEST_ASSERT_TRUE(planner.execute(m));
    TEST_ASSERT_EQUAL_INT(1, stepper_stub::startMoveCalls);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 50.0f, stepper_stub::lastDx);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 20.0f, stepper_stub::lastDy);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1500.0f, stepper_stub::lastFeedrate);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 50.0f, planner.getX());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 20.0f, planner.getY());
}

void test_execute_invalid_move_refused() {
    PlannedMove m = planner.planAbsolute(X_MAX_MM + 10.0f, 0.0f, 3000.0f);
    TEST_ASSERT_FALSE(planner.execute(m));
    TEST_ASSERT_EQUAL_INT(0, stepper_stub::startMoveCalls);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f, planner.getX());
}

void test_execute_zero_step_move_skips_stepper() {
    planner.setPosition(30.0f, 40.0f);
    stepper_stub::reset();
    PlannedMove m = planner.planAbsolute(30.0f, 40.0f, 3000.0f);
    TEST_ASSERT_TRUE(planner.execute(m));
    TEST_ASSERT_EQUAL_INT(0, stepper_stub::startMoveCalls);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 30.0f, planner.getX());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 40.0f, planner.getY());
}

void test_execute_stepper_failure_keeps_position() {
    stepper_stub::startMoveResult = false;
    PlannedMove m = planner.planAbsolute(50.0f, 20.0f, 1500.0f);
    TEST_ASSERT_FALSE(planner.execute(m));
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f, planner.getX());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f, planner.getY());
}

// --------------------------------------------------------- state passthrough

void test_set_position_resets_stepper() {
    planner.setPosition(5.0f, 7.0f);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 5.0f, planner.getX());
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 7.0f, planner.getY());
    TEST_ASSERT_EQUAL_INT(1, stepper_stub::resetPositionCalls);
}

void test_stop_and_is_moving_passthrough() {
    planner.stop();
    TEST_ASSERT_EQUAL_INT(1, stepper_stub::stopCalls);
    stepper_stub::movingFlag = true;
    TEST_ASSERT_TRUE(planner.isMoving());
    stepper_stub::movingFlag = false;
    TEST_ASSERT_FALSE(planner.isMoving());
}

// -------------------------------------------------------------------- main

int main(int, char**) {
    UNITY_BEGIN();
    RUN_TEST(test_plan_absolute_basic);
    RUN_TEST(test_plan_bresenham_x_major);
    RUN_TEST(test_plan_bresenham_y_major);
    RUN_TEST(test_plan_bresenham_equal_steps);
    RUN_TEST(test_plan_pure_x_move);
    RUN_TEST(test_plan_pure_y_move);
    RUN_TEST(test_plan_relative_directions);
    RUN_TEST(test_plan_zero_distance_move);
    RUN_TEST(test_plan_scurve_integration);
    RUN_TEST(test_plan_bounds_max_corner_ok);
    RUN_TEST(test_plan_out_of_bounds_rejected);
    RUN_TEST(test_plan_feedrate_clamped);
    RUN_TEST(test_execute_dispatches_to_stepper);
    RUN_TEST(test_execute_invalid_move_refused);
    RUN_TEST(test_execute_zero_step_move_skips_stepper);
    RUN_TEST(test_execute_stepper_failure_keeps_position);
    RUN_TEST(test_set_position_resets_stepper);
    RUN_TEST(test_stop_and_is_moving_passthrough);
    return UNITY_END();
}
