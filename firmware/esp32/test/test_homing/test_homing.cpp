/**
 * @file test_homing.cpp
 * @brief Native unit tests for HomingManager (SM-DES-007 §7, audit F-6).
 *
 * Regression suite for the false-home defect: every approach phase must
 * verify the endstop actually fired before advancing. With a dead switch
 * or broken wire the approach move runs its full travel and completes -
 * the old code then fell through to backoff/COMPLETE and resetPosition(),
 * declaring a false home at the crash position.
 *
 * Uses the StepperDriver fake (stepper_stub) and EndstopManager fake
 * (endstop_stub). movingFlag stays false so approach moves "complete
 * instantly"; endstop hooks decide whether the switch fired.
 *
 * Run with: pio test -e native
 */

#include <unity.h>
#include "homing.h"
#include "stepper_stub.h"
#include "endstop_stub.h"

static HomingManager& homing = HomingManager::getInstance();

void setUp() {
    stepper_stub::reset();
    endstop_stub::reset();
    homing.begin();
}

void tearDown() {}

// Run update() until homing reaches a terminal state (or safety cap)
static void runToTerminal() {
    for (int i = 0; i < 20 && homing.isHoming(); i++) {
        homing.update();
    }
}

// ------------------------------------------------- dead-endstop failures

void test_x_fast_approach_fails_without_trigger() {
    // Dead X switch: fast approach completes full travel, no contact
    TEST_ASSERT_TRUE(homing.start(true, false));
    runToTerminal();

    TEST_ASSERT_TRUE(homing.hasError());
    TEST_ASSERT_FALSE(homing.isComplete());
    TEST_ASSERT_FALSE(homing.isHoming());
    // Fail-fast: only the fast-approach move was commanded (no backoff/slow)
    TEST_ASSERT_EQUAL_INT(1, stepper_stub::startMoveCalls);
    // Position must NOT be zeroed at the crash location
    TEST_ASSERT_EQUAL_INT(0, stepper_stub::resetPositionCalls);
}

void test_x_slow_approach_fails_if_trigger_lost() {
    // Switch fires at fast approach, then goes dead (intermittent wire)
    endstop_stub::xTriggered = true;
    TEST_ASSERT_TRUE(homing.start(true, false));
    homing.update();  // X_FAST pre-check triggered -> X_BACKOFF
    homing.update();  // X_BACKOFF -> X_SLOW_APPROACH (move starts)
    TEST_ASSERT_TRUE(homing.getPhase() == HomingPhase::X_SLOW_APPROACH);

    endstop_stub::xTriggered = false;  // dies before the slow approach
    homing.update();  // slow approach completes with no contact

    TEST_ASSERT_TRUE(homing.hasError());
    TEST_ASSERT_EQUAL_INT(0, stepper_stub::resetPositionCalls);
}

void test_y_fast_approach_fails_after_x_homes() {
    // X healthy, Y switch dead: X homes, Y fast approach must error
    endstop_stub::xTriggered = true;
    TEST_ASSERT_TRUE(homing.start(true, true));
    runToTerminal();

    TEST_ASSERT_TRUE(homing.hasError());
    TEST_ASSERT_FALSE(homing.isComplete());
    TEST_ASSERT_EQUAL_INT(0, stepper_stub::resetPositionCalls);
}

void test_y_slow_approach_fails_if_trigger_lost() {
    // Both switches fire until Y's slow approach, then Y dies. This is the
    // worst pre-fix path: the old Y_SLOW fall-through called resetPosition()
    // unconditionally.
    endstop_stub::xTriggered = true;
    endstop_stub::yTriggered = true;
    TEST_ASSERT_TRUE(homing.start(true, true));
    homing.update();  // X_FAST -> X_BACKOFF
    homing.update();  // X_BACKOFF -> X_SLOW
    homing.update();  // X_SLOW verified -> Y_FAST
    homing.update();  // Y_FAST pre-check triggered -> Y_BACKOFF
    homing.update();  // Y_BACKOFF -> Y_SLOW
    TEST_ASSERT_TRUE(homing.getPhase() == HomingPhase::Y_SLOW_APPROACH);

    endstop_stub::yTriggered = false;
    homing.update();

    TEST_ASSERT_TRUE(homing.hasError());
    TEST_ASSERT_FALSE(homing.isComplete());
    TEST_ASSERT_EQUAL_INT(0, stepper_stub::resetPositionCalls);
}

// ------------------------------------------------------- healthy paths

void test_x_only_home_succeeds() {
    endstop_stub::xTriggered = true;
    TEST_ASSERT_TRUE(homing.start(true, false));
    runToTerminal();

    TEST_ASSERT_TRUE(homing.isComplete());
    TEST_ASSERT_FALSE(homing.hasError());
    TEST_ASSERT_EQUAL_INT(1, stepper_stub::resetPositionCalls);
    TEST_ASSERT_EQUAL_INT(1, stepper_stub::enableCalls);
}

void test_xy_home_succeeds() {
    endstop_stub::xTriggered = true;
    endstop_stub::yTriggered = true;
    TEST_ASSERT_TRUE(homing.start(true, true));
    runToTerminal();

    TEST_ASSERT_TRUE(homing.isComplete());
    TEST_ASSERT_FALSE(homing.hasError());
    TEST_ASSERT_EQUAL_INT(1, stepper_stub::resetPositionCalls);
}

void test_slow_approach_commands_home_direction() {
    // The X slow approach must command a full-travel move toward negative X
    endstop_stub::xTriggered = true;
    TEST_ASSERT_TRUE(homing.start(true, false));
    homing.update();  // X_FAST pre-check triggered -> X_BACKOFF
    homing.update();  // X_BACKOFF -> X_SLOW (startHomeMove records last args)

    TEST_ASSERT_FLOAT_WITHIN(0.001f, -X_MAX_MM, stepper_stub::lastDx);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f, stepper_stub::lastDy);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, HOMING_SLOW_MM_S * 60.0f,
                             stepper_stub::lastFeedrate);
}

int main(int argc, char** argv) {
    UNITY_BEGIN();

    RUN_TEST(test_x_fast_approach_fails_without_trigger);
    RUN_TEST(test_x_slow_approach_fails_if_trigger_lost);
    RUN_TEST(test_y_fast_approach_fails_after_x_homes);
    RUN_TEST(test_y_slow_approach_fails_if_trigger_lost);
    RUN_TEST(test_x_only_home_succeeds);
    RUN_TEST(test_xy_home_succeeds);
    RUN_TEST(test_slow_approach_commands_home_direction);

    return UNITY_END();
}
