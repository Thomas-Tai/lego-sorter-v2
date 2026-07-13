/**
 * @file test_homing.cpp
 * @brief Native unit tests for HomingManager (SM-DES-007 §7, audit F-6/F-8).
 *
 * Regression suite for two homing safety guards:
 *  - F-6 approach guard: every approach phase must verify the endstop
 *    actually fired before advancing. With a dead switch the approach move
 *    runs its full travel and completes - the old code then fell through
 *    to backoff/COMPLETE and resetPosition(), declaring a false home at
 *    the crash position.
 *  - F-8 release guard: after the 5 mm backoff the switch must read
 *    released. Still-triggered means a jammed lever/striker or, with NC
 *    wiring, a broken wire faking a permanent trigger (NC + pull-up reads
 *    HIGH = triggered on a severed wire, so without this guard a broken
 *    wire would produce an instant false home at the current position).
 *
 * Uses the StepperDriver fake (stepper_stub) and EndstopManager fake
 * (endstop_stub). movingFlag stays false so approach moves "complete
 * instantly"; tests script the endstop hooks between update() calls to
 * simulate the physical trigger lifecycle (press at approach, release
 * after backoff, press again at slow approach).
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

// Drive a healthy X homing lifecycle: press -> backoff release -> re-press.
// Leaves the state machine just past X (phase Y_FAST_APPROACH when homeY
// was requested, else terminal COMPLETE).
static void driveHealthyX() {
    endstop_stub::xTriggered = true;   // switch pressed (or already held)
    homing.update();                   // X_FAST pre-check -> X_BACKOFF
    endstop_stub::xTriggered = false;  // 5 mm backoff releases the lever
    homing.update();                   // X_BACKOFF release guard -> X_SLOW
    endstop_stub::xTriggered = true;   // slow approach presses it again
    homing.update();                   // X_SLOW approach guard -> X done
}

// ------------------------------------------------- dead-endstop failures

void test_x_fast_approach_fails_without_trigger() {
    // Dead X switch (NO-era failure): fast approach completes full travel
    // with no contact
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

void test_x_backoff_fails_if_switch_never_releases() {
    // Permanent trigger: jammed lever/striker, or broken NC wire (pull-up
    // reads HIGH = triggered forever). Release guard must refuse to home.
    endstop_stub::xTriggered = true;
    TEST_ASSERT_TRUE(homing.start(true, false));
    homing.update();  // X_FAST pre-check triggered -> X_BACKOFF
    homing.update();  // X_BACKOFF release guard times out

    TEST_ASSERT_TRUE(homing.hasError());
    TEST_ASSERT_FALSE(homing.isComplete());
    // Only the backoff move was commanded; slow approach never started
    TEST_ASSERT_EQUAL_INT(1, stepper_stub::startMoveCalls);
    TEST_ASSERT_EQUAL_INT(0, stepper_stub::resetPositionCalls);
}

void test_x_slow_approach_fails_if_trigger_lost() {
    // Switch fires at fast approach, releases at backoff, then goes dead
    // (intermittent wire) - slow approach completes with no contact
    endstop_stub::xTriggered = true;
    TEST_ASSERT_TRUE(homing.start(true, false));
    homing.update();                   // X_FAST pre-check -> X_BACKOFF
    endstop_stub::xTriggered = false;  // backoff releases normally
    homing.update();                   // release guard passes -> X_SLOW
    TEST_ASSERT_TRUE(homing.getPhase() == HomingPhase::X_SLOW_APPROACH);

    homing.update();  // slow approach completes, switch never fires again

    TEST_ASSERT_TRUE(homing.hasError());
    TEST_ASSERT_EQUAL_INT(0, stepper_stub::resetPositionCalls);
}

void test_y_fast_approach_fails_after_x_homes() {
    // X healthy, Y switch dead: X homes, Y fast approach must error
    TEST_ASSERT_TRUE(homing.start(true, true));
    driveHealthyX();
    runToTerminal();

    TEST_ASSERT_TRUE(homing.hasError());
    TEST_ASSERT_FALSE(homing.isComplete());
    TEST_ASSERT_EQUAL_INT(0, stepper_stub::resetPositionCalls);
}

void test_y_backoff_fails_if_switch_never_releases() {
    // X healthy; Y permanently triggered (jam or broken NC wire)
    TEST_ASSERT_TRUE(homing.start(true, true));
    driveHealthyX();
    endstop_stub::yTriggered = true;
    homing.update();  // Y_FAST pre-check triggered -> Y_BACKOFF
    homing.update();  // Y_BACKOFF release guard times out

    TEST_ASSERT_TRUE(homing.hasError());
    TEST_ASSERT_FALSE(homing.isComplete());
    TEST_ASSERT_EQUAL_INT(0, stepper_stub::resetPositionCalls);
}

void test_y_slow_approach_fails_if_trigger_lost() {
    // Both axes healthy until Y's slow approach, then Y dies. This is the
    // worst pre-F-6 path: the old Y_SLOW fall-through called
    // resetPosition() unconditionally.
    TEST_ASSERT_TRUE(homing.start(true, true));
    driveHealthyX();
    endstop_stub::yTriggered = true;
    homing.update();                   // Y_FAST pre-check -> Y_BACKOFF
    endstop_stub::yTriggered = false;  // backoff releases normally
    homing.update();                   // release guard passes -> Y_SLOW
    TEST_ASSERT_TRUE(homing.getPhase() == HomingPhase::Y_SLOW_APPROACH);

    homing.update();  // slow approach completes, switch never fires again

    TEST_ASSERT_TRUE(homing.hasError());
    TEST_ASSERT_FALSE(homing.isComplete());
    TEST_ASSERT_EQUAL_INT(0, stepper_stub::resetPositionCalls);
}

// ------------------------------------------------------- healthy paths

void test_x_only_home_succeeds() {
    TEST_ASSERT_TRUE(homing.start(true, false));
    driveHealthyX();

    TEST_ASSERT_TRUE(homing.isComplete());
    TEST_ASSERT_FALSE(homing.hasError());
    TEST_ASSERT_EQUAL_INT(1, stepper_stub::resetPositionCalls);
    TEST_ASSERT_EQUAL_INT(1, stepper_stub::enableCalls);
}

void test_xy_home_succeeds() {
    TEST_ASSERT_TRUE(homing.start(true, true));
    driveHealthyX();                   // ends at Y_FAST_APPROACH
    endstop_stub::yTriggered = true;
    homing.update();                   // Y_FAST pre-check -> Y_BACKOFF
    endstop_stub::yTriggered = false;  // backoff releases
    homing.update();                   // release guard passes -> Y_SLOW
    endstop_stub::yTriggered = true;   // slow approach presses it
    homing.update();                   // Y_SLOW guard -> COMPLETE

    TEST_ASSERT_TRUE(homing.isComplete());
    TEST_ASSERT_FALSE(homing.hasError());
    TEST_ASSERT_EQUAL_INT(1, stepper_stub::resetPositionCalls);
}

void test_slow_approach_commands_home_direction() {
    // The X slow approach must command a full-travel move toward negative X
    endstop_stub::xTriggered = true;
    TEST_ASSERT_TRUE(homing.start(true, false));
    homing.update();                   // X_FAST pre-check -> X_BACKOFF
    endstop_stub::xTriggered = false;  // backoff releases
    homing.update();                   // release guard passes, slow move starts

    TEST_ASSERT_TRUE(homing.getPhase() == HomingPhase::X_SLOW_APPROACH);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, -X_MAX_MM, stepper_stub::lastDx);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f, stepper_stub::lastDy);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, HOMING_SLOW_MM_S * 60.0f,
                             stepper_stub::lastFeedrate);
}

int main(int argc, char** argv) {
    UNITY_BEGIN();

    RUN_TEST(test_x_fast_approach_fails_without_trigger);
    RUN_TEST(test_x_backoff_fails_if_switch_never_releases);
    RUN_TEST(test_x_slow_approach_fails_if_trigger_lost);
    RUN_TEST(test_y_fast_approach_fails_after_x_homes);
    RUN_TEST(test_y_backoff_fails_if_switch_never_releases);
    RUN_TEST(test_y_slow_approach_fails_if_trigger_lost);
    RUN_TEST(test_x_only_home_succeeds);
    RUN_TEST(test_xy_home_succeeds);
    RUN_TEST(test_slow_approach_commands_home_direction);

    return UNITY_END();
}
