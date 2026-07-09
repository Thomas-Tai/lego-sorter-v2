/**
 * @file stepper_stub.h
 * @brief Test hooks for the native StepperDriver stub.
 *
 * stepper_stub.cpp provides a fake implementation of the StepperDriver
 * class (declared in include/stepper_driver.h) so MotionPlanner can be
 * unit-tested without ESP32 timer hardware. These hooks let tests inspect
 * and control the fake.
 */

#ifndef STEPPER_STUB_H
#define STEPPER_STUB_H

namespace stepper_stub {

extern int beginCalls;
extern int startMoveCalls;
extern int stopCalls;
extern int resetPositionCalls;

extern float lastDx;
extern float lastDy;
extern float lastFeedrate;

extern bool startMoveResult;  // return value for the next startMove()
extern bool movingFlag;       // what isMoving() reports

// Reset all counters and recorded values to defaults
void reset();

}  // namespace stepper_stub

#endif // STEPPER_STUB_H
