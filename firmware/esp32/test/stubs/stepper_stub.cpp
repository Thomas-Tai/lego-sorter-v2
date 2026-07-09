/**
 * @file stepper_stub.cpp
 * @brief Native fake of StepperDriver for MotionPlanner unit tests.
 *
 * Implements the subset of StepperDriver (include/stepper_driver.h) that
 * motion_planner.cpp links against. Compiled only in [env:native] via
 * build_src_filter; the real stepper_driver.cpp is excluded there.
 */

#include "stepper_driver.h"
#include "stepper_stub.h"

namespace stepper_stub {

int beginCalls = 0;
int startMoveCalls = 0;
int stopCalls = 0;
int resetPositionCalls = 0;

float lastDx = 0.0f;
float lastDy = 0.0f;
float lastFeedrate = 0.0f;

bool startMoveResult = true;
bool movingFlag = false;

void reset() {
    beginCalls = 0;
    startMoveCalls = 0;
    stopCalls = 0;
    resetPositionCalls = 0;
    lastDx = 0.0f;
    lastDy = 0.0f;
    lastFeedrate = 0.0f;
    startMoveResult = true;
    movingFlag = false;
}

}  // namespace stepper_stub

StepperDriver& StepperDriver::getInstance() {
    static StepperDriver instance;
    return instance;
}

StepperDriver::StepperDriver()
    : enabled(false)
    , moving(false)
    , moveComplete(false)
    , posX(0.0f)
    , posY(0.0f)
    , bresenhamError(0.0f)
    , bresenhamRatio(0.0f)
    , dirX(false)
    , dirY(false)
    , xIsMajor(false)
    , timer(nullptr)
    , timerRunning(false)
    , stepPinHigh(false)
    , stepPinMaskX(0)
    , stepPinMaskY(0) {
    segment.stepsX = 0;
    segment.stepsY = 0;
    segment.totalSteps = 0;
    segment.currentVelocity = 0.0f;
    segment.active = false;
}

void StepperDriver::begin() {
    stepper_stub::beginCalls++;
}

bool StepperDriver::startMove(float dx, float dy, float feedrate) {
    stepper_stub::startMoveCalls++;
    stepper_stub::lastDx = dx;
    stepper_stub::lastDy = dy;
    stepper_stub::lastFeedrate = feedrate;
    return stepper_stub::startMoveResult;
}

bool StepperDriver::isMoving() const {
    return stepper_stub::movingFlag;
}

void StepperDriver::stop() {
    stepper_stub::stopCalls++;
}

void StepperDriver::resetPosition() {
    stepper_stub::resetPositionCalls++;
}
