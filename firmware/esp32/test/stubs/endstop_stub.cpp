/**
 * @file endstop_stub.cpp
 * @brief Native fake of EndstopManager for HomingManager unit tests.
 *
 * Implements the subset of EndstopManager (include/endstop.h) that
 * homing.cpp links against. Compiled only in [env:native] via
 * build_src_filter; the real endstop.cpp is excluded there.
 */

#include "endstop.h"
#include "endstop_stub.h"

namespace endstop_stub {

int updateCalls = 0;

bool xTriggered = false;
bool yTriggered = false;

void reset() {
    updateCalls = 0;
    xTriggered = false;
    yTriggered = false;
}

}  // namespace endstop_stub

EndstopManager& EndstopManager::getInstance() {
    static EndstopManager instance;
    return instance;
}

EndstopManager::EndstopManager()
    : xTriggeredState(false)
    , yTriggeredState(false)
    , xLastRaw(false)
    , yLastRaw(false)
    , xDebounceStart(0)
    , yDebounceStart(0)
    , triggeredFlag(false)
    , xPrevState(false)
    , yPrevState(false)
    , lastChangeMs(0) {
}

void EndstopManager::update() {
    endstop_stub::updateCalls++;
}

bool EndstopManager::isXTriggered() const {
    return endstop_stub::xTriggered;
}

bool EndstopManager::isYTriggered() const {
    return endstop_stub::yTriggered;
}
