/**
 * @file endstop_stub.h
 * @brief Test hooks for the native EndstopManager stub.
 *
 * endstop_stub.cpp provides a fake implementation of the EndstopManager
 * class (declared in include/endstop.h) so HomingManager can be
 * unit-tested without GPIO hardware. These hooks let tests inspect and
 * control the fake.
 */

#ifndef ENDSTOP_STUB_H
#define ENDSTOP_STUB_H

namespace endstop_stub {

extern int updateCalls;

extern bool xTriggered;  // what isXTriggered() reports
extern bool yTriggered;  // what isYTriggered() reports

// Reset all counters and states to defaults (untriggered)
void reset();

}  // namespace endstop_stub

#endif // ENDSTOP_STUB_H
