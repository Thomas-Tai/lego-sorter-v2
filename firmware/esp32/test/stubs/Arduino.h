/**
 * @file Arduino.h
 * @brief Native-build shim standing in for the ESP32 Arduino core.
 *
 * Used only by [env:native] (pio test -e native). Provides just enough of
 * the Arduino surface for protocol.cpp / scurve.cpp / motion_planner.cpp /
 * homing.cpp to compile on a desktop toolchain. Never included in the
 * esp32dev build.
 */

#ifndef ARDUINO_NATIVE_SHIM_H
#define ARDUINO_NATIVE_SHIM_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <ctype.h>

#define HIGH 1
#define LOW 0

// ESP32-specific attribute: no-op on native
#define IRAM_ATTR

// Opaque hardware-timer type (only ever used as a pointer in headers)
typedef struct hw_timer_s hw_timer_t;

inline void delay(unsigned long) {}
inline void delayMicroseconds(unsigned int) {}
inline void digitalWrite(int, int) {}

inline unsigned long millis() {
    static unsigned long fakeMillis = 0;
    return ++fakeMillis;
}

// Arduino provides min/max; native stdlib does not (as free functions on
// mixed integer types the firmware uses)
#ifndef ARDUINO
template <typename T>
inline T max(T a, T b) { return (a > b) ? a : b; }
template <typename T>
inline T min(T a, T b) { return (a < b) ? a : b; }
#endif

#endif // ARDUINO_NATIVE_SHIM_H
