/**
 * @file stepper_driver.cpp
 * @brief Stepper Motor Driver Implementation
 * 
 * Timer ISR-driven step generation per SM-DES-007 §6.
 * Uses ESP32 hardware Timer0 with one-shot mode.
 */

#include "stepper_driver.h"
#include "config.h"

// Global instance for ISR access
static StepperDriver* instance = nullptr;

// External ISR function
void IRAM_ATTR stepperTimerISR() {
    if (instance != nullptr) {
        instance->handleISR();
    }
}

StepperDriver& StepperDriver::getInstance() {
    static StepperDriver driver;
    return driver;
}

StepperDriver::StepperDriver()
    : enabled(false)
    , moving(false)
    , moveComplete(false)
    , posX(0.0f)
    , posY(0.0f)
    , bresenhamError(0.0f)
    , bresenhamRatio(0.0f)
    , dirX(true)
    , dirY(true)
    , xIsMajor(true)
    , timer(nullptr)
    , timerRunning(false)
    , stepPinHigh(false)
    , stepPinMaskX(0)
    , stepPinMaskY(0)
{
    segment.active = false;
    segment.stepsX = 0;
    segment.stepsY = 0;
    segment.totalSteps = 0;
    segment.currentVelocity = 0.0f;
}

void StepperDriver::begin() {
    // Configure GPIO pins
    pinMode(PIN_X_STEP, OUTPUT);
    pinMode(PIN_X_DIR, OUTPUT);
    pinMode(PIN_Y_STEP, OUTPUT);
    pinMode(PIN_Y_DIR, OUTPUT);
    pinMode(PIN_ENABLE, OUTPUT);
    
    // Set initial state
    digitalWrite(PIN_X_STEP, LOW);
    digitalWrite(PIN_Y_STEP, LOW);
    digitalWrite(PIN_X_DIR, LOW);
    digitalWrite(PIN_Y_DIR, LOW);
    digitalWrite(PIN_ENABLE, HIGH);  // Disabled (active LOW)
    
    // Calculate GPIO masks for fast GPIO writes
    stepPinMaskX = 1 << PIN_X_STEP;
    stepPinMaskY = 1 << PIN_Y_STEP;
    
    // Store instance for ISR
    instance = this;
}

void StepperDriver::enable() {
    digitalWrite(PIN_ENABLE, LOW);  // Active LOW = enabled
    enabled = true;
}

void StepperDriver::disable() {
    stop();  // Stop any motion first
    digitalWrite(PIN_ENABLE, HIGH);  // Active LOW = disabled
    enabled = false;
}

bool StepperDriver::isEnabled() const {
    return enabled;
}

bool StepperDriver::startMove(float dx, float dy, float feedrate) {
    // Check if already moving
    if (moving) {
        return false;
    }
    
    // Calculate step counts
    int32_t stepsX = abs((int32_t)(dx * STEPS_PER_MM));
    int32_t stepsY = abs((int32_t)(dy * STEPS_PER_MM));
    
    if (stepsX == 0 && stepsY == 0) {
        // No movement needed
        moveComplete = true;
        return true;
    }
    
    // Set directions
    dirX = (dx >= 0);
    dirY = (dy >= 0);
    digitalWrite(PIN_X_DIR, dirX ? HIGH : LOW);
    digitalWrite(PIN_Y_DIR, dirY ? HIGH : LOW);
    
    // Setup motion segment
    segment.stepsX = stepsX;
    segment.stepsY = stepsY;
    segment.totalSteps = max(stepsX, stepsY);
    segment.currentVelocity = feedrate / 60.0f;  // mm/min to mm/s
    segment.active = true;
    
    // Clamp velocity
    if (segment.currentVelocity > V_MAX_MM_S) {
        segment.currentVelocity = V_MAX_MM_S;
    }
    
    // Setup Bresenham
    if (stepsX >= stepsY) {
        xIsMajor = true;
        bresenhamRatio = (stepsY > 0) ? (float)stepsY / (float)stepsX : 0.0f;
    } else {
        xIsMajor = false;
        bresenhamRatio = (stepsX > 0) ? (float)stepsX / (float)stepsY : 0.0f;
    }
    bresenhamError = 0.0f;
    
    // Initialize and start timer
    setupTimer();
    moving = true;
    moveComplete = false;
    
    return true;
}

bool StepperDriver::isMoving() const {
    return moving;
}

void StepperDriver::stop() {
    if (timer != nullptr && timerRunning) {
        timerStop(timer);
        timerRunning = false;
    }
    moving = false;
    segment.active = false;
}

bool StepperDriver::isMoveComplete() const {
    return moveComplete;
}

void StepperDriver::clearMoveComplete() {
    moveComplete = false;
}

float StepperDriver::getPositionX() const {
    return posX;
}

float StepperDriver::getPositionY() const {
    return posY;
}

void StepperDriver::resetPosition() {
    posX = 0.0f;
    posY = 0.0f;
}

void StepperDriver::setupTimer() {
    // Use hardware timer 0, with 1 MHz base clock (1 µs resolution)
    if (timer == nullptr) {
        timer = timerBegin(0, 80, true);  // Timer 0, prescaler 80 (80MHz/80=1MHz), count up
        timerAttachInterrupt(timer, &stepperTimerISR, true);
    }
    
    // Calculate initial step interval
    float interval_us = 1000000.0f / (segment.currentVelocity * STEPS_PER_MM);
    interval_us = max(interval_us, 10.0f);  // Minimum 10 µs
    
    // Set one-shot alarm
    timerAlarmWrite(timer, (uint64_t)interval_us, false);
    timerAlarmEnable(timer);
    timerStart(timer);
    timerRunning = true;
}

void StepperDriver::setStepInterval(float velocity_mm_s) {
    if (timer == nullptr || velocity_mm_s <= 0) return;
    
    float interval_us = 1000000.0f / (velocity_mm_s * STEPS_PER_MM);
    interval_us = max(interval_us, 10.0f);
    
    timerAlarmWrite(timer, (uint64_t)interval_us, false);
    timerAlarmEnable(timer);
}

void StepperDriver::handleISR() {
    // This is called from ISR - keep it minimal
    
    if (!segment.active || segment.totalSteps <= 0) {
        // Motion complete
        segment.active = false;
        moving = false;
        timerRunning = false;
        timerStop(timer);
        moveComplete = true;
        return;
    }
    
    // Generate step pulse for major axis
    if (xIsMajor) {
        // X is major axis
        digitalWrite(PIN_X_STEP, HIGH);
        
        // Update position
        if (dirX) {
            // Moving in positive X direction
        }
        segment.stepsX--;
        
        // Bresenham: check if we need to step minor axis
        bresenhamError += bresenhamRatio;
        if (bresenhamError >= 1.0f) {
            digitalWrite(PIN_Y_STEP, HIGH);
            segment.stepsY--;
            bresenhamError -= 1.0f;
            // Small delay for step pulse
            delayMicroseconds(2);
            digitalWrite(PIN_Y_STEP, LOW);
        }
        
        // Small delay for step pulse width
        delayMicroseconds(2);
        digitalWrite(PIN_X_STEP, LOW);
        
    } else {
        // Y is major axis
        digitalWrite(PIN_Y_STEP, HIGH);
        
        segment.stepsY--;
        
        // Bresenham: check if we need to step minor axis
        bresenhamError += bresenhamRatio;
        if (bresenhamError >= 1.0f) {
            digitalWrite(PIN_X_STEP, HIGH);
            segment.stepsX--;
            bresenhamError -= 1.0f;
            delayMicroseconds(2);
            digitalWrite(PIN_X_STEP, LOW);
        }
        
        delayMicroseconds(2);
        digitalWrite(PIN_Y_STEP, LOW);
    }
    
    segment.totalSteps--;
    
    // Rearm timer for next step
    if (segment.totalSteps > 0) {
        setStepInterval(segment.currentVelocity);
    } else {
        // Motion complete
        segment.active = false;
        moving = false;
        timerRunning = false;
        timerStop(timer);
        moveComplete = true;
    }
}

void StepperDriver::stepX(bool direction) {
    digitalWrite(PIN_X_DIR, direction ? HIGH : LOW);
    digitalWrite(PIN_X_STEP, HIGH);
    delayMicroseconds(2);
    digitalWrite(PIN_X_STEP, LOW);
}

void StepperDriver::stepY(bool direction) {
    digitalWrite(PIN_Y_DIR, direction ? HIGH : LOW);
    digitalWrite(PIN_Y_STEP, HIGH);
    delayMicroseconds(2);
    digitalWrite(PIN_Y_STEP, LOW);
}