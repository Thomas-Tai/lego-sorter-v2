/**
 * @file servo_gate.cpp
 * @brief Servo Gate Control Implementation
 * 
 * Uses ESP32 LEDC for PWM generation.
 * Per SM-DES-007 §10: 50 Hz, 500-2500 µs pulse range.
 */

#include "servo_gate.h"

ServoGate& ServoGate::getInstance() {
    static ServoGate gate;
    return gate;
}

ServoGate::ServoGate()
    : state(GateState::CLOSED)
    , holdStartMs(0)
    , busy(false)
    , gateOpen(false)
    , pwmChannel(0)
    , currentPulse(SERVO_CLOSE_US)
{
}

void ServoGate::begin() {
    // Configure LEDC for servo PWM
    // 50 Hz = 20 ms period
    // 16-bit resolution: max duty = 65535
    // Period = 20 ms = 20000 µs
    // Duty = (pulse_us / 20000) * 65535
    
    ledcSetup(pwmChannel, SERVO_FREQ_HZ, 16);
    ledcAttachPin(PIN_SERVO, pwmChannel);
    
    // Start closed
    setServoPwm(SERVO_CLOSE_US);
    state = GateState::CLOSED;
    gateOpen = false;
    busy = false;
}

void ServoGate::setServoPwm(uint16_t pulseUs) {
    // Calculate duty cycle for 16-bit resolution
    // Duty = (pulse_us / 20000) * 65535
    uint32_t duty = (pulseUs * 65535UL) / 20000UL;
    ledcWrite(pwmChannel, duty);
    currentPulse = pulseUs;
}

void ServoGate::open() {
    if (busy) return;  // Ignore if already busy
    
    state = GateState::OPENING;
    setServoPwm(SERVO_OPEN_US);
    holdStartMs = millis();
    busy = true;
    gateOpen = true;
    state = GateState::HOLDING;
}

void ServoGate::close() {
    if (busy) return;  // Ignore if already busy
    
    state = GateState::CLOSING;
    setServoPwm(SERVO_CLOSE_US);
    holdStartMs = millis();
    busy = true;
    gateOpen = false;
    state = GateState::HOLDING;
}

bool ServoGate::update() {
    if (!busy || state != GateState::HOLDING) {
        return false;
    }
    
    // Check if hold time has elapsed
    uint32_t elapsed = millis() - holdStartMs;
    if (elapsed >= GATE_HOLD_MS) {
        // Gate operation complete
        busy = false;
        state = gateOpen ? GateState::OPEN : GateState::CLOSED;
        return true;  // Signal that !GATE_DONE should be sent
    }
    
    return false;
}

bool ServoGate::isBusy() const {
    return busy;
}

bool ServoGate::isOpen() const {
    return gateOpen;
}

GateState ServoGate::getState() const {
    return state;
}

void ServoGate::setPulse(uint16_t pulseUs) {
    // Direct pulse control for calibration
    if (pulseUs < 500) pulseUs = 500;
    if (pulseUs > 2500) pulseUs = 2500;
    setServoPwm(pulseUs);
}