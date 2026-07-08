/**
 * @file main.cpp
 * @brief ESP32 Firmware Main Entry Point for LEGO Sorter V2
 * 
 * Full integration of:
 * - UART protocol parser
 * - State machine
 * - Stepper driver (Timer ISR)
 * - Endstop manager (debounced)
 * - S-curve motion planner
 * - Servo gate control
 * - Homing sequence
 * - Watchdog timer
 * 
 * @version 1.0
 * @date 2026-03-10
 */

#include <Arduino.h>
#include "config.h"
#include "state_machine.h"
#include "protocol.h"
#include "stepper_driver.h"
#include "endstop.h"
#include "scurve.h"
#include "motion_planner.h"
#include "servo_gate.h"
#include "homing.h"

// Forward declarations
void processCommand(const Command& cmd);
void sendNotification(const char* notification);

// Global instances
StateMachine stateMachine;
ProtocolParser parser;
StepperDriver& stepper = StepperDriver::getInstance();
EndstopManager& endstops = EndstopManager::getInstance();
MotionPlanner& motionPlanner = MotionPlanner::getInstance();
ServoGate& servoGate = ServoGate::getInstance();
HomingManager& homing = HomingManager::getInstance();

// Serial buffer
char serialBuffer[UART_MAX_LINE + 1];
size_t bufferPos = 0;

// Command queue for G1 moves
#define CMD_QUEUE_SIZE 4
Command cmdQueue[CMD_QUEUE_SIZE];
volatile uint8_t cmdQueueHead = 0;
volatile uint8_t cmdQueueTail = 0;
volatile uint8_t cmdQueueCount = 0;

// Watchdog
volatile uint32_t lastCommandMs = 0;

/**
 * @brief Setup function - runs once at boot
 */
void setup() {
    // Initialize serial communication
    Serial.begin(UART_BAUD);
    
    // Initialize subsystems
    stepper.begin();
    endstops.begin();
    motionPlanner.begin();
    servoGate.begin();
    homing.begin();
    
    // Enable steppers (active LOW)
    stepper.enable();
    
    // Transition from BOOT to IDLE_NOT_HOMED
    stateMachine.transitionTo(State::IDLE_NOT_HOMED);
    
    // Send !READY notification
    Serial.println(NOTIFY_READY);
    
    // Initialize watchdog timestamp
    lastCommandMs = millis();
}

/**
 * @brief Main loop - runs repeatedly
 */
void loop() {
    uint32_t now = millis();
    
    // ========================================
    // 1. Poll UART for incoming commands
    // ========================================
    while (Serial.available()) {
        char c = Serial.read();
        
        if (c == '\n' || c == '\r') {
            if (bufferPos > 0) {
                // Null-terminate the buffer
                serialBuffer[bufferPos] = '\0';
                
                // Parse and process command
                Command cmd = parser.parse(serialBuffer);
                processCommand(cmd);
                
                // Reset buffer
                bufferPos = 0;
            }
        } else if (bufferPos < UART_MAX_LINE) {
            // Add to buffer (skip control chars)
            if (c >= 32) {
                serialBuffer[bufferPos++] = c;
            }
        }
    }
    
    // ========================================
    // 2. Update endstop state
    // ========================================
    endstops.update();
    
    // Check for endstop trigger during motion (E-stop)
    if (stepper.isMoving() && !homing.isHoming()) {
        // Check if either endstop was triggered unexpectedly
        if (endstops.wasTriggered()) {
            // Emergency stop
            stepper.stop();
            stepper.disable();
            stateMachine.triggerEstop();
            Serial.println(RESP_OK);
            Serial.println(NOTIFY_ESTOP);
            endstops.clearTriggered();
            return;
        }
    }
    endstops.clearTriggered();
    
    // ========================================
    // 3. Update homing state machine
    // ========================================
    if (homing.isHoming()) {
        if (homing.update()) {
            // Homing complete
            stateMachine.transitionTo(State::READY);
            stateMachine.setHomed(true);
            Serial.println(NOTIFY_HOMED);
        } else if (homing.hasError()) {
            // Homing error
            stateMachine.transitionTo(State::IDLE_NOT_HOMED);
        }
        return;  // Skip other processing during homing
    }
    
    // ========================================
    // 4. Update motion (check for completion)
    // ========================================
    if (stepper.isMoving()) {
        // Check if move just completed
        if (stepper.isMoveComplete()) {
            stepper.clearMoveComplete();
            
            // Check if there are queued commands
            if (cmdQueueCount > 0) {
                // Execute next queued G1
                Command nextCmd = cmdQueue[cmdQueueTail];
                cmdQueueTail = (cmdQueueTail + 1) % CMD_QUEUE_SIZE;
                cmdQueueCount--;
                
                // Execute the queued move
                if (!isnan(nextCmd.x)) {
                    float dx = nextCmd.x - motionPlanner.getX();
                    float dy = (isnan(nextCmd.y) ? 0 : nextCmd.y) - motionPlanner.getY();
                    stepper.startMove(dx, dy, nextCmd.f);
                }
            } else {
                // All moves complete
                if (stateMachine.getState() == State::MOVING) {
                    stateMachine.transitionTo(State::READY);
                    Serial.println(NOTIFY_MOVE_DONE);
                }
            }
        }
    }
    
    // ========================================
    // 5. Update servo gate
    // ========================================
    if (servoGate.update()) {
        // Gate operation complete
        Serial.println(NOTIFY_GATE_DONE);
    }
    
    // ========================================
    // 6. Watchdog check
    // ========================================
    if ((now - lastCommandMs) > WATCHDOG_TIMEOUT_MS) {
        // No command received for watchdog period
        // Reset to safe state
        stepper.stop();
        stepper.disable();
        stateMachine.triggerEstop();
        Serial.println(NOTIFY_ESTOP);
    }
    
    // Small delay to prevent tight loop
    delay(1);
}

/**
 * @brief Process a parsed command
 * @param cmd Parsed command structure
 */
void processCommand(const Command& cmd) {
    char response[64];
    lastCommandMs = millis();
    
    // Handle invalid commands
    if (!cmd.valid) {
        ProtocolParser::formatErrorResponse(cmd.errorCode, response, sizeof(response));
        Serial.println(response);
        return;
    }
    
    // Check if command is allowed in current state
    if (!stateMachine.canAccept(cmd.type)) {
        uint8_t errCode;
        if (stateMachine.isEstopActive()) {
            errCode = ERR_ESTOP_ACTIVE;
        } else if (cmd.type == CommandType::G1 && !stateMachine.isHomed()) {
            errCode = ERR_NOT_HOMED;
        } else {
            errCode = ERR_BUSY;
        }
        ProtocolParser::formatErrorResponse(errCode, response, sizeof(response));
        Serial.println(response);
        return;
    }
    
    // Process command based on type
    switch (cmd.type) {
        case CommandType::G1: {
            // Linear move
            float targetX = isnan(cmd.x) ? motionPlanner.getX() : cmd.x;
            float targetY = isnan(cmd.y) ? motionPlanner.getY() : cmd.y;
            float feedrate = isnan(cmd.f) ? parser.getFeedrate() : cmd.f;
            
            // Validate bounds
            if (targetX < 0 || targetX > X_MAX_MM || 
                targetY < 0 || targetY > Y_MAX_MM) {
                ProtocolParser::formatErrorResponse(ERR_OUT_OF_BOUNDS, response, sizeof(response));
                Serial.println(response);
                return;
            }
            
            // Check if already moving (queue command)
            if (stepper.isMoving()) {
                if (cmdQueueCount >= CMD_QUEUE_SIZE) {
                    // Queue full
                    ProtocolParser::formatErrorResponse(ERR_BUSY, response, sizeof(response));
                    Serial.println(response);
                    return;
                }
                // Add to queue
                cmdQueue[cmdQueueHead] = cmd;
                cmdQueueHead = (cmdQueueHead + 1) % CMD_QUEUE_SIZE;
                cmdQueueCount++;
                Serial.println(RESP_OK);
            } else {
                // Execute immediately
                PlannedMove move = motionPlanner.planAbsolute(targetX, targetY, feedrate);
                if (motionPlanner.execute(move)) {
                    stateMachine.transitionTo(State::MOVING);
                    Serial.println(RESP_OK);
                } else {
                    ProtocolParser::formatErrorResponse(ERR_OUT_OF_BOUNDS, response, sizeof(response));
                    Serial.println(response);
                }
            }
            break;
        }
            
        case CommandType::G28: {
            // Home axes
            Serial.println(RESP_OK);
            
            // Determine which axes to home
            bool homeX = cmd.homeX;
            bool homeY = cmd.homeY;
            
            // Start homing
            if (homing.start(homeX, homeY)) {
                // Homing will complete in loop()
            }
            break;
        }
            
        case CommandType::G90:
            parser.setAbsoluteMode(true);
            Serial.println(RESP_OK);
            break;
            
        case CommandType::G91:
            parser.setAbsoluteMode(false);
            Serial.println(RESP_OK);
            break;
            
        case CommandType::M3:
            // Open gate
            Serial.println(RESP_OK);
            servoGate.open();
            break;
            
        case CommandType::M5:
            // Close gate
            Serial.println(RESP_OK);
            servoGate.close();
            break;
            
        case CommandType::M17:
            stepper.enable();
            Serial.println(RESP_OK);
            break;
            
        case CommandType::M18:
            stepper.disable();
            Serial.println(RESP_OK);
            break;
            
        case CommandType::M112:
            // Emergency stop
            stepper.stop();
            stepper.disable();
            homing.cancel();
            stateMachine.triggerEstop();
            Serial.println(RESP_OK);
            Serial.println(NOTIFY_ESTOP);
            break;
            
        case CommandType::M114: {
            float x = motionPlanner.getX();
            float y = motionPlanner.getY();
            ProtocolParser::formatPositionResponse(x, y, response, sizeof(response));
            Serial.println(response);
            break;
        }
            
        case CommandType::M119: {
            bool xEndstop = endstops.isXTriggered();
            bool yEndstop = endstops.isYTriggered();
            ProtocolParser::formatEndstopResponse(xEndstop, yEndstop, response, sizeof(response));
            Serial.println(response);
            break;
        }
            
        case CommandType::M400:
            // Wait for queue empty
            while (stepper.isMoving() || cmdQueueCount > 0) {
                delay(1);
            }
            Serial.println(RESP_OK);
            break;
            
        case CommandType::M999:
            // Reset from ESTOPPED
            if (stateMachine.isEstopActive()) {
                stateMachine.transitionTo(State::IDLE_NOT_HOMED);
                stepper.enable();
            }
            Serial.println(RESP_OK);
            break;
            
        default:
            ProtocolParser::formatErrorResponse(ERR_UNKNOWN_CMD, response, sizeof(response));
            Serial.println(response);
            break;
    }
}

/**
 * @brief Send an async notification
 * @param notification Notification string
 */
void sendNotification(const char* notification) {
    Serial.println(notification);
}