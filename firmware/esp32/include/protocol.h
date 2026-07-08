/**
 * @file protocol.h
 * @brief UART Protocol Parser for LEGO Sorter V2
 * 
 * Parses G-code style ASCII commands from Raspberry Pi.
 * Protocol spec: SM-DES-003 UART Protocol Specification
 * 
 * @version 1.0
 * @date 2026-03-10
 */

#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <Arduino.h>
#include "state_machine.h"
#include "config.h"

/**
 * @brief Parsed command structure
 */
struct Command {
    CommandType type = CommandType::NONE;
    
    // G1 parameters
    float x = NAN;          // Target X position (mm), NAN if not specified
    float y = NAN;          // Target Y position (mm), NAN if not specified
    float f = NAN;          // Feedrate (mm/min), NAN if not specified
    
    // G28 parameters
    bool homeX = false;     // Home X axis
    bool homeY = false;     // Home Y axis
    
    // Validity flag
    bool valid = false;
    uint8_t errorCode = 0;  // Error code if valid == false
};

/**
 * @brief Protocol parser class
 */
class ProtocolParser {
public:
    ProtocolParser();
    
    /**
     * @brief Parse a line of UART input
     * @param line Null-terminated string (without \n)
     * @return Parsed command structure
     */
    Command parse(const char* line);
    
    /**
     * @brief Get current sticky feedrate
     */
    float getFeedrate() const;
    
    /**
     * @brief Set sticky feedrate
     */
    void setFeedrate(float feedrate);
    
    /**
     * @brief Check if positioning mode is absolute
     */
    bool isAbsoluteMode() const;
    
    /**
     * @brief Set positioning mode
     */
    void setAbsoluteMode(bool absolute);
    
    /**
     * @brief Format position response for M114
     * @param x Current X position
     * @param y Current Y position
     * @param buffer Output buffer
     * @param bufferSize Buffer size
     */
    static void formatPositionResponse(float x, float y, char* buffer, size_t bufferSize);
    
    /**
     * @brief Format endstop response for M119
     * @param xTriggered X endstop state
     * @param yTriggered Y endstop state
     * @param buffer Output buffer
     * @param bufferSize Buffer size
     */
    static void formatEndstopResponse(bool xTriggered, bool yTriggered, char* buffer, size_t bufferSize);
    
    /**
     * @brief Format error response
     * @param errorCode Error code (1-6)
     * @param buffer Output buffer
     * @param bufferSize Buffer size
     */
    static void formatErrorResponse(uint8_t errorCode, char* buffer, size_t bufferSize);

private:
    float stickyFeedrate;       // Sticky F parameter (mm/min)
    bool absoluteMode;          // G90/G91 mode
    
    /**
     * @brief Parse a G-code command
     */
    Command parseGCode(const char* params);
    
    /**
     * @brief Parse an M-code command
     */
    Command parseMCode(const char* params);
    
    /**
     * @brief Parse parameters from a command string
     * @param str Parameter string (e.g., "X10.5 Y20.0 F3000")
     * @param cmd Command to populate
     */
    void parseParameters(const char* str, Command& cmd);
    
    /**
     * @brief Skip whitespace in string
     */
    const char* skipWhitespace(const char* str);
    
    /**
     * @brief Check if character is end of token
     */
    bool isEndOfToken(char c);
};

#endif // PROTOCOL_H