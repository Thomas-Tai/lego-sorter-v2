/**
 * @file protocol.cpp
 * @brief UART Protocol Parser Implementation
 * 
 * Parses G-code style commands per SM-DES-003:
 *   G1 Xnnn Ynnn Fnnn  - Linear move
 *   G28 [X] [Y]        - Home axes
 *   G90                - Absolute mode
 *   G91                - Relative mode
 *   M3                 - Open gate
 *   M5                 - Close gate
 *   M17                - Enable steppers
 *   M18                - Disable steppers
 *   M112               - Emergency stop
 *   M114               - Report position
 *   M119               - Report endstop status
 *   M400               - Wait for queue empty
 *   M999               - Reset from ESTOP
 */

#include "protocol.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

ProtocolParser::ProtocolParser()
    : stickyFeedrate(3000.0f)  // Default: 3000 mm/min = 50 mm/s
    , absoluteMode(true) {
}

Command ProtocolParser::parse(const char* line) {
    Command cmd;
    cmd.valid = false;
    cmd.errorCode = 0;
    
    if (line == nullptr || *line == '\0') {
        cmd.errorCode = ERR_INVALID_PARAM;
        return cmd;
    }
    
    // Skip leading whitespace
    line = skipWhitespace(line);
    
    // Skip comments (semicolon or parenthesis)
    if (*line == ';' || *line == '(') {
        cmd.valid = true;
        cmd.type = CommandType::NONE;
        return cmd;
    }
    
    // Check for G or M code
    char codeType = toupper(*line);
    
    if (codeType == 'G') {
        cmd = parseGCode(line + 1);
    } else if (codeType == 'M') {
        cmd = parseMCode(line + 1);
    } else {
        cmd.errorCode = ERR_UNKNOWN_CMD;
        return cmd;
    }
    
    return cmd;
}

float ProtocolParser::getFeedrate() const {
    return stickyFeedrate;
}

void ProtocolParser::setFeedrate(float feedrate) {
    if (feedrate > 0 && feedrate <= 3000.0f) {
        stickyFeedrate = feedrate;
    }
}

bool ProtocolParser::isAbsoluteMode() const {
    return absoluteMode;
}

void ProtocolParser::setAbsoluteMode(bool absolute) {
    absoluteMode = absolute;
}

void ProtocolParser::formatPositionResponse(float x, float y, char* buffer, size_t bufferSize) {
    snprintf(buffer, bufferSize, "ok X:%.2f Y:%.2f", x, y);
}

void ProtocolParser::formatEndstopResponse(bool xTriggered, bool yTriggered, char* buffer, size_t bufferSize) {
    snprintf(buffer, bufferSize, "ok ENDSTOPS X:%d Y:%d",
             xTriggered ? 1 : 0,
             yTriggered ? 1 : 0);
}

void ProtocolParser::formatErrorResponse(uint8_t errorCode, char* buffer, size_t bufferSize) {
    const char* msg = "";
    switch (errorCode) {
        case ERR_UNKNOWN_CMD:    msg = "Unknown command"; break;
        case ERR_INVALID_PARAM:  msg = "Invalid parameter"; break;
        case ERR_OUT_OF_BOUNDS:  msg = "Move out of bounds"; break;
        case ERR_NOT_HOMED:      msg = "Not homed"; break;
        case ERR_BUSY:           msg = "Busy"; break;
        case ERR_ESTOP_ACTIVE:   msg = "Emergency stop active"; break;
        default:                 msg = "Unknown error"; break;
    }
    snprintf(buffer, bufferSize, "error:%d %s", errorCode, msg);
}

Command ProtocolParser::parseGCode(const char* params) {
    Command cmd;
    cmd.valid = false;
    
    // Extract G code number
    params = skipWhitespace(params);
    
    if (!isdigit(*params)) {
        cmd.errorCode = ERR_UNKNOWN_CMD;
        return cmd;
    }
    
    int codeNum = atoi(params);
    
    // Advance past the number
    while (isdigit(*params)) params++;
    
    switch (codeNum) {
        case 1:  // G1 - Linear move
            cmd.type = CommandType::G1;
            parseParameters(params, cmd);
            // Apply sticky feedrate if not specified
            if (isnan(cmd.f)) {
                cmd.f = stickyFeedrate;
            } else {
                // Update sticky feedrate
                stickyFeedrate = cmd.f;
            }
            // Validate coordinates
            if (!isnan(cmd.x) && (cmd.x < 0 || cmd.x > X_MAX_MM)) {
                cmd.errorCode = ERR_OUT_OF_BOUNDS;
                return cmd;
            }
            if (!isnan(cmd.y) && (cmd.y < 0 || cmd.y > Y_MAX_MM)) {
                cmd.errorCode = ERR_OUT_OF_BOUNDS;
                return cmd;
            }
            cmd.valid = true;
            break;
            
        case 28:  // G28 - Home
            cmd.type = CommandType::G28;
            params = skipWhitespace(params);
            // Check for X and/or Y
            if (*params == '\0' || *params == ';') {
                // No parameters: home both
                cmd.homeX = true;
                cmd.homeY = true;
            } else {
                // Parse X and/or Y flags
                while (*params && *params != ';') {
                    char c = toupper(*params);
                    if (c == 'X') {
                        cmd.homeX = true;
                    } else if (c == 'Y') {
                        cmd.homeY = true;
                    }
                    params++;
                }
                // If neither specified, home both
                if (!cmd.homeX && !cmd.homeY) {
                    cmd.homeX = true;
                    cmd.homeY = true;
                }
            }
            cmd.valid = true;
            break;
            
        case 90:  // G90 - Absolute mode
            cmd.type = CommandType::G90;
            absoluteMode = true;
            cmd.valid = true;
            break;
            
        case 91:  // G91 - Relative mode
            cmd.type = CommandType::G91;
            absoluteMode = false;
            cmd.valid = true;
            break;
            
        default:
            cmd.errorCode = ERR_UNKNOWN_CMD;
            break;
    }
    
    return cmd;
}

Command ProtocolParser::parseMCode(const char* params) {
    Command cmd;
    cmd.valid = false;
    
    // Extract M code number
    params = skipWhitespace(params);
    
    if (!isdigit(*params)) {
        cmd.errorCode = ERR_UNKNOWN_CMD;
        return cmd;
    }
    
    int codeNum = atoi(params);
    
    switch (codeNum) {
        case 3:   // M3 - Open gate
            cmd.type = CommandType::M3;
            cmd.valid = true;
            break;
            
        case 5:   // M5 - Close gate
            cmd.type = CommandType::M5;
            cmd.valid = true;
            break;
            
        case 17:  // M17 - Enable steppers
            cmd.type = CommandType::M17;
            cmd.valid = true;
            break;
            
        case 18:  // M18 - Disable steppers
            cmd.type = CommandType::M18;
            cmd.valid = true;
            break;
            
        case 112:  // M112 - Emergency stop
            cmd.type = CommandType::M112;
            cmd.valid = true;
            break;
            
        case 114:  // M114 - Report position
            cmd.type = CommandType::M114;
            cmd.valid = true;
            break;
            
        case 119:  // M119 - Report endstop status
            cmd.type = CommandType::M119;
            cmd.valid = true;
            break;
            
        case 400:  // M400 - Wait for queue empty
            cmd.type = CommandType::M400;
            cmd.valid = true;
            break;
            
        case 999:  // M999 - Reset from ESTOP
            cmd.type = CommandType::M999;
            cmd.valid = true;
            break;
            
        default:
            cmd.errorCode = ERR_UNKNOWN_CMD;
            break;
    }
    
    return cmd;
}

void ProtocolParser::parseParameters(const char* str, Command& cmd) {
    str = skipWhitespace(str);
    
    while (*str && *str != ';') {
        char param = toupper(*str);
        
        if (param == 'X' || param == 'Y' || param == 'F') {
            str++;
            str = skipWhitespace(str);
            
            // Parse float value
            char* end;
            float value = strtof(str, &end);
            
            if (end != str) {
                // Successfully parsed
                switch (param) {
                    case 'X': cmd.x = value; break;
                    case 'Y': cmd.y = value; break;
                    case 'F': cmd.f = value; break;
                }
                str = end;
            }
        } else {
            str++;
        }
        
        str = skipWhitespace(str);
    }
}

const char* ProtocolParser::skipWhitespace(const char* str) {
    while (*str && (*str == ' ' || *str == '\t')) {
        str++;
    }
    return str;
}

bool ProtocolParser::isEndOfToken(char c) {
    return c == '\0' || c == ' ' || c == '\t' || c == ';' || c == '(';
}