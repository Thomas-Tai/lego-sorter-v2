/**
 * @file test_protocol.cpp
 * @brief Native unit tests for ProtocolParser (SM-DES-003, todolist S5-20).
 *
 * Run with: pio test -e native
 */

#include <unity.h>
#include "protocol.h"

static ProtocolParser parser;

void setUp() {
    parser = ProtocolParser();
}

void tearDown() {}

// ---------------------------------------------------------------- G1 parsing

void test_g1_full_parameters() {
    Command cmd = parser.parse("G1 X10.5 Y20.0 F1500");
    TEST_ASSERT_TRUE(cmd.valid);
    TEST_ASSERT_EQUAL(CommandType::G1, cmd.type);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 10.5f, cmd.x);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 20.0f, cmd.y);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1500.0f, cmd.f);
}

void test_g1_compact_no_spaces() {
    Command cmd = parser.parse("G1X10Y20F1500");
    TEST_ASSERT_TRUE(cmd.valid);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 10.0f, cmd.x);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 20.0f, cmd.y);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1500.0f, cmd.f);
}

void test_g1_lowercase() {
    Command cmd = parser.parse("g1 x10 y20");
    TEST_ASSERT_TRUE(cmd.valid);
    TEST_ASSERT_EQUAL(CommandType::G1, cmd.type);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 10.0f, cmd.x);
}

void test_g1_leading_whitespace() {
    Command cmd = parser.parse("   G1 X5");
    TEST_ASSERT_TRUE(cmd.valid);
    TEST_ASSERT_EQUAL(CommandType::G1, cmd.type);
}

void test_g1_default_feedrate() {
    // Fresh parser: sticky feedrate defaults to 3000 mm/min
    Command cmd = parser.parse("G1 X10");
    TEST_ASSERT_TRUE(cmd.valid);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 3000.0f, cmd.f);
}

void test_g1_sticky_feedrate() {
    parser.parse("G1 X10 F1200");
    Command cmd = parser.parse("G1 X20");
    TEST_ASSERT_TRUE(cmd.valid);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1200.0f, cmd.f);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1200.0f, parser.getFeedrate());
}

void test_g1_omitted_axes_are_nan() {
    Command cmd = parser.parse("G1 X10");
    TEST_ASSERT_TRUE(cmd.valid);
    TEST_ASSERT_TRUE(isnan(cmd.y));
}

// ------------------------------------------------------------ G1 validation

void test_g1_x_bound_max_ok() {
    Command cmd = parser.parse("G1 X355 Y0");
    TEST_ASSERT_TRUE(cmd.valid);
}

void test_g1_x_out_of_bounds() {
    Command cmd = parser.parse("G1 X355.1");
    TEST_ASSERT_FALSE(cmd.valid);
    TEST_ASSERT_EQUAL_UINT8(ERR_OUT_OF_BOUNDS, cmd.errorCode);
}

void test_g1_y_bound_max_ok() {
    Command cmd = parser.parse("G1 Y160");
    TEST_ASSERT_TRUE(cmd.valid);
}

void test_g1_y_out_of_bounds() {
    Command cmd = parser.parse("G1 Y160.1");
    TEST_ASSERT_FALSE(cmd.valid);
    TEST_ASSERT_EQUAL_UINT8(ERR_OUT_OF_BOUNDS, cmd.errorCode);
}

void test_g1_negative_x_rejected() {
    Command cmd = parser.parse("G1 X-0.5");
    TEST_ASSERT_FALSE(cmd.valid);
    TEST_ASSERT_EQUAL_UINT8(ERR_OUT_OF_BOUNDS, cmd.errorCode);
}

// -------------------------------------------------------------- G28 homing

void test_g28_no_params_homes_both() {
    Command cmd = parser.parse("G28");
    TEST_ASSERT_TRUE(cmd.valid);
    TEST_ASSERT_EQUAL(CommandType::G28, cmd.type);
    TEST_ASSERT_TRUE(cmd.homeX);
    TEST_ASSERT_TRUE(cmd.homeY);
}

void test_g28_x_only() {
    Command cmd = parser.parse("G28 X");
    TEST_ASSERT_TRUE(cmd.valid);
    TEST_ASSERT_TRUE(cmd.homeX);
    TEST_ASSERT_FALSE(cmd.homeY);
}

void test_g28_y_only() {
    Command cmd = parser.parse("G28 Y");
    TEST_ASSERT_TRUE(cmd.valid);
    TEST_ASSERT_FALSE(cmd.homeX);
    TEST_ASSERT_TRUE(cmd.homeY);
}

void test_g28_both_axes() {
    Command cmd = parser.parse("G28 X Y");
    TEST_ASSERT_TRUE(cmd.valid);
    TEST_ASSERT_TRUE(cmd.homeX);
    TEST_ASSERT_TRUE(cmd.homeY);
}

void test_g28_unknown_axis_falls_back_to_both() {
    // Coded behavior: unrecognized axis letters -> home both
    Command cmd = parser.parse("G28 Z");
    TEST_ASSERT_TRUE(cmd.valid);
    TEST_ASSERT_TRUE(cmd.homeX);
    TEST_ASSERT_TRUE(cmd.homeY);
}

// ------------------------------------------------------- G90/G91 mode flags

void test_g90_sets_absolute_mode() {
    parser.parse("G91");
    TEST_ASSERT_FALSE(parser.isAbsoluteMode());
    Command cmd = parser.parse("G90");
    TEST_ASSERT_TRUE(cmd.valid);
    TEST_ASSERT_EQUAL(CommandType::G90, cmd.type);
    TEST_ASSERT_TRUE(parser.isAbsoluteMode());
}

void test_g91_sets_relative_mode() {
    Command cmd = parser.parse("G91");
    TEST_ASSERT_TRUE(cmd.valid);
    TEST_ASSERT_EQUAL(CommandType::G91, cmd.type);
    TEST_ASSERT_FALSE(parser.isAbsoluteMode());
}

// ------------------------------------------------------------- M code table

void test_m_code_dispatch() {
    struct { const char* line; CommandType expected; } cases[] = {
        {"M3", CommandType::M3},
        {"M5", CommandType::M5},
        {"M17", CommandType::M17},
        {"M18", CommandType::M18},
        {"M112", CommandType::M112},
        {"M114", CommandType::M114},
        {"M119", CommandType::M119},
        {"M400", CommandType::M400},
        {"M999", CommandType::M999},
    };
    for (auto& c : cases) {
        Command cmd = parser.parse(c.line);
        TEST_ASSERT_TRUE_MESSAGE(cmd.valid, c.line);
        TEST_ASSERT_EQUAL_MESSAGE(c.expected, cmd.type, c.line);
    }
}

// ---------------------------------------------------------------- rejection

void test_unknown_g_code() {
    Command cmd = parser.parse("G999");
    TEST_ASSERT_FALSE(cmd.valid);
    TEST_ASSERT_EQUAL_UINT8(ERR_UNKNOWN_CMD, cmd.errorCode);
}

void test_unknown_m_code() {
    Command cmd = parser.parse("M42");
    TEST_ASSERT_FALSE(cmd.valid);
    TEST_ASSERT_EQUAL_UINT8(ERR_UNKNOWN_CMD, cmd.errorCode);
}

void test_non_gm_line_rejected() {
    Command cmd = parser.parse("X10 Y20");
    TEST_ASSERT_FALSE(cmd.valid);
    TEST_ASSERT_EQUAL_UINT8(ERR_UNKNOWN_CMD, cmd.errorCode);
}

void test_bare_g_rejected() {
    Command cmd = parser.parse("G");
    TEST_ASSERT_FALSE(cmd.valid);
    TEST_ASSERT_EQUAL_UINT8(ERR_UNKNOWN_CMD, cmd.errorCode);
}

void test_null_line_invalid_param() {
    Command cmd = parser.parse(nullptr);
    TEST_ASSERT_FALSE(cmd.valid);
    TEST_ASSERT_EQUAL_UINT8(ERR_INVALID_PARAM, cmd.errorCode);
}

void test_empty_line_invalid_param() {
    Command cmd = parser.parse("");
    TEST_ASSERT_FALSE(cmd.valid);
    TEST_ASSERT_EQUAL_UINT8(ERR_INVALID_PARAM, cmd.errorCode);
}

void test_comment_line_is_noop() {
    Command cmd = parser.parse("; just a comment");
    TEST_ASSERT_TRUE(cmd.valid);
    TEST_ASSERT_EQUAL(CommandType::NONE, cmd.type);
}

void test_paren_comment_is_noop() {
    Command cmd = parser.parse("(comment)");
    TEST_ASSERT_TRUE(cmd.valid);
    TEST_ASSERT_EQUAL(CommandType::NONE, cmd.type);
}

// ------------------------------------------------------- feedrate mutators

void test_set_feedrate_bounds() {
    parser.setFeedrate(1500.0f);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1500.0f, parser.getFeedrate());
    parser.setFeedrate(5000.0f);  // > 3000: rejected
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1500.0f, parser.getFeedrate());
    parser.setFeedrate(-1.0f);  // <= 0: rejected
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1500.0f, parser.getFeedrate());
}

// -------------------------------------------------------- response formats

void test_format_position_response() {
    char buf[64];
    ProtocolParser::formatPositionResponse(12.5f, 34.0f, buf, sizeof(buf));
    TEST_ASSERT_EQUAL_STRING("ok X:12.50 Y:34.00", buf);
}

void test_format_endstop_response() {
    char buf[64];
    ProtocolParser::formatEndstopResponse(true, false, buf, sizeof(buf));
    TEST_ASSERT_EQUAL_STRING("ok ENDSTOPS X:1 Y:0", buf);
}

void test_format_error_responses() {
    // Regression net for the SM-DES-003 code table, including the
    // historically-swapped codes 5 (ESTOP) and 6 (BUSY)
    char buf[64];
    ProtocolParser::formatErrorResponse(ERR_UNKNOWN_CMD, buf, sizeof(buf));
    TEST_ASSERT_EQUAL_STRING("error:1 Unknown command", buf);
    ProtocolParser::formatErrorResponse(ERR_OUT_OF_BOUNDS, buf, sizeof(buf));
    TEST_ASSERT_EQUAL_STRING("error:3 Move out of bounds", buf);
    ProtocolParser::formatErrorResponse(ERR_ESTOP_ACTIVE, buf, sizeof(buf));
    TEST_ASSERT_EQUAL_STRING("error:5 Emergency stop active", buf);
    ProtocolParser::formatErrorResponse(ERR_BUSY, buf, sizeof(buf));
    TEST_ASSERT_EQUAL_STRING("error:6 Busy", buf);
}

// -------------------------------------------------------------------- main

int main(int, char**) {
    UNITY_BEGIN();
    RUN_TEST(test_g1_full_parameters);
    RUN_TEST(test_g1_compact_no_spaces);
    RUN_TEST(test_g1_lowercase);
    RUN_TEST(test_g1_leading_whitespace);
    RUN_TEST(test_g1_default_feedrate);
    RUN_TEST(test_g1_sticky_feedrate);
    RUN_TEST(test_g1_omitted_axes_are_nan);
    RUN_TEST(test_g1_x_bound_max_ok);
    RUN_TEST(test_g1_x_out_of_bounds);
    RUN_TEST(test_g1_y_bound_max_ok);
    RUN_TEST(test_g1_y_out_of_bounds);
    RUN_TEST(test_g1_negative_x_rejected);
    RUN_TEST(test_g28_no_params_homes_both);
    RUN_TEST(test_g28_x_only);
    RUN_TEST(test_g28_y_only);
    RUN_TEST(test_g28_both_axes);
    RUN_TEST(test_g28_unknown_axis_falls_back_to_both);
    RUN_TEST(test_g90_sets_absolute_mode);
    RUN_TEST(test_g91_sets_relative_mode);
    RUN_TEST(test_m_code_dispatch);
    RUN_TEST(test_unknown_g_code);
    RUN_TEST(test_unknown_m_code);
    RUN_TEST(test_non_gm_line_rejected);
    RUN_TEST(test_bare_g_rejected);
    RUN_TEST(test_null_line_invalid_param);
    RUN_TEST(test_empty_line_invalid_param);
    RUN_TEST(test_comment_line_is_noop);
    RUN_TEST(test_paren_comment_is_noop);
    RUN_TEST(test_set_feedrate_bounds);
    RUN_TEST(test_format_position_response);
    RUN_TEST(test_format_endstop_response);
    RUN_TEST(test_format_error_responses);
    return UNITY_END();
}
