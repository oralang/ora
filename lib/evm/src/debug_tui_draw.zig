//! Pure drawing primitives for the EVM debugger TUI.
//!
//! Holds the small style functions, segment helpers, and border
//! glyphs that don't reach into Ui state — extracted from
//! debug_tui.zig so the rest of the file isn't padded with ~70 LOC
//! of color constants. The bigger gutter / SIR-range painters live
//! in debug_tui.zig for now since they touch a lot of Ui internals
//! and pub-ifying them all wouldn't pay for itself.

const std = @import("std");
const vaxis = @import("vaxis");

pub const Style = vaxis.Style;
pub const Color = vaxis.Color;
pub const Segment = vaxis.Segment;
pub const Window = vaxis.Window;

pub const ascii_border_glyphs: [6][]const u8 = .{ "+", "-", "+", "|", "+", "+" };

pub fn seg(text: []const u8, style: Style) Segment {
    return .{ .text = text, .style = style };
}

pub fn drawSegments(win: Window, col: u16, row: u16, segments: []const Segment) void {
    _ = win.print(segments, .{ .col_offset = col, .row_offset = row, .wrap = .none });
}

pub fn style_header_title() Style {
    return .{ .fg = Color.rgbFromUint(0x191F24), .bg = Color.rgbFromUint(0xE8EFF6), .bold = true };
}

pub fn style_header_meta() Style {
    return .{ .fg = Color.rgbFromUint(0xD3DBE3), .bg = Color.rgbFromUint(0x1A1D21) };
}

pub fn style_footer_note() Style {
    return .{ .fg = Color.rgbFromUint(0xA8B0B8), .bg = Color.rgbFromUint(0x1A1D21) };
}

pub fn style_border() Style {
    return .{ .fg = Color.rgbFromUint(0x78838E) };
}

pub fn style_title() Style {
    return .{ .fg = Color.rgbFromUint(0xDEE4EB), .bold = true };
}

pub fn style_text() Style {
    return .{ .fg = Color.rgbFromUint(0xD6DCE2) };
}

pub fn style_emphasis() Style {
    return .{ .fg = Color.rgbFromUint(0xF5F7FA), .bold = true };
}

pub fn style_changed() Style {
    return .{ .fg = Color.rgbFromUint(0xFFD666), .bold = true };
}

pub fn style_guard() Style {
    return .{ .fg = Color.rgbFromUint(0xFFAD66), .bold = true };
}

pub fn style_hint() Style {
    return .{ .fg = Color.rgbFromUint(0x969EA6), .italic = true };
}

pub fn style_muted() Style {
    return .{ .fg = Color.rgbFromUint(0xC0C7CF), .dim = true };
}

pub fn style_dead() Style {
    return .{ .fg = Color.rgbFromUint(0x8B929A), .dim = true, .italic = true };
}

pub fn style_tab_active() Style {
    return .{ .fg = Color.rgbFromUint(0xEEF2F8), .bold = true, .ul_style = .single };
}

pub fn style_tab_inactive() Style {
    return .{ .fg = Color.rgbFromUint(0xA0AAB4) };
}

pub fn style_error() Style {
    return .{ .fg = Color.rgbFromUint(0xFF6B6B), .bold = true };
}

pub fn style_command_bg() Style {
    return .{ .fg = Color.rgbFromUint(0xDDE5ED), .bg = Color.rgbFromUint(0x111417) };
}

pub fn style_command() Style {
    return .{ .fg = Color.rgbFromUint(0xE7EDF4), .bg = Color.rgbFromUint(0x111417), .bold = true };
}
