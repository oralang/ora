const ast = @import("ast/mod.zig");

pub const Options = struct {
    skip_first_runtime_parameter: bool = false,
};

pub const Entry = struct {
    parameter: ast.Parameter,
    parameter_index: usize,
    arg: ast.ExprId,
    arg_index: usize,
    comptime_index: ?usize,
    runtime_index: ?usize,
};

pub const Iterator = struct {
    parameters: []const ast.Parameter,
    args: []const ast.ExprId,
    skip_first_runtime_parameter: bool,
    parameter_index: usize = 0,
    arg_index: usize = 0,
    runtime_index: usize = 0,
    comptime_index: usize = 0,
    skipped_first_runtime: bool = false,

    pub fn next(self: *Iterator) error{InvalidGenericArgumentCount}!?Entry {
        while (self.parameter_index < self.parameters.len) {
            const parameter_index = self.parameter_index;
            const parameter = self.parameters[parameter_index];
            self.parameter_index += 1;

            if (!parameter.is_comptime and
                self.skip_first_runtime_parameter and
                !self.skipped_first_runtime and
                self.runtime_index == 0)
            {
                self.skipped_first_runtime = true;
                self.runtime_index += 1;
                continue;
            }

            if (self.arg_index >= self.args.len) return error.InvalidGenericArgumentCount;
            const arg_index = self.arg_index;
            const arg = self.args[arg_index];
            self.arg_index += 1;

            if (parameter.is_comptime) {
                const comptime_index = self.comptime_index;
                self.comptime_index += 1;
                return .{
                    .parameter = parameter,
                    .parameter_index = parameter_index,
                    .arg = arg,
                    .arg_index = arg_index,
                    .comptime_index = comptime_index,
                    .runtime_index = null,
                };
            }

            const runtime_index = self.runtime_index;
            self.runtime_index += 1;
            return .{
                .parameter = parameter,
                .parameter_index = parameter_index,
                .arg = arg,
                .arg_index = arg_index,
                .comptime_index = null,
                .runtime_index = runtime_index,
            };
        }
        return null;
    }
};

pub fn iterator(function: ast.FunctionItem, call: ast.CallExpr, options: Options) Iterator {
    return .{
        .parameters = function.parameters,
        .args = call.args,
        .skip_first_runtime_parameter = options.skip_first_runtime_parameter,
    };
}

pub fn comptimeParameterCount(function: ast.FunctionItem) usize {
    var count: usize = 0;
    for (function.parameters) |parameter| {
        if (parameter.is_comptime) count += 1;
    }
    return count;
}

pub fn runtimeParameterCount(function: ast.FunctionItem) usize {
    var count: usize = 0;
    for (function.parameters) |parameter| {
        if (!parameter.is_comptime) count += 1;
    }
    return count;
}

pub fn explicitArgumentCount(function: ast.FunctionItem, options: Options) usize {
    const comptime_count = comptimeParameterCount(function);
    const runtime_count = runtimeParameterCount(function);
    const elided_runtime: usize = if (options.skip_first_runtime_parameter and runtime_count > 0) 1 else 0;
    return comptime_count + runtime_count - elided_runtime;
}
