#include "yul_wrapper.h"
#include <string>
#include <cstring>
#include <memory>

// Solidity includes
#include "libyul/YulStack.h"
#include "libyul/backends/evm/EVMDialect.h"
#include "libevmasm/Instruction.h"
#include "liblangutil/ErrorReporter.h"
#include "liblangutil/Scanner.h"
#include "liblangutil/EVMVersion.h"
#include "liblangutil/DebugInfoSelection.h"
#include "libsolutil/CommonData.h"
#include "libsolidity/interface/OptimiserSettings.h"

using namespace solidity;
using namespace solidity::yul;
using namespace solidity::langutil;

extern "C"
{

    YulCompileResult *yul_compile_to_bytecode(const char *yul_source)
    {
        auto result = new YulCompileResult();
        result->success = 0;
        result->bytecode = nullptr;
        result->error_message = nullptr;
        result->bytecode_length = 0;

        try
        {
            // Create Yul stack with proper constructor
            YulStack stack(
                EVMVersion::london(),                          // Use London EVM version
                std::nullopt,                                  // No EOF version
                Language::StrictAssembly,                      // Use StrictAssembly language
                solidity::frontend::OptimiserSettings::none(), // No optimization for now
                DebugInfoSelection::All());

            // Parse and compile the Yul source
            std::string source(yul_source);
            bool success = stack.parseAndAnalyze("input.yul", source);

            if (!success || stack.hasErrors())
            {
                // Compilation failed - collect error messages
                std::string error_msg = "Yul compilation failed";
                if (!stack.errors().empty())
                {
                    error_msg += ": ";
                    for (const auto &error : stack.errors())
                    {
                        error_msg += error->what();
                        error_msg += "; ";
                    }
                }
                result->error_message = new char[error_msg.length() + 1];
                std::strcpy(result->error_message, error_msg.c_str());
                return result;
            }

            // Generate bytecode
            stack.optimize();
            auto machineCode = stack.assemble(YulStack::Machine::EVM);

            if (!machineCode.bytecode)
            {
                std::string error_msg = "Failed to generate bytecode";
                result->error_message = new char[error_msg.length() + 1];
                std::strcpy(result->error_message, error_msg.c_str());
                return result;
            }

            // Convert bytecode to hex string
            std::string bytecodeHex = util::toHex(machineCode.bytecode->bytecode);

            // Copy result
            result->bytecode_length = bytecodeHex.length();
            result->bytecode = new char[result->bytecode_length + 1];
            std::strcpy(result->bytecode, bytecodeHex.c_str());
            result->success = 1;
        }
        catch (const std::exception &e)
        {
            std::string error_msg = std::string("Exception: ") + e.what();
            result->error_message = new char[error_msg.length() + 1];
            std::strcpy(result->error_message, error_msg.c_str());
        }
        catch (...)
        {
            std::string error_msg = "Unknown exception occurred";
            result->error_message = new char[error_msg.length() + 1];
            std::strcpy(result->error_message, error_msg.c_str());
        }

        return result;
    }

    void yul_free_result(YulCompileResult *result)
    {
        if (result)
        {
            delete[] result->bytecode;
            delete[] result->error_message;
            delete result;
        }
    }

    const char *yul_get_version(void)
    {
        return "Solidity Yul Compiler (via wrapper)";
    }

} // extern "C"