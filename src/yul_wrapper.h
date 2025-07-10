/**
 * @file yul_wrapper.h
 * @brief C wrapper for Solidity Yul compiler
 *
 * This header provides a C interface to the Solidity Yul compiler,
 * enabling compilation of Yul intermediate language to EVM bytecode.
 * The wrapper handles memory management and error reporting.
 */

#ifndef YUL_WRAPPER_H
#define YUL_WRAPPER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Result structure for Yul compilation
    typedef struct
    {
        int success;
        char *bytecode;
        char *error_message;
        size_t bytecode_length;
    } YulCompileResult;

    // Compile Yul source code to EVM bytecode
    YulCompileResult *yul_compile_to_bytecode(const char *yul_source);

    // Free the result structure
    void yul_free_result(YulCompileResult *result);

    // Get version information
    const char *yul_get_version(void);

#ifdef __cplusplus
}
#endif

#endif // YUL_WRAPPER_H