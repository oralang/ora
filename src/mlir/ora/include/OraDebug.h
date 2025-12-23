//===- OraDebug.h - Debug flag for Ora MLIR ------------------------------===//
//
// This file provides a debug flag to control debug message output.
//
//===----------------------------------------------------------------------===//

#ifndef ORA_DEBUG_H
#define ORA_DEBUG_H

#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <cstring>

namespace mlir
{
    namespace ora
    {
        /// Global debug flag - can be set via environment variable ORA_DEBUG
        /// or programmatically via setDebugEnabled()
        inline bool &getDebugFlag()
        {
            static bool debugEnabled = []()
            {
                const char *env = std::getenv("ORA_DEBUG");
                return env != nullptr && (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0);
            }();
            return debugEnabled;
        }

        /// Set debug flag programmatically
        inline void setDebugEnabled(bool enabled)
        {
            getDebugFlag() = enabled;
        }

        /// Check if debug is enabled
        inline bool isDebugEnabled()
        {
            return getDebugFlag();
        }
    } // namespace ora
} // namespace mlir

// Debug logging macro that respects the debug flag
#define ORA_DEBUG(msg)                   \
    do                                   \
    {                                    \
        if (mlir::ora::isDebugEnabled()) \
        {                                \
            llvm::errs() << msg << "\n"; \
            llvm::errs().flush();        \
        }                                \
    } while (0)

// Debug logging macro with prefix
#define ORA_DEBUG_PREFIX(prefix, msg)                             \
    do                                                            \
    {                                                             \
        if (mlir::ora::isDebugEnabled())                          \
        {                                                         \
            llvm::errs() << "[" << prefix << "] " << msg << "\n"; \
            llvm::errs().flush();                                 \
        }                                                         \
    } while (0)

#endif // ORA_DEBUG_H
