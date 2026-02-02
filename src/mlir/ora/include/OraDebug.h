//===- OraDebug.h - Debug flag for Ora MLIR ------------------------------===//
//
// This file provides a debug flag to control debug message output.
//
//===----------------------------------------------------------------------===//

#ifndef ORA_DEBUG_H
#define ORA_DEBUG_H

#include "llvm/Support/raw_ostream.h"

namespace mlir
{
    namespace ora
    {
        /// Global debug flag - can be set via environment variable ORA_DEBUG
        /// or programmatically via setDebugEnabled()
        bool &getDebugFlag();

        /// Set debug flag programmatically
        void setDebugEnabled(bool enabled);

        /// Check if debug is enabled
        bool isDebugEnabled();
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
