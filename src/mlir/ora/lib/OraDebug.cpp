//===- OraDebug.cpp - Debug flag for Ora MLIR ----------------------------===//
//
// Provides a single debug flag definition shared across translation units.
//
//===----------------------------------------------------------------------===//

#include "OraDebug.h"
#include <cstdlib>
#include <cstring>

namespace mlir
{
    namespace ora
    {
        static bool debugEnabled = []()
        {
            const char *env = std::getenv("ORA_DEBUG");
            return env != nullptr && (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0);
        }();

        bool &getDebugFlag()
        {
            return debugEnabled;
        }

        void setDebugEnabled(bool enabled)
        {
            getDebugFlag() = enabled;
        }

        bool isDebugEnabled()
        {
            return getDebugFlag();
        }
    } // namespace ora
} // namespace mlir
