#pragma once

#include "mlir/Transforms/DialectConversion.h"

namespace mlir
{
    namespace ora
    {

        class OraToSIRTypeConverter : public mlir::TypeConverter
        {
        public:
            OraToSIRTypeConverter();
        };

    } // namespace ora
} // namespace mlir