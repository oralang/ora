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

            void setEnableMemRefLowering(bool enable) { enableMemRefLowering = enable; }
            void setEnableStructLowering(bool enable) { enableStructLowering = enable; }
            void setEnableTensorLowering(bool enable) { enableTensorLowering = enable; }

        private:
            bool enableMemRefLowering = false;
            bool enableStructLowering = false;
            bool enableTensorLowering = false;
        };

    } // namespace ora
} // namespace mlir
