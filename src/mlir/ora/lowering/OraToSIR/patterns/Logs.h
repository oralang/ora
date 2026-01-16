#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "OraDialect.h"
#include "SIR/SIRDialect.h"

namespace mlir
{
    namespace ora
    {
        class OraToSIRTypeConverter;

        class ConvertLogOp : public OpConversionPattern<ora::LogOp>
        {
        public:
            using OpConversionPattern::OpConversionPattern;

            LogicalResult matchAndRewrite(
                ora::LogOp op,
                typename ora::LogOp::Adaptor adaptor,
                ConversionPatternRewriter &rewriter) const override;
        };
    } // namespace ora
} // namespace mlir
