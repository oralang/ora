#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"
#include <optional>

namespace mlir
{
    namespace ora
    {
        mlir::UnrealizedConversionCastOp createMaterializationCastOp(
            mlir::OpBuilder &builder,
            mlir::Location loc,
            mlir::TypeRange resultTypes,
            mlir::ValueRange inputs,
            llvm::StringRef kind);

        mlir::Value createMaterializationCast(
            mlir::OpBuilder &builder,
            mlir::Location loc,
            mlir::Type type,
            mlir::Value input,
            llvm::StringRef kind);

        mlir::Value createMaterializationCast(
            mlir::OpBuilder &builder,
            mlir::Location loc,
            mlir::Type type,
            mlir::ValueRange inputs,
            llvm::StringRef kind);

        bool hasMaterializationKind(
            mlir::UnrealizedConversionCastOp castOp,
            llvm::StringRef kind);

        std::optional<mlir::Value> materializePtrCarrierFromOraValue(
            mlir::OpBuilder &builder,
            mlir::Location loc,
            mlir::Type ptrType,
            mlir::Value input);

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
