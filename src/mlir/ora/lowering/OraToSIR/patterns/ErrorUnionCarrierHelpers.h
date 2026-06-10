#pragma once

#include "SIR/SIRDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include <utility>

namespace mlir
{
    namespace ora
    {
        namespace error_union_helpers
        {

            // Narrow Result/ErrorUnion runtime carrier:
            //   packed = (payload << 1) | tag
            // where tag is 0 for Ok and 1 for Err. Wider/local carriers use
            // explicit two-word layouts and must not route through this helper.
            inline ::mlir::Value narrowTagMaskConst(::mlir::OpBuilder &builder,
                                                    ::mlir::Location loc)
            {
                auto u256Type = ::sir::U256Type::get(builder.getContext());
                auto u256IntType = ::mlir::IntegerType::get(
                    builder.getContext(), 256, ::mlir::IntegerType::Unsigned);
                return builder.create<::sir::ConstOp>(
                    loc, u256Type, ::mlir::IntegerAttr::get(u256IntType, 1));
            }

            inline std::pair<::mlir::Value, ::mlir::Value>
            splitNarrowPackedCarrier(::mlir::OpBuilder &builder,
                                     ::mlir::Location loc,
                                     ::mlir::Value packed)
            {
                auto u256Type = ::sir::U256Type::get(builder.getContext());
                ::mlir::Value one = narrowTagMaskConst(builder, loc);
                ::mlir::Value tag = builder.create<::sir::AndOp>(loc, u256Type, packed, one);
                ::mlir::Value payload = builder.create<::sir::ShrOp>(loc, u256Type, one, packed);
                return {tag, payload};
            }

            inline ::mlir::Value packNarrowCarrier(::mlir::OpBuilder &builder,
                                                   ::mlir::Location loc,
                                                   ::mlir::Value tag,
                                                   ::mlir::Value payload)
            {
                auto u256Type = ::sir::U256Type::get(builder.getContext());
                ::mlir::Value one = narrowTagMaskConst(builder, loc);
                ::mlir::Value shifted = builder.create<::sir::ShlOp>(loc, u256Type, one, payload);
                return builder.create<::sir::OrOp>(loc, u256Type, shifted, tag);
            }

        } // namespace error_union_helpers
    } // namespace ora
} // namespace mlir
