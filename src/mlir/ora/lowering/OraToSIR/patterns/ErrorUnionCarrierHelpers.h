#pragma once

#include "SIR/SIRDialect.h"
#include "OraDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/Casting.h"

#include <optional>
#include <utility>

namespace mlir
{
    namespace ora
    {
        namespace error_union_helpers
        {
            inline std::optional<unsigned> getOraBitWidth(::mlir::Type type)
            {
                if (!type)
                    return std::nullopt;
                if (llvm::isa<::mlir::NoneType>(type))
                    return 0u;
                if (auto builtinInt = llvm::dyn_cast<::mlir::IntegerType>(type))
                    return builtinInt.getWidth();
                if (auto intType = llvm::dyn_cast<::mlir::ora::IntegerType>(type))
                    return intType.getWidth();
                if (llvm::isa<::mlir::ora::BoolType>(type))
                    return 1u;
                if (llvm::isa<::mlir::ora::AddressType, ::mlir::ora::NonZeroAddressType>(type))
                    return 160u;
                if (auto enumType = llvm::dyn_cast<::mlir::ora::EnumType>(type))
                    return getOraBitWidth(enumType.getReprType());
                if (auto errType = llvm::dyn_cast<::mlir::ora::ErrorUnionType>(type))
                    return getOraBitWidth(errType.getSuccessType());
                if (auto minType = llvm::dyn_cast<::mlir::ora::MinValueType>(type))
                    return getOraBitWidth(minType.getBaseType());
                if (auto maxType = llvm::dyn_cast<::mlir::ora::MaxValueType>(type))
                    return getOraBitWidth(maxType.getBaseType());
                if (auto rangeType = llvm::dyn_cast<::mlir::ora::InRangeType>(type))
                    return getOraBitWidth(rangeType.getBaseType());
                if (auto scaledType = llvm::dyn_cast<::mlir::ora::ScaledType>(type))
                    return getOraBitWidth(scaledType.getBaseType());
                if (auto exactType = llvm::dyn_cast<::mlir::ora::ExactType>(type))
                    return getOraBitWidth(exactType.getBaseType());
                if (llvm::isa<::mlir::ora::StringType, ::mlir::ora::BytesType,
                              ::mlir::ora::StructType, ::mlir::ora::AnonymousStructType,
                              ::mlir::ora::MapType>(type))
                    return 256u;
                return std::nullopt;
            }

            inline bool isNarrowErrorUnion(::mlir::ora::ErrorUnionType type)
            {
                auto widthOpt = getOraBitWidth(type.getSuccessType());
                return widthOpt && *widthOpt <= 255;
            }

            inline bool isNarrowErrorUnionType(::mlir::Type type)
            {
                auto errType = llvm::dyn_cast<::mlir::ora::ErrorUnionType>(type);
                return errType && isNarrowErrorUnion(errType);
            }

            inline bool isScalarErrorUnionMemRefCarrier(::mlir::Type type)
            {
                auto errType = llvm::dyn_cast<::mlir::ora::ErrorUnionType>(type);
                if (!errType)
                    return false;
                ::mlir::Type successType = errType.getSuccessType();
                return llvm::isa<::mlir::IntegerType, ::mlir::ora::IntegerType,
                                 ::mlir::NoneType, ::mlir::ora::AddressType,
                                 ::mlir::ora::NonZeroAddressType>(successType);
            }

            inline ::mlir::Type getWideErrorUnionCarrierType(::mlir::MLIRContext *ctx,
                                                             ::mlir::Type successType)
            {
                if (!ctx)
                    return ::mlir::Type();
                if (llvm::isa<::sir::PtrType, ::mlir::ora::TupleType,
                              ::mlir::ora::StructType, ::mlir::ora::AnonymousStructType,
                              ::mlir::ora::StringType, ::mlir::ora::BytesType,
                              ::mlir::MemRefType, ::mlir::UnrankedMemRefType>(successType))
                    return ::sir::PtrType::get(ctx, /*addrSpace*/ 1);
                return ::sir::U256Type::get(ctx);
            }

            inline bool hasForceWideErrorUnionAttr(::mlir::Operation *op)
            {
                if (!op)
                    return false;
                if (auto attr = op->getAttrOfType<::mlir::BoolAttr>("ora.force_wide_error_union"))
                    return attr.getValue();
                if (auto func = op->getParentOfType<::mlir::func::FuncOp>())
                {
                    if (auto attr = func->getAttrOfType<::mlir::BoolAttr>("ora.force_wide_error_union"))
                        return attr.getValue();
                }
                return false;
            }

            inline bool valueHasForceWideErrorUnion(::mlir::Value value)
            {
                if (!value)
                    return false;
                if (::mlir::Operation *def = value.getDefiningOp())
                    return hasForceWideErrorUnionAttr(def);
                return false;
            }

            inline bool shouldUseWideErrorUnionCarrier(::mlir::ora::ErrorUnionType type,
                                                       ::mlir::Operation *op)
            {
                return !isNarrowErrorUnion(type) || hasForceWideErrorUnionAttr(op);
            }

            inline ::mlir::Value narrowTagMaskI256Const(::mlir::OpBuilder &builder,
                                                        ::mlir::Location loc)
            {
                auto i256Type = ::mlir::IntegerType::get(builder.getContext(), 256);
                return builder.create<::mlir::arith::ConstantOp>(
                    loc, i256Type, ::mlir::IntegerAttr::get(i256Type, 1));
            }

            inline ::mlir::Value narrowOkTagI256Const(::mlir::OpBuilder &builder,
                                                      ::mlir::Location loc)
            {
                auto i256Type = ::mlir::IntegerType::get(builder.getContext(), 256);
                return builder.create<::mlir::arith::ConstantOp>(
                    loc, i256Type, ::mlir::IntegerAttr::get(i256Type, 0));
            }

            inline ::mlir::Value packNarrowCarrierI256WithShift(::mlir::OpBuilder &builder,
                                                                ::mlir::Location loc,
                                                                ::mlir::Value tag,
                                                                ::mlir::Value payload,
                                                                ::mlir::Value one)
            {
                auto shifted = builder.create<::mlir::arith::ShLIOp>(loc, payload, one);
                return builder.create<::mlir::arith::OrIOp>(loc, shifted.getResult(), tag).getResult();
            }

            inline ::mlir::Value packNarrowCarrierI256(::mlir::OpBuilder &builder,
                                                       ::mlir::Location loc,
                                                       ::mlir::Value tag,
                                                       ::mlir::Value payload)
            {
                ::mlir::Value one = narrowTagMaskI256Const(builder, loc);
                return packNarrowCarrierI256WithShift(builder, loc, tag, payload, one);
            }

            inline ::mlir::Value narrowPackedCarrierTagI256WithMask(::mlir::OpBuilder &builder,
                                                                    ::mlir::Location loc,
                                                                    ::mlir::Value packed,
                                                                    ::mlir::Value one)
            {
                return builder.create<::mlir::arith::AndIOp>(loc, packed, one);
            }

            inline ::mlir::Value tagWordIsErrorI256WithMask(::mlir::OpBuilder &builder,
                                                            ::mlir::Location loc,
                                                            ::mlir::Value tagWord,
                                                            ::mlir::Value one)
            {
                ::mlir::Value tag = narrowPackedCarrierTagI256WithMask(builder, loc, tagWord, one);
                return builder.create<::mlir::arith::CmpIOp>(
                    loc, ::mlir::arith::CmpIPredicate::eq, tag, one);
            }

            inline ::mlir::Value narrowPackedCarrierTagI256(::mlir::OpBuilder &builder,
                                                            ::mlir::Location loc,
                                                            ::mlir::Value packed)
            {
                ::mlir::Value one = narrowTagMaskI256Const(builder, loc);
                return narrowPackedCarrierTagI256WithMask(builder, loc, packed, one);
            }

            inline std::pair<::mlir::Value, ::mlir::Value>
            splitNarrowPackedCarrierI256WithMask(::mlir::OpBuilder &builder,
                                                 ::mlir::Location loc,
                                                 ::mlir::Value packed,
                                                 ::mlir::Value one)
            {
                ::mlir::Value tag = narrowPackedCarrierTagI256WithMask(builder, loc, packed, one);
                ::mlir::Value payload = builder.create<::mlir::arith::ShRUIOp>(loc, packed, one);
                return {tag, payload};
            }

            inline std::pair<::mlir::Value, ::mlir::Value>
            splitNarrowPackedCarrierI256(::mlir::OpBuilder &builder,
                                         ::mlir::Location loc,
                                         ::mlir::Value packed)
            {
                ::mlir::Value one = narrowTagMaskI256Const(builder, loc);
                return splitNarrowPackedCarrierI256WithMask(builder, loc, packed, one);
            }

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

            inline ::mlir::Value narrowOkTagConst(::mlir::OpBuilder &builder,
                                                  ::mlir::Location loc)
            {
                auto u256Type = ::sir::U256Type::get(builder.getContext());
                auto u256IntType = ::mlir::IntegerType::get(
                    builder.getContext(), 256, ::mlir::IntegerType::Unsigned);
                return builder.create<::sir::ConstOp>(
                    loc, u256Type, ::mlir::IntegerAttr::get(u256IntType, 0));
            }

            inline ::mlir::Value narrowErrTagConst(::mlir::OpBuilder &builder,
                                                   ::mlir::Location loc)
            {
                return narrowTagMaskConst(builder, loc);
            }

            inline ::mlir::Value narrowPackedCarrierTagWithMask(::mlir::OpBuilder &builder,
                                                                ::mlir::Location loc,
                                                                ::mlir::Value packed,
                                                                ::mlir::Value one)
            {
                auto u256Type = ::sir::U256Type::get(builder.getContext());
                return builder.create<::sir::AndOp>(loc, u256Type, packed, one);
            }

            inline ::mlir::Value tagWordIsErrorWithMask(::mlir::OpBuilder &builder,
                                                        ::mlir::Location loc,
                                                        ::mlir::Value tagWord,
                                                        ::mlir::Value one)
            {
                auto u256Type = ::sir::U256Type::get(builder.getContext());
                ::mlir::Value tag = narrowPackedCarrierTagWithMask(builder, loc, tagWord, one);
                return builder.create<::sir::EqOp>(loc, u256Type, tag, one);
            }

            inline ::mlir::Value extractedTagIsErrorWithMask(::mlir::OpBuilder &builder,
                                                             ::mlir::Location loc,
                                                             ::mlir::Value tag,
                                                             ::mlir::Value one)
            {
                auto u256Type = ::sir::U256Type::get(builder.getContext());
                return builder.create<::sir::EqOp>(loc, u256Type, tag, one);
            }

            inline ::mlir::Value tagWordIsError(::mlir::OpBuilder &builder,
                                                ::mlir::Location loc,
                                                ::mlir::Value tagWord)
            {
                ::mlir::Value one = narrowTagMaskConst(builder, loc);
                return tagWordIsErrorWithMask(builder, loc, tagWord, one);
            }

            inline ::mlir::Value narrowPackedCarrierTag(::mlir::OpBuilder &builder,
                                                        ::mlir::Location loc,
                                                        ::mlir::Value packed)
            {
                ::mlir::Value one = narrowTagMaskConst(builder, loc);
                return narrowPackedCarrierTagWithMask(builder, loc, packed, one);
            }

            inline std::pair<::mlir::Value, ::mlir::Value>
            splitNarrowPackedCarrierWithMask(::mlir::OpBuilder &builder,
                                             ::mlir::Location loc,
                                             ::mlir::Value packed,
                                             ::mlir::Value one)
            {
                auto u256Type = ::sir::U256Type::get(builder.getContext());
                ::mlir::Value tag = narrowPackedCarrierTagWithMask(builder, loc, packed, one);
                ::mlir::Value payload = builder.create<::sir::ShrOp>(loc, u256Type, one, packed);
                return {tag, payload};
            }

            inline std::pair<::mlir::Value, ::mlir::Value>
            splitNarrowPackedCarrier(::mlir::OpBuilder &builder,
                                     ::mlir::Location loc,
                                     ::mlir::Value packed)
            {
                ::mlir::Value one = narrowTagMaskConst(builder, loc);
                return splitNarrowPackedCarrierWithMask(builder, loc, packed, one);
            }

            inline ::mlir::Value packNarrowCarrierWithShift(::mlir::OpBuilder &builder,
                                                            ::mlir::Location loc,
                                                            ::mlir::Value tag,
                                                            ::mlir::Value payload,
                                                            ::mlir::Value one)
            {
                auto u256Type = ::sir::U256Type::get(builder.getContext());
                ::mlir::Value shifted = builder.create<::sir::ShlOp>(loc, u256Type, one, payload);
                return builder.create<::sir::OrOp>(loc, u256Type, shifted, tag);
            }

            inline ::mlir::Value packNarrowCarrier(::mlir::OpBuilder &builder,
                                                   ::mlir::Location loc,
                                                   ::mlir::Value tag,
                                                   ::mlir::Value payload)
            {
                ::mlir::Value one = narrowTagMaskConst(builder, loc);
                return packNarrowCarrierWithShift(builder, loc, tag, payload, one);
            }

        } // namespace error_union_helpers
    } // namespace ora
} // namespace mlir
