/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: OraDialect.td                                                        *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace ora {

/// Ora smart contract language dialect
/// The Ora dialect provides operations for representing Ora smart contract
///     language constructs in MLIR. Supports comprehensive Ora language features:
///     
///     - Contract declarations (ora.contract)
///     - Storage operations (ora.global, ora.sload, ora.sstore)
///     - Memory operations (ora.mload, ora.mstore)
///     - Transient storage (ora.tload, ora.tstore)
///     - Constants (ora.string.constant, ora.address.constant, ora.hex.constant, ora.binary.constant)
///     - Arithmetic operations (ora.power)
///     - Type conversions (ora.cast)
///     - Destructuring (ora.destructure)
///     - Enum operations (ora.enum.decl, ora.enum_constant)
///     - Struct operations (ora.struct.decl, ora.struct_field_store, ora.struct_instantiate, ora.struct_init)
///     - Map operations (ora.map_get, ora.map_store)
///     - Formal verification (ora.requires, ora.ensures, ora.invariant, ora.old)
///     - Control flow (ora.if, ora.while, ora.for, ora.switch, ora.break, ora.continue, ora.return)
///     - Financial operations (ora.move)
///     - Event logging (ora.log)
///     - Locking (ora.lock, ora.unlock)
///     - Error handling (ora.try)
///     - Variable declarations (ora.const, ora.immutable)
///     
///     This dialect provides complete coverage of the Ora language grammar.
class OraDialect : public ::mlir::Dialect {
  explicit OraDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~OraDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("ora");
  }

  /// Parse an attribute registered to this dialect.
  ::mlir::Attribute parseAttribute(::mlir::DialectAsmParser &parser,
                                   ::mlir::Type type) const override;

  /// Print an attribute registered to this dialect.
  void printAttribute(::mlir::Attribute attr,
                      ::mlir::DialectAsmPrinter &os) const override;

  /// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;
};
} // namespace ora
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ora::OraDialect)
