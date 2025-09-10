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
///     language constructs in MLIR. Currently supports:
///     
///     - Contract declarations (ora.contract)
///     - Storage variable declarations (ora.global)  
///     - Storage operations (ora.sload, ora.sstore)
///     
///     This is a minimal dialect definition focusing on the core operations
///     that are currently implemented in the Ora compiler.
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
