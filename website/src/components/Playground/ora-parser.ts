// Simple Ora Parser for Playground
// Based on the Ora grammar and codebase structure

export interface Token {
  type: TokenType;
  value: string;
  line: number;
  column: number;
}

export enum TokenType {
  // End of file
  EOF = 'EOF',
  
  // Keywords
  CONTRACT = 'CONTRACT',
  PUB = 'PUB',
  FN = 'FN',
  LET = 'LET',
  VAR = 'VAR',
  CONST = 'CONST',
  STORAGE = 'STORAGE',
  MEMORY = 'MEMORY',
  RETURN = 'RETURN',
  IF = 'IF',
  ELSE = 'ELSE',
  WHILE = 'WHILE',
  FOR = 'FOR',
  STRUCT = 'STRUCT',
  ENUM = 'ENUM',
  LOG = 'LOG',
  ERROR = 'ERROR',
  TRY = 'TRY',
  CATCH = 'CATCH',
  SWITCH = 'SWITCH',
  REQUIRES = 'REQUIRES',
  ENSURES = 'ENSURES',
  
  // Types
  U8 = 'U8', U16 = 'U16', U32 = 'U32', U64 = 'U64', U128 = 'U128', U256 = 'U256',
  I8 = 'I8', I16 = 'I16', I32 = 'I32', I64 = 'I64', I128 = 'I128', I256 = 'I256',
  BOOL = 'BOOL',
  ADDRESS = 'ADDRESS',
  STRING = 'STRING',
  BYTES = 'BYTES',
  VOID = 'VOID',
  
  // Literals
  IDENTIFIER = 'IDENTIFIER',
  STRING_LITERAL = 'STRING_LITERAL',
  INTEGER_LITERAL = 'INTEGER_LITERAL',
  HEX_LITERAL = 'HEX_LITERAL',
  BINARY_LITERAL = 'BINARY_LITERAL',
  ADDRESS_LITERAL = 'ADDRESS_LITERAL',
  TRUE = 'TRUE',
  FALSE = 'FALSE',
  
  // Operators
  PLUS = 'PLUS', MINUS = 'MINUS', STAR = 'STAR', SLASH = 'SLASH', PERCENT = 'PERCENT',
  EQUAL = 'EQUAL', EQUAL_EQUAL = 'EQUAL_EQUAL', BANG_EQUAL = 'BANG_EQUAL',
  LESS = 'LESS', LESS_EQUAL = 'LESS_EQUAL', GREATER = 'GREATER', GREATER_EQUAL = 'GREATER_EQUAL',
  BANG = 'BANG', AMPERSAND_AMPERSAND = 'AMPERSAND_AMPERSAND', PIPE_PIPE = 'PIPE_PIPE',
  ARROW = 'ARROW',
  
  // Delimiters
  LEFT_PAREN = 'LEFT_PAREN', RIGHT_PAREN = 'RIGHT_PAREN',
  LEFT_BRACE = 'LEFT_BRACE', RIGHT_BRACE = 'RIGHT_BRACE',
  LEFT_BRACKET = 'LEFT_BRACKET', RIGHT_BRACKET = 'RIGHT_BRACKET',
  COMMA = 'COMMA', SEMICOLON = 'SEMICOLON', COLON = 'COLON', DOT = 'DOT',
  AT = 'AT',
}

const KEYWORDS: Record<string, TokenType> = {
  'contract': TokenType.CONTRACT,
  'pub': TokenType.PUB,
  'fn': TokenType.FN,
  'let': TokenType.LET,
  'var': TokenType.VAR,
  'const': TokenType.CONST,
  'storage': TokenType.STORAGE,
  'memory': TokenType.MEMORY,
  'return': TokenType.RETURN,
  'if': TokenType.IF,
  'else': TokenType.ELSE,
  'while': TokenType.WHILE,
  'for': TokenType.FOR,
  'struct': TokenType.STRUCT,
  'enum': TokenType.ENUM,
  'log': TokenType.LOG,
  'error': TokenType.ERROR,
  'try': TokenType.TRY,
  'catch': TokenType.CATCH,
  'switch': TokenType.SWITCH,
  'requires': TokenType.REQUIRES,
  'ensures': TokenType.ENSURES,
  'u8': TokenType.U8, 'u16': TokenType.U16, 'u32': TokenType.U32,
  'u64': TokenType.U64, 'u128': TokenType.U128, 'u256': TokenType.U256,
  'i8': TokenType.I8, 'i16': TokenType.I16, 'i32': TokenType.I32,
  'i64': TokenType.I64, 'i128': TokenType.I128, 'i256': TokenType.I256,
  'bool': TokenType.BOOL,
  'address': TokenType.ADDRESS,
  'string': TokenType.STRING,
  'bytes': TokenType.BYTES,
  'void': TokenType.VOID,
  'true': TokenType.TRUE,
  'false': TokenType.FALSE,
};

export class Lexer {
  private source: string;
  private tokens: Token[] = [];
  private start: number = 0;
  private current: number = 0;
  private line: number = 1;
  private column: number = 1;
  private errors: string[] = [];

  constructor(source: string) {
    this.source = source;
  }

  scanTokens(): { tokens: Token[]; errors: string[] } {
    while (!this.isAtEnd()) {
      this.start = this.current;
      this.scanToken();
    }

    this.tokens.push({
      type: TokenType.EOF,
      value: '',
      line: this.line,
      column: this.column,
    });

    return { tokens: this.tokens, errors: this.errors };
  }

  private isAtEnd(): boolean {
    return this.current >= this.source.length;
  }

  private scanToken(): void {
    const c = this.advance();

    switch (c) {
      case '(': this.addToken(TokenType.LEFT_PAREN); break;
      case ')': this.addToken(TokenType.RIGHT_PAREN); break;
      case '{': this.addToken(TokenType.LEFT_BRACE); break;
      case '}': this.addToken(TokenType.RIGHT_BRACE); break;
      case '[': this.addToken(TokenType.LEFT_BRACKET); break;
      case ']': this.addToken(TokenType.RIGHT_BRACKET); break;
      case ',': this.addToken(TokenType.COMMA); break;
      case ';': this.addToken(TokenType.SEMICOLON); break;
      case ':': this.addToken(TokenType.COLON); break;
      case '.': this.addToken(TokenType.DOT); break;
      case '@': this.addToken(TokenType.AT); break;
      case '+': this.addToken(TokenType.PLUS); break;
      case '-': 
        if (this.match('>')) {
          this.addToken(TokenType.ARROW);
        } else {
          this.addToken(TokenType.MINUS);
        }
        break;
      case '*': this.addToken(TokenType.STAR); break;
      case '/': 
        if (this.match('/')) {
          // Line comment
          while (this.peek() !== '\n' && !this.isAtEnd()) this.advance();
        } else if (this.match('*')) {
          // Block comment
          while (!this.isAtEnd()) {
            if (this.peek() === '*' && this.peekNext() === '/') {
              this.advance();
              this.advance();
              break;
            }
            this.advance();
          }
        } else {
          this.addToken(TokenType.SLASH);
        }
        break;
      case '%': this.addToken(TokenType.PERCENT); break;
      case '!': 
        this.addToken(this.match('=') ? TokenType.BANG_EQUAL : TokenType.BANG);
        break;
      case '=': 
        this.addToken(this.match('=') ? TokenType.EQUAL_EQUAL : TokenType.EQUAL);
        break;
      case '<': 
        this.addToken(this.match('=') ? TokenType.LESS_EQUAL : TokenType.LESS);
        break;
      case '>': 
        this.addToken(this.match('=') ? TokenType.GREATER_EQUAL : TokenType.GREATER);
        break;
      case '&': 
        if (this.match('&')) {
          this.addToken(TokenType.AMPERSAND_AMPERSAND);
        }
        break;
      case '|': 
        if (this.match('|')) {
          this.addToken(TokenType.PIPE_PIPE);
        }
        break;
      case '"': this.string(); break;
      case '0':
        if (this.match('x') || this.match('X')) {
          this.hex();
        } else if (this.match('b') || this.match('B')) {
          this.binary();
        } else {
          this.number();
        }
        break;
      case ' ':
      case '\r':
      case '\t':
        break;
      case '\n':
        this.line++;
        this.column = 1;
        break;
      default:
        if (this.isDigit(c)) {
          this.number();
        } else if (this.isAlpha(c)) {
          this.identifier();
        } else {
          this.errors.push(`Unexpected character '${c}' at line ${this.line}, column ${this.column}`);
        }
    }
  }

  private advance(): string {
    this.column++;
    return this.source[this.current++];
  }

  private match(expected: string): boolean {
    if (this.isAtEnd()) return false;
    if (this.source[this.current] !== expected) return false;
    this.current++;
    this.column++;
    return true;
  }

  private peek(): string {
    if (this.isAtEnd()) return '\0';
    return this.source[this.current];
  }

  private peekNext(): string {
    if (this.current + 1 >= this.source.length) return '\0';
    return this.source[this.current + 1];
  }

  private string(): void {
    while (this.peek() !== '"' && !this.isAtEnd()) {
      if (this.peek() === '\n') this.line++;
      if (this.peek() === '\\') this.advance(); // Skip escape sequences
      this.advance();
    }

    if (this.isAtEnd()) {
      this.errors.push(`Unterminated string at line ${this.line}`);
      return;
    }

    this.advance(); // Closing quote
    const value = this.source.substring(this.start + 1, this.current - 1);
    this.addToken(TokenType.STRING_LITERAL, value);
  }

  private number(): void {
    while (this.isDigit(this.peek())) this.advance();
    if (this.peek() === '.' && this.isDigit(this.peekNext())) {
      this.advance();
      while (this.isDigit(this.peek())) this.advance();
    }
    const value = this.source.substring(this.start, this.current);
    this.addToken(TokenType.INTEGER_LITERAL, value);
  }

  private hex(): void {
    while (this.isHexDigit(this.peek())) this.advance();
    const value = this.source.substring(this.start, this.current);
    this.addToken(TokenType.HEX_LITERAL, value);
  }

  private binary(): void {
    while (this.peek() === '0' || this.peek() === '1' || this.peek() === '_') this.advance();
    const value = this.source.substring(this.start, this.current);
    this.addToken(TokenType.BINARY_LITERAL, value);
  }

  private identifier(): void {
    while (this.isAlphaNumeric(this.peek())) this.advance();
    const text = this.source.substring(this.start, this.current);
    const type = KEYWORDS[text] || TokenType.IDENTIFIER;
    this.addToken(type, text);
  }

  private isDigit(c: string): boolean {
    return c >= '0' && c <= '9';
  }

  private isAlpha(c: string): boolean {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c === '_';
  }

  private isAlphaNumeric(c: string): boolean {
    return this.isAlpha(c) || this.isDigit(c);
  }

  private isHexDigit(c: string): boolean {
    return this.isDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
  }

  private addToken(type: TokenType, value: string = ''): void {
    const text = value || this.source.substring(this.start, this.current);
    this.tokens.push({
      type,
      value: text,
      line: this.line,
      column: this.column - text.length,
    });
  }
}

// Simple Parser
export interface ParseError {
  message: string;
  line: number;
  column: number;
}

export class Parser {
  private tokens: Token[];
  private current: number = 0;
  private errors: ParseError[] = [];

  constructor(tokens: Token[]) {
    this.tokens = tokens;
  }

  parse(): { success: boolean; errors: ParseError[] } {
    try {
      while (!this.isAtEnd()) {
        this.parseTopLevel();
      }
      return { success: this.errors.length === 0, errors: this.errors };
    } catch (e) {
      return { success: false, errors: this.errors };
    }
  }

  private parseTopLevel(): void {
    if (this.match(TokenType.CONTRACT)) {
      this.parseContract();
    } else if (this.match(TokenType.FN) || this.match(TokenType.PUB)) {
      if (this.previous().type === TokenType.PUB) {
        this.consume(TokenType.FN, "Expected 'fn' after 'pub'");
      }
      this.parseFunction();
    } else if (this.match(TokenType.STRUCT)) {
      this.parseStruct();
    } else if (this.match(TokenType.ENUM)) {
      this.parseEnum();
    } else if (!this.isAtEnd() && this.peek().type !== TokenType.EOF) {
      this.error("Expected contract, function, struct, or enum");
    }
  }

  private parseContract(): void {
    this.consume(TokenType.IDENTIFIER, "Expected contract name");
    this.consume(TokenType.LEFT_BRACE, "Expected '{' after contract name");
    
    while (!this.check(TokenType.RIGHT_BRACE) && !this.isAtEnd()) {
      if (this.match(TokenType.PUB) || this.match(TokenType.FN)) {
        if (this.previous().type === TokenType.PUB) {
          this.consume(TokenType.FN, "Expected 'fn' after 'pub'");
        }
        this.parseFunction();
      } else if (this.match(TokenType.STRUCT)) {
        this.parseStruct();
      } else if (this.match(TokenType.ENUM)) {
        this.parseEnum();
      } else if (this.match(TokenType.LET) || this.match(TokenType.VAR) || 
                 this.match(TokenType.CONST) || this.match(TokenType.STORAGE) || 
                 this.match(TokenType.MEMORY)) {
        this.parseVariableDecl();
      } else if (this.check(TokenType.RIGHT_BRACE)) {
        break;
      } else {
        this.error("Expected contract member");
        this.synchronize();
      }
    }
    
    this.consume(TokenType.RIGHT_BRACE, "Expected '}' after contract body");
  }

  private parseFunction(): void {
    this.consume(TokenType.IDENTIFIER, "Expected function name");
    
    // Parameters
    this.consume(TokenType.LEFT_PAREN, "Expected '(' after function name");
    if (!this.check(TokenType.RIGHT_PAREN)) {
      do {
        this.parseParameter();
        if (!this.match(TokenType.COMMA)) break;
      } while (!this.check(TokenType.RIGHT_PAREN));
    }
    this.consume(TokenType.RIGHT_PAREN, "Expected ')' after parameters");
    
    // Return type
    if (this.match(TokenType.ARROW)) {
      this.parseType();
    }
    
    // Body
    this.consume(TokenType.LEFT_BRACE, "Expected '{' after function signature");
    this.parseBlock();
    this.consume(TokenType.RIGHT_BRACE, "Expected '}' after function body");
  }

  private parseParameter(): void {
    this.consume(TokenType.IDENTIFIER, "Expected parameter name");
    this.consume(TokenType.COLON, "Expected ':' after parameter name");
    this.parseType();
  }

  private parseType(): void {
    if (this.match(TokenType.U8) || this.match(TokenType.U16) || 
        this.match(TokenType.U32) || this.match(TokenType.U64) || 
        this.match(TokenType.U128) || this.match(TokenType.U256) ||
        this.match(TokenType.I8) || this.match(TokenType.I16) || 
        this.match(TokenType.I32) || this.match(TokenType.I64) || 
        this.match(TokenType.I128) || this.match(TokenType.I256) ||
        this.match(TokenType.BOOL) || this.match(TokenType.ADDRESS) || 
        this.match(TokenType.STRING) || this.match(TokenType.BYTES) ||
        this.match(TokenType.VOID)) {
      // Primitive type
    } else if (this.match(TokenType.IDENTIFIER)) {
      // Custom type
    } else {
      this.error("Expected type");
    }
  }

  private parseVariableDecl(): void {
    this.consume(TokenType.IDENTIFIER, "Expected variable name");
    if (this.match(TokenType.COLON)) {
      this.parseType();
    }
    if (this.match(TokenType.EQUAL)) {
      this.parseExpression();
    }
    this.consume(TokenType.SEMICOLON, "Expected ';' after variable declaration");
  }

  private parseStruct(): void {
    this.consume(TokenType.IDENTIFIER, "Expected struct name");
    this.consume(TokenType.LEFT_BRACE, "Expected '{' after struct name");
    
    while (!this.check(TokenType.RIGHT_BRACE) && !this.isAtEnd()) {
      this.consume(TokenType.IDENTIFIER, "Expected field name");
      this.consume(TokenType.COLON, "Expected ':' after field name");
      this.parseType();
      this.consume(TokenType.SEMICOLON, "Expected ';' after field");
    }
    
    this.consume(TokenType.RIGHT_BRACE, "Expected '}' after struct body");
  }

  private parseEnum(): void {
    this.consume(TokenType.IDENTIFIER, "Expected enum name");
    this.consume(TokenType.LEFT_BRACE, "Expected '{' after enum name");
    
    while (!this.check(TokenType.RIGHT_BRACE) && !this.isAtEnd()) {
      this.consume(TokenType.IDENTIFIER, "Expected variant name");
      if (this.match(TokenType.COMMA)) continue;
      if (this.check(TokenType.RIGHT_BRACE)) break;
    }
    
    this.consume(TokenType.RIGHT_BRACE, "Expected '}' after enum body");
  }

  private parseBlock(): void {
    while (!this.check(TokenType.RIGHT_BRACE) && !this.isAtEnd()) {
      this.parseStatement();
    }
  }

  private parseStatement(): void {
    if (this.match(TokenType.LET) || this.match(TokenType.VAR)) {
      this.parseVariableDecl();
    } else if (this.match(TokenType.RETURN)) {
      if (!this.check(TokenType.SEMICOLON)) {
        this.parseExpression();
      }
      this.consume(TokenType.SEMICOLON, "Expected ';' after return");
    } else if (this.match(TokenType.IF)) {
      this.parseIfStatement();
    } else if (this.match(TokenType.WHILE)) {
      this.parseWhileStatement();
    } else if (this.match(TokenType.FOR)) {
      this.parseForStatement();
    } else if (this.match(TokenType.LEFT_BRACE)) {
      this.parseBlock();
      this.consume(TokenType.RIGHT_BRACE, "Expected '}' after block");
    } else {
      this.parseExpression();
      this.consume(TokenType.SEMICOLON, "Expected ';' after expression");
    }
  }

  private parseIfStatement(): void {
    this.consume(TokenType.LEFT_PAREN, "Expected '(' after 'if'");
    this.parseExpression();
    this.consume(TokenType.RIGHT_PAREN, "Expected ')' after condition");
    this.consume(TokenType.LEFT_BRACE, "Expected '{' after if condition");
    this.parseBlock();
    this.consume(TokenType.RIGHT_BRACE, "Expected '}' after if body");
    if (this.match(TokenType.ELSE)) {
      if (this.match(TokenType.IF)) {
        this.parseIfStatement();
      } else {
        this.consume(TokenType.LEFT_BRACE, "Expected '{' after 'else'");
        this.parseBlock();
        this.consume(TokenType.RIGHT_BRACE, "Expected '}' after else body");
      }
    }
  }

  private parseWhileStatement(): void {
    this.consume(TokenType.LEFT_PAREN, "Expected '(' after 'while'");
    this.parseExpression();
    this.consume(TokenType.RIGHT_PAREN, "Expected ')' after condition");
    this.consume(TokenType.LEFT_BRACE, "Expected '{' after while condition");
    this.parseBlock();
    this.consume(TokenType.RIGHT_BRACE, "Expected '}' after while body");
  }

  private parseForStatement(): void {
    this.consume(TokenType.LEFT_PAREN, "Expected '(' after 'for'");
    if (!this.check(TokenType.SEMICOLON)) {
      this.parseStatement();
    }
    this.consume(TokenType.SEMICOLON, "Expected ';' after for initializer");
    if (!this.check(TokenType.SEMICOLON)) {
      this.parseExpression();
    }
    this.consume(TokenType.SEMICOLON, "Expected ';' after for condition");
    if (!this.check(TokenType.RIGHT_PAREN)) {
      this.parseExpression();
    }
    this.consume(TokenType.RIGHT_PAREN, "Expected ')' after for clause");
    this.consume(TokenType.LEFT_BRACE, "Expected '{' after for");
    this.parseBlock();
    this.consume(TokenType.RIGHT_BRACE, "Expected '}' after for body");
  }

  private parseExpression(): void {
    this.parseAssignment();
  }

  private parseAssignment(): void {
    this.parseOr();
    if (this.match(TokenType.EQUAL)) {
      this.parseAssignment();
    }
  }

  private parseOr(): void {
    this.parseAnd();
    while (this.match(TokenType.PIPE_PIPE)) {
      this.parseAnd();
    }
  }

  private parseAnd(): void {
    this.parseEquality();
    while (this.match(TokenType.AMPERSAND_AMPERSAND)) {
      this.parseEquality();
    }
  }

  private parseEquality(): void {
    this.parseComparison();
    while (this.match(TokenType.BANG_EQUAL) || this.match(TokenType.EQUAL_EQUAL)) {
      this.parseComparison();
    }
  }

  private parseComparison(): void {
    this.parseTerm();
    while (this.match(TokenType.GREATER) || this.match(TokenType.GREATER_EQUAL) ||
           this.match(TokenType.LESS) || this.match(TokenType.LESS_EQUAL)) {
      this.parseTerm();
    }
  }

  private parseTerm(): void {
    this.parseFactor();
    while (this.match(TokenType.MINUS) || this.match(TokenType.PLUS)) {
      this.parseFactor();
    }
  }

  private parseFactor(): void {
    this.parseUnary();
    while (this.match(TokenType.SLASH) || this.match(TokenType.STAR) || 
           this.match(TokenType.PERCENT)) {
      this.parseUnary();
    }
  }

  private parseUnary(): void {
    if (this.match(TokenType.BANG) || this.match(TokenType.MINUS)) {
      this.parseUnary();
    } else {
      this.parsePrimary();
    }
  }

  private parsePrimary(): void {
    if (this.match(TokenType.TRUE) || this.match(TokenType.FALSE) ||
        this.match(TokenType.INTEGER_LITERAL) || this.match(TokenType.HEX_LITERAL) ||
        this.match(TokenType.BINARY_LITERAL) || this.match(TokenType.STRING_LITERAL)) {
      // Literal
    } else if (this.match(TokenType.IDENTIFIER)) {
      if (this.match(TokenType.LEFT_PAREN)) {
        // Function call
        if (!this.check(TokenType.RIGHT_PAREN)) {
          do {
            this.parseExpression();
            if (!this.match(TokenType.COMMA)) break;
          } while (!this.check(TokenType.RIGHT_PAREN));
        }
        this.consume(TokenType.RIGHT_PAREN, "Expected ')' after arguments");
      } else if (this.match(TokenType.DOT)) {
        // Member access
        this.consume(TokenType.IDENTIFIER, "Expected member name");
      }
    } else if (this.match(TokenType.LEFT_PAREN)) {
      this.parseExpression();
      this.consume(TokenType.RIGHT_PAREN, "Expected ')' after expression");
    } else {
      this.error("Expected expression");
    }
  }

  private match(...types: TokenType[]): boolean {
    for (const type of types) {
      if (this.check(type)) {
        this.advance();
        return true;
      }
    }
    return false;
  }

  private check(type: TokenType): boolean {
    if (this.isAtEnd()) return false;
    return this.peek().type === type;
  }

  private advance(): Token {
    if (!this.isAtEnd()) this.current++;
    return this.previous();
  }

  private isAtEnd(): boolean {
    return this.peek().type === TokenType.EOF;
  }

  private peek(): Token {
    return this.tokens[this.current];
  }

  private previous(): Token {
    return this.tokens[this.current - 1];
  }

  private consume(type: TokenType, message: string): Token {
    if (this.check(type)) return this.advance();
    const token = this.peek();
    this.errorAt(token, message);
    return token;
  }

  private error(message: string): void {
    const token = this.peek();
    this.errorAt(token, message);
  }

  private errorAt(token: Token, message: string): void {
    this.errors.push({
      message,
      line: token.line,
      column: token.column,
    });
  }

  private synchronize(): void {
    this.advance();
    while (!this.isAtEnd()) {
      if (this.previous().type === TokenType.SEMICOLON) return;
      if (this.peek().type === TokenType.CONTRACT || 
          this.peek().type === TokenType.FN ||
          this.peek().type === TokenType.STRUCT ||
          this.peek().type === TokenType.ENUM ||
          this.peek().type === TokenType.LET ||
          this.peek().type === TokenType.VAR ||
          this.peek().type === TokenType.RETURN) {
        return;
      }
      this.advance();
    }
  }
}

// Main validation function
export interface ValidationResult {
  success: boolean;
  error_type: 'none' | 'lexical' | 'syntax';
  error_message: string;
  line: number;
  column: number;
  token_count?: number;
  ast_node_count?: number;
}

export function validateOraSyntax(source: string): ValidationResult {
  // Lexical analysis
  const lexer = new Lexer(source);
  const { tokens, errors: lexErrors } = lexer.scanTokens();

  if (lexErrors.length > 0) {
    const firstError = lexErrors[0];
    return {
      success: false,
      error_type: 'lexical',
      error_message: firstError,
      line: 1,
      column: 1,
      token_count: tokens.length,
    };
  }

  // Syntax analysis
  const parser = new Parser(tokens);
  const { success, errors: parseErrors } = parser.parse();

  if (!success && parseErrors.length > 0) {
    const firstError = parseErrors[0];
    return {
      success: false,
      error_type: 'syntax',
      error_message: firstError.message,
      line: firstError.line,
      column: firstError.column,
      token_count: tokens.length,
    };
  }

  return {
    success: true,
    error_type: 'none',
    error_message: '',
    line: 0,
    column: 0,
    token_count: tokens.length,
  };
}

