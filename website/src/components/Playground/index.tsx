import React, { useState, useRef, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import { validateOraSyntax } from './ora-parser';
import styles from './styles.module.css';

interface ValidationResult {
  success: boolean;
  error_type: 'none' | 'lexical' | 'syntax';
  error_message: string;
  line: number;
  column: number;
  token_count?: number;
  ast_node_count?: number;
}

const DEFAULT_CODE = `contract SimpleContract {

    fn test(x: u256) -> u256 {
        return x * 2;
    }

    pub fn main(x: u256) {
        let y = x + 1;
        test(y);
    }
}`;

const EXAMPLES = [
  {
    name: 'Simple Contract',
    category: 'Basics',
    code: `contract SimpleContract {

    fn test(x: u256) -> u256 {
        return x * 2;
    }

    pub fn main(x: u256) {
        let y = x + 1;
        test(y);
    }
}`,
  },
  {
    name: 'Storage Variables',
    category: 'Storage',
    code: `// Basic storage variable declarations

contract StorageTest {
    // Basic storage variables with different types
    storage var counter: u256;
    storage var active: bool;
    storage var owner: address;
    storage var name: string;
    storage var data: bytes;
    
    // Constant (immutable) storage variable
    storage let token_name: string;
    
    // Storage mapping
    storage var balances: map[address, u256];
    
    // Storage double mapping
    storage var allowances: map[address, map[address, u256]];
    
    // Storage slice (dynamic array)
    storage var history: slice[u256];
}`,
  },
  {
    name: 'Structs',
    category: 'Types',
    code: `// Struct declarations

// Basic struct with simple fields
struct Point {
    x: u256;
    y: u256;
}

// Struct with various types
struct UserProfile {
    id: u256;
    username: string;
    active: bool;
    creation_time: u256;
    wallet: address;
}

// Struct with comments
struct TokenInfo {
    id: u256;        // Unique identifier
    symbol: string;  // Token symbol (e.g., "ETH")
    decimals: u8;    // Number of decimal places
}

// Nested struct
struct AccountData {
    profile: UserProfile;
    balance: u256;
    last_login: u256;
}`,
  },
  {
    name: 'Enums',
    category: 'Types',
    code: `// Enum declarations

// Basic enum without explicit values
enum Status {
    Pending,
    Active,
    Completed,
    Cancelled
}

// Basic enum with explicit values
enum StatusExplicit : u8 {
    Active = 0,
    Pending = 1,
    Suspended = 2,
    Closed = 3
}

// Enum for operation types
enum OperationType : u8 {
    Create = 1,
    Read = 2,
    Update = 3,
    Delete = 4
}

// Enum with comments
enum TokenStandard : u8 {
    ERC20 = 0,   // Fungible token
    ERC721 = 1,  // Non-fungible token
    ERC1155 = 2  // Multi-token
}`,
  },
  {
    name: 'Switch Expressions',
    category: 'Control Flow',
    code: `// Switch expression syntax

enum Role : u8 {
    User = 0,
    Admin = 1
}

contract SwitchExpressionTest {
    pub fn test_switch_expression() {
        // Switch expression returning a value
        var value: u256 = 1;
        var switch_result: u256 = switch (value) {
            0 => 10,
            1 => 20,
            2 => 30,
            else => 0,
        };
    }
    
    pub fn test_switch_with_enum() {
        var role: Role = Role.Admin;
        
        // Switch with enum values
        var permission_level: u256 = switch (role) {
            Role.User => 1,
            Role.Admin => 10,
        };
    }
    
    pub fn test_condition_switch() {
        var balance: u256 = 100;
        var is_active: bool = true;
        
        // Switch with boolean condition
        var status: string = switch (is_active) {
            false => "Inactive",
            true => "Active",
        };
    }
}`,
  },
  {
    name: 'Control Flow',
    category: 'Control Flow',
    code: `// Basic control flow constructs

contract BasicControlFlowTest {
    // Test if statement
    fn testIf() -> u256 {
        let x: u256 = 10;
        if (x > 5) {
            return 1;
        } else {
            return 0;
        }
    }
    
    // Test while loop
    fn testWhile() -> u256 {
        var counter: u256 = 0;
        while (counter < 5) {
            counter = counter + 1;
        }
        return counter;
    }
    
    // Test simple switch
    fn testSwitch() -> u256 {
        let value: u256 = 2;
        switch (value) {
            1 => { return 10; }
            2 => { return 20; }
            3 => { return 30; }
            else => { return 0; }
        }
    }
    
    // Test return statement
    fn testReturn() -> u256 {
        return 42;
    }
}`,
  },
  {
    name: 'Memory Operations',
    category: 'Memory',
    code: `// Memory variable declarations

contract MemoryDeclarationTest {
    pub fn memory_declarations() {
        // Local memory variables with explicit type
        let x: u256 = 10;
        let y: u256 = 20;
        let sum: u256 = x + y;
        
        // Memory variable with explicit type
        var product: u256 = x * y;
        
        // Memory constants (cannot be reassigned)
        const MAX_VALUE: u256 = 1000;
        const MIN_VALUE: u256 = 0;
        
        // Memory arrays
        let numbers: slice[u256] = [1, 2, 3, 4, 5];
    }
}`,
  },
  {
    name: 'Refinement Types',
    category: 'Advanced',
    code: `// Test basic refinement types

contract TestRefinements {
    // Test MinValue
    fn testMinValue(amount: MinValue<u256, 1000>) {
        // Should compile - amount is guaranteed >= 1000
    }

    // Test MaxValue
    fn testMaxValue(limit: MaxValue<u256, 10000>) {
        // Should compile - limit is guaranteed <= 10000
    }

    // Test InRange
    fn testInRange(rate: InRange<u256, 0, 10000>) {
        // Should compile - rate is guaranteed 0 <= rate <= 10000
    }

    // Test Scaled
    fn testScaled(amount: Scaled<u256, 18>) {
        // Should compile - amount is scaled by 10^18
    }

    // Test NonZero alias
    fn testNonZero(divisor: NonZero<u256>) {
        // Should compile - divisor is guaranteed > 0
    }

    // Test BasisPoints alias
    fn testBasisPoints(fee: BasisPoints<u256>) {
        // Should compile - fee is 0-10000 (basis points)
    }

    // Test variable declarations
    fn testVariables() {
        let deposit: MinValue<u256, 1_000_000> = 1_000_000;
        let fee: BasisPoints<u256> = 250; // 2.5%
        let total: Exact<u256> = 1000;
    }
}`,
  },
];

export default function Playground(): JSX.Element {
  const [code, setCode] = useState<string>(DEFAULT_CODE);
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [selectedExample, setSelectedExample] = useState<string>('Simple Contract');
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  const editorRef = useRef<any>(null);

  // Handle editor mount
  const handleEditorDidMount = (editor: any, monaco: any) => {
    editorRef.current = editor;
    
    // Scroll to top to show line 1
    editor.setScrollTop(0);
    editor.setScrollLeft(0);
    
    // Register Ora language if not already registered
    if (!monaco.languages.getLanguages().find((lang: any) => lang.id === 'ora')) {
      monaco.languages.register({ id: 'ora' });
    }
    
    // Configure Monaco for Ora
    monaco.languages.setMonarchTokensProvider('ora', {
      tokenizer: {
        root: [
          [/contract|storage|pub|fn|requires|ensures|return|if|else|while|for|switch|try|catch/, 'keyword'],
          [/u256|u128|u64|u32|u16|u8|i256|i128|i64|i32|i16|i8|bool|address|bytes/, 'type'],
          [/[a-z_][a-z0-9_]*/, 'identifier'],
          [/[A-Z][a-zA-Z0-9_]*/, 'type.identifier'],
          [/0x[0-9a-fA-F]+/, 'number.hex'],
          [/[0-9]+/, 'number'],
          [/\/\/.*$/, 'comment'],
          [/\/\*[\s\S]*?\*\//, 'comment'],
          [/[{}()\[\]]/, 'delimiter.bracket'],
          [/[;,]/, 'delimiter'],
          [/[=+\-*/%<>!&|]/, 'operator'],
        ],
      },
    });

    monaco.languages.setLanguageConfiguration('ora', {
      comments: {
        lineComment: '//',
        blockComment: ['/*', '*/'],
      },
      brackets: [
        ['{', '}'],
        ['[', ']'],
        ['(', ')'],
      ],
      autoClosingPairs: [
        { open: '{', close: '}' },
        { open: '[', close: ']' },
        { open: '(', close: ')' },
        { open: '"', close: '"' },
      ],
      surroundingPairs: [
        { open: '{', close: '}' },
        { open: '[', close: ']' },
        { open: '(', close: ')' },
        { open: '"', close: '"' },
      ],
    });
  };

  // Validate syntax using JavaScript parser
  const validateSyntax = async (sourceCode: string) => {
    setIsValidating(true);
    
    try {
      // Small delay for UI feedback
      await new Promise(resolve => setTimeout(resolve, 50));
      
      const result = validateOraSyntax(sourceCode);
      setValidationResult(result);
    } catch (error) {
      console.error('Validation error:', error);
      setValidationResult({
        success: false,
        error_type: 'syntax',
        error_message: error instanceof Error ? error.message : 'Validation failed',
        line: 1,
        column: 1,
      });
    } finally {
      setIsValidating(false);
    }
  };

  // Handle code changes
  const handleCodeChange = (value: string | undefined) => {
    const newCode = value || '';
    setCode(newCode);
    
    // Debounce validation
    const timeoutId = setTimeout(() => {
      if (newCode.trim().length > 0) {
        validateSyntax(newCode);
      }
    }, 500);
    
    return () => clearTimeout(timeoutId);
  };

  // Load example
  const loadExample = (exampleName: string) => {
    const example = EXAMPLES.find(ex => ex.name === exampleName);
    if (example) {
      setCode(example.code);
      setSelectedExample(exampleName);
      validateSyntax(example.code);
      
      // Scroll editor to top when loading example
      if (editorRef.current) {
        editorRef.current.setScrollTop(0);
        editorRef.current.setScrollLeft(0);
        editorRef.current.setPosition({ lineNumber: 1, column: 1 });
        editorRef.current.revealLine(1);
      }
    }
  };

  // Group examples by category
  const examplesByCategory = EXAMPLES.reduce((acc, example) => {
    const category = example.category || 'Other';
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(example);
    return acc;
  }, {} as Record<string, typeof EXAMPLES>);

  return (
    <div className={styles.playground}>
      <div className={styles.toolbar}>
        <div className={styles.toolbarLeft}>
          <button
            className={styles.sidebarToggle}
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label="Toggle sidebar"
          >
            {sidebarOpen ? '◀' : '▶'}
          </button>
          <h2 className={styles.title}>Ora Playground</h2>
          <span className={styles.subtitle}>Syntax validation only</span>
        </div>
      </div>

      <div className={styles.content}>
        {sidebarOpen && (
          <div className={styles.leftSidebar}>
            <div className={styles.sidebarHeader}>
              <h3 className={styles.sidebarTitle}>Examples</h3>
            </div>
            <div className={styles.sidebarContent}>
              {Object.entries(examplesByCategory).map(([category, examples]) => (
                <div key={category} className={styles.categorySection}>
                  <div className={styles.categoryTitle}>{category}</div>
                  {examples.map((example) => (
                    <button
                      key={example.name}
                      className={`${styles.exampleButton} ${
                        selectedExample === example.name ? styles.exampleButtonActive : ''
                      }`}
                      onClick={() => loadExample(example.name)}
                    >
                      {example.name}
                    </button>
                  ))}
                </div>
              ))}
            </div>
          </div>
        )}
        <div className={styles.editorContainer}>
          <Editor
            height="100%"
            language="ora"
            value={code}
            onChange={handleCodeChange}
            onMount={handleEditorDidMount}
            theme="vs-dark"
            options={{
              minimap: { enabled: false },
              fontSize: 14,
              lineNumbers: 'on',
              scrollBeyondLastLine: false,
              automaticLayout: true,
              tabSize: 4,
              wordWrap: 'on',
              padding: { top: 0, bottom: 0 },
              glyphMargin: true,
              folding: true,
              lineDecorationsWidth: 10,
              lineNumbersMinChars: 3,
              scrollbar: {
                vertical: 'auto',
                horizontal: 'auto',
                useShadows: false,
              },
            }}
          />
        </div>

        <div className={styles.sidebar}>
          <div className={styles.panel}>
            <h3 className={styles.panelTitle}>Validation</h3>
            {isValidating ? (
              <div className={styles.loading}>Validating...</div>
            ) : validationResult ? (
              <div className={styles.validationResult}>
                {validationResult.success ? (
                  <div className={styles.success}>
                    <div>
                      <span className={styles.successIcon}>✓</span>
                      <span>Syntax is valid</span>
                    </div>
                    {validationResult.token_count && (
                      <div className={styles.stats}>
                        Tokens: {validationResult.token_count}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className={styles.error}>
                    <div className={styles.errorHeader}>
                      <span className={styles.errorIcon}>✗</span>
                      <span className={styles.errorType}>
                        {validationResult.error_type === 'lexical' ? 'Lexical Error' : 'Syntax Error'}
                      </span>
                    </div>
                    <div className={styles.errorMessage}>
                      {validationResult.error_message || 'Unknown error'}
                    </div>
                    {validationResult.line > 0 && (
                      <div className={styles.errorLocation}>
                        Line {validationResult.line}, Column {validationResult.column}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className={styles.placeholder}>
                Start typing to validate syntax...
              </div>
            )}
          </div>

          <div className={styles.panel}>
            <h3 className={styles.panelTitle}>About</h3>
            <div className={styles.info}>
              <p>The playground validates Ora syntax using the lexer and parser.</p>
              <p><strong>Note:</strong> This is syntax validation only. No type checking or compilation.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

