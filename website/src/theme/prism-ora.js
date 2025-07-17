export default function (Prism) {
    // Ora language syntax highlighting
    Prism.languages.ora = {
        'comment': {
            pattern: /\/\/.*$/m,
            greedy: true
        },
        'string': {
            pattern: /"(?:[^"\\]|\\.)*"/,
            greedy: true
        },
        'keyword': /\b(?:storage|var|const|contract|function|fn|event|log|if|else|for|while|match|try|catch|return|break|continue|import|export|struct|enum|interface|trait|impl|requires|ensures|invariant|let|mut|pub|priv|static|async|await|yield|type|alias|where|in|as|is|not|and|or|xor|mod|div|rem|shl|shr|bitand|bitor|bitxor|bitnot|self|Self|super|crate|extern|unsafe|move|ref|dyn|abstract|virtual|override|final|class|namespace|using|throw|new|delete|this|base|null|true|false|undefined|void|bool|i8|i16|i32|i64|i128|u8|u16|u32|u64|u128|usize|isize|f32|f64|char|str|String|Vec|HashMap|HashSet|Option|Result|Ok|Err|Some|None|from|to|immutable|memory|tstore|old|comptime|error|doublemap|slice|forall|where|result)\b/,
        'builtin': /\b(?:@lock|@unlock|@divTrunc|@divFloor|@mod|@rem|@shl|@shr|@bitAnd|@bitOr|@bitXor|@bitNot|@addWithOverflow|@subWithOverflow|@mulWithOverflow|@divWithOverflow|@shlWithOverflow|@shrWithOverflow|@intCast|@floatCast|@ptrCast|@errdefer|@defer|@compileError|@compileLog|@import|@cImport|@embedFile|@This|@TypeOf|@hasDecl|@hasField|@tagName|@errorName|@panic|@setRuntimeSafety|@setFloatMode|@setGlobalLinkage|@setAlignStack|@frame|@Frame|@frameSize|@frameAddress|@returnAddress|@src|@sqrt|@sin|@cos|@tan|@exp|@exp2|@log|@log2|@log10|@fabs|@floor|@ceil|@trunc|@round|@min|@max|@clamp)\b/,
        'stdlib': /\b(?:std\.transaction\.sender|std\.transaction\.origin|std\.transaction\.gasPrice|std\.transaction\.gasLimit|std\.block\.number|std\.block\.timestamp|std\.block\.difficulty|std\.block\.gaslimit|std\.block\.coinbase|std\.constants\.ZERO_ADDRESS|std\.constants\.MAX_UINT256|std\.math\.min|std\.math\.max|std\.math\.abs|std\.memory\.alloc|std\.memory\.free|std\.crypto\.keccak256|std\.crypto\.sha256|std\.crypto\.ripemd160|std\.crypto\.ecrecover|std\.storage\.get|std\.storage\.set|std\.events\.emit|std\.abi\.encode|std\.abi\.decode)\b/,
        'type': /\b(?:address|bytes|bytes32|uint256|u256|map|Array|Option|Result|Contract|Event|Function|Tuple|Union|Enum|Struct|Interface|Trait)\b/,
        'transfer-syntax': /\b(\w+)\s+(from)\s+(.+?)\s+(->)\s+(.+?)\s+(:)\s+(.+?)\b/,
        'operator': /->|[+\-*/%=<>!&|^~?:;.,\[\](){}]/,
        'number': /\b(?:0x[0-9a-fA-F]+|0b[01]+|0o[0-7]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\b/,
        'punctuation': /[{}[\];(),.:]/,
        'function': /\b[a-zA-Z_][a-zA-Z0-9_]*(?=\s*\()/,
        'variable': /\b[a-zA-Z_][a-zA-Z0-9_]*\b/,
        'annotation': /@[a-zA-Z_][a-zA-Z0-9_]*/,
        'macro': /![a-zA-Z_][a-zA-Z0-9_]*/,
        'lifetime': /'[a-zA-Z_][a-zA-Z0-9_]*/,
        'error-type': /![a-zA-Z_][a-zA-Z0-9_]*\b/,
        'generic': /<[^>]+>/,
        'attribute': /#\[[^\]]+\]/
    };
} 