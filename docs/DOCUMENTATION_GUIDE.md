# Documentation Maintenance Guide

This guide helps maintainers keep Ora documentation accurate and up-to-date.

## Documentation Structure

```
Ora/
├── README.md                  # Main project README
├── CONTRIBUTING.md            # Contributor guide
├── GRAMMAR.bnf                # Language grammar
├── docs/                     # Lose docs (like this one)
├── ora-example/              # Example Ora programs
│   └── README.md             # Example status and testing
├── website/
│   ├── docs/                 # User-facing documentation
│   │   ├── intro.md
│   │   ├── getting-started.md
│   │   ├── language-basics.md
│   │   ├── examples.md
│   │   ├── roadmap-to-asuka.md
│   │   └── specifications/
│   └── docusaurus.config.ts  # Website configuration
└── scripts/
    └── validate-examples.sh   # Example validation script
```

## Documentation Types

### 1. User-Facing Documentation (`website/docs/`)

**Purpose**: Help users learn and use Ora  
**Audience**: Developers, contributors, users

**Requirements**:
- Always accurate to current implementation
- Clear examples that compile
- Up-to-date status indicators (✅ 🚧 📋)
- Link to relevant specifications

**Update When**:
- Language syntax changes
- Features are added/removed
- Examples stop working
- Compiler behavior changes

### 2. Technical Documentation (`docs/tech-work/`)

**Purpose**: AI-assisted development notes and reviews  
**Audience**: Maintainers, AI assistants

**Requirements**:
- Detailed implementation notes
- Session summaries
- Technical decisions
- Internal reviews

**Note**: This is NOT for user consumption. Keep separate from website docs.

### 3. Examples (`ora-example/`)

**Purpose**: Demonstrate language features  
**Audience**: All users

**Requirements**:
- Must compile with current compiler
- Focused on specific features
- Well-commented
- Listed in `ora-example/README.md`

### 4. Specifications (`website/docs/specifications/`)

**Purpose**: Formal language and compiler specs  
**Audience**: Language designers, compiler developers

**Requirements**:
- Match implementation
- Cite grammar files
- Include status indicators
- Technical accuracy

## Maintenance Checklist

### After Language Changes

- [ ] Update `GRAMMAR.bnf` if syntax changes
- [ ] Run `./scripts/validate-examples.sh`
- [ ] Update broken examples in `ora-example/`
- [ ] Update `website/docs/language-basics.md`
- [ ] Update feature status indicators
- [ ] Test all code snippets in docs
- [ ] Update `roadmap-to-asuka.md` if needed

### Before Release

- [ ] Validate all examples (100% pass rate)
- [ ] Update version numbers
- [ ] Update timestamps
- [ ] Check all internal links
- [ ] Verify external links
- [ ] Update feature status tables
- [ ] Review getting-started guide
- [ ] Test setup scripts

### Monthly Maintenance

- [ ] Run example validation
- [ ] Check for outdated timestamps
- [ ] Review open documentation issues
- [ ] Update roadmap progress
- [ ] Check broken links
- [ ] Update contributor stats

## Validation Scripts

### Example Validation

```bash
./scripts/validate-examples.sh
```

Validates all `.ora` files in `ora-example/`. Should run clean before releases.

### Link Validation (TODO)

```bash
./scripts/validate-docs.sh
```

Checks internal and external links in documentation.

## Writing Guidelines

### Status Indicators

Use consistent status indicators:

- ✅ **Complete/Working**: Fully implemented and tested
- 🚧 **In Progress**: Partially implemented
- 📋 **Planned**: Designed but not started

### Code Examples

**Requirements**:
- Must compile with current compiler
- Include comments explaining behavior
- Show realistic use cases
- Keep focused and concise

**Testing**:
```bash
# Test a single example
./zig-out/bin/ora parse example.ora

# Test a code snippet
cat > /tmp/test.ora << 'EOF'
contract Test {
    storage var value: u256;
}
EOF
./zig-out/bin/ora parse /tmp/test.ora
```

### Links

**Internal links** (within website):
```markdown
[Getting Started](./getting-started.md)
[Grammar Spec](./specifications/grammar.md)
```

**External links** (GitHub, etc):
```markdown
[GitHub](https://github.com/oralang/Ora)
[CONTRIBUTING](https://github.com/oralang/Ora/blob/main/CONTRIBUTING.md)
```

### Timestamps

Update timestamps when making significant changes:

```markdown
*Last updated: October 2025*
```

Use month-year format, no specific dates (they become outdated quickly).

## Common Issues

### Outdated Examples

**Symptom**: `validate-examples.sh` fails  
**Fix**: Update examples to use current syntax  
**Prevention**: Run validation before commits

### Broken Links

**Symptom**: 404 errors in documentation  
**Fix**: Update or remove broken links  
**Prevention**: Link validation script (TODO)

### Version Conflicts

**Symptom**: Different version numbers across docs  
**Fix**: Update all references to match `build.zig.zon`  
**Source of truth**: `build.zig.zon` `.minimum_zig_version`

### Misleading Status

**Symptom**: Features marked "complete" but don't work  
**Fix**: Test features, update status indicators  
**Prevention**: Regular feature audits

## Documentation Workflow

### Adding New Features

1. Implement feature
2. Add tests
3. Add example to `ora-example/`
4. Update `language-basics.md`
5. Update specifications if needed
6. Add to roadmap or mark complete
7. Run validation scripts

### Deprecating Features

1. Mark as deprecated in docs
2. Update examples
3. Add migration guide
4. Remove after grace period
5. Update all documentation

### Fixing Documentation Bugs

1. Create issue with `documentation` label
2. Fix documentation
3. Validate affected examples
4. Update related docs
5. Close issue with PR reference

## Tools

### Build and Test

```bash
# Build compiler
zig build

# Run tests
zig build test

# Build website locally
cd website
npm install
npm run start
```

### Validation

```bash
# Validate examples
./scripts/validate-examples.sh

# Check specific example
./zig-out/bin/ora parse ora-example/smoke.ora

# Lex/Parse/AST commands
./zig-out/bin/ora lex file.ora
./zig-out/bin/ora parse file.ora
./zig-out/bin/ora ast file.ora
```

## Contact

- **Documentation Issues**: [GitHub Issues](https://github.com/oralang/Ora/issues) with `documentation` label
- **General Questions**: [GitHub Discussions](https://github.com/oralang/Ora/discussions)

---

*Last updated: October 2025*

