import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'The Ora Little Book',
      items: [
        'book/hello-ora',
        'book/types-and-variables',
        'book/functions-and-operators',
        'book/control-flow',
        'book/composite-types',
        {
          type: 'category',
          label: 'Part II: Safety First',
          items: [
            'book/error-unions',
            'book/memory-regions',
            'book/refinement-types',
            'book/logs-and-events',
          ],
        },
        {
          type: 'category',
          label: 'Part III: Verification',
          items: [
            'book/specification-clauses',
            'book/ghost-state',
            'book/locks',
            'book/standard-library',
          ],
        },
        {
          type: 'category',
          label: 'Part IV: Abstraction',
          items: [
            'book/traits',
            'book/generics',
            'book/comptime',
            'book/extern-traits',
            'book/bitfields',
          ],
        },
        {
          type: 'category',
          label: 'Part V: Real-World Ora',
          items: [
            'book/projects',
            'book/full-vault',
          ],
        },
        {
          type: 'category',
          label: 'Appendices',
          items: [
            'book/ora-vs-solidity',
          ],
        },
      ],
    },
    'getting-started',
    'language-basics',
    'imports',
    'examples',
    'code-formatter',
    'standard-library',
    {
      type: 'category',
      label: 'Language Features',
      items: [
        'switch',
        'struct-types',
        'generics',
        'traits',
        'extern-traits',
        'research/comptime',
        'signed-integers',
        'bitfield-types',
        'memory-regions',
        'error-unions',
        'refinement-types',
        'logs-and-events',
      ],
    },
    {
      type: 'category',
      label: 'Specifications',
      items: [
        'specifications/index',
        'specifications/grammar',
        'specifications/mlir',
        'specifications/sensei-ir',
        'specifications/api',
      ],
    },
    {
      type: 'category',
      label: 'Research',
      items: [
        'research/index',
        'research/research-snapshot',
        'formal-verification',
        'research/compiler-architecture',
        'design-documents',
        'design-documents/type-system-v0.1',
        'research/type-system',
        'research/comptime',
        'research/smt-verification',
        'research/refinement-types',
      ],
    },
    {
      type: 'category',
      label: 'Compiler Field Guide',
      items: [
        'compiler/field-guide/index',
        'compiler/field-guide/welcome',
        'compiler/field-guide/first-win',
        'compiler/field-guide/compiler-shape',
        'compiler/field-guide/contribution-tracks',
        'compiler/field-guide/walkthrough',
        'compiler/field-guide/04b-failing-walkthrough',
        'compiler/field-guide/core-systems',
        'compiler/field-guide/debugging-playbook',
        'compiler/field-guide/first-compiler-pr',
        'compiler/field-guide/advanced',
        {
          type: 'category',
          label: 'Appendices',
          items: [
            'compiler/field-guide/appendix-a-glossary',
            'compiler/field-guide/appendix-b-codebase-map',
            'compiler/field-guide/appendix-c-feature-checklist',
            'compiler/field-guide/appendix-d-tests',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Optimization',
      items: [
        'inline-functions',
      ],
    },
    {
      type: 'category',
      label: 'Specs',
      items: [
        'specs/type-system',
        'specs/abi',
        'specs/bitfield',
      ],
    },
    'roadmap-to-asuka',
  ],
};

export default sidebars;
