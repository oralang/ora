import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Main documentation sidebar - minimal essential pages
  tutorialSidebar: [
    'intro',
    'getting-started',
    'language-basics',
    'examples',
    {
      type: 'category',
      label: 'Language Features',
      items: [
        'switch',
        'struct-types',
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
      label: 'Specifications',
      items: [
        'specifications/index',
        'specifications/grammar',
        'specifications/mlir',
        'specifications/api',
      ],
    },
    {
      type: 'category',
      label: 'Design Documents',
      items: [
        'design-documents',
        'design-documents/type-system-v0.1',
      ],
    },
    {
      type: 'category',
      label: 'EVM IR Specification',
      items: [
        'evm-ir/intro',
        'evm-ir/types',
        'evm-ir/ops',
        'evm-ir/legalizer',
        'evm-ir/stackifier',
        'evm-ir/debug',
        'evm-ir/abi-lowering',
        'evm-ir/examples',
        'evm-ir/appendix',
      ],
    },
  ],
};

export default sidebars;
