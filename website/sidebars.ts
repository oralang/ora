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
    'code-formatter',
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
      label: 'Specs',
      items: [
        'specs/type-system',
        'specs/abi',
      ],
    },
  ],
};

export default sidebars;
