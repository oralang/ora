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
    'asuka-release',
    'language-basics',
    'struct-types',
    'examples',
    {
      type: 'category',
      label: 'Specifications',
      items: [
        'specifications/index',
        'specifications/grammar',
        'specifications/hir',
        'specifications/formal-verification',
        'specifications/api',
      ],
    },
    'api-reference',
  ],
};

export default sidebars;
