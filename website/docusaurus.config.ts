import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Ora Development Notebook',
  tagline: 'Pre-ASUKA Alpha - Smart Contract Language for EVM',
  favicon: 'img/favicon-32.png',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://ora-lang.org',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'oralang', // Usually your GitHub org/user name.
  projectName: 'Ora', // Usually your repo name.

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl:
            'https://github.com/oralang/Ora/tree/main/website/',
        },
        blog: {
          showReadingTime: true,
          blogTitle: 'Ora Development Blog',
          blogDescription: 'Updates on Ora language development, compiler progress, and technical deep-dives',
          postsPerPage: 'ALL',
          blogSidebarTitle: 'Recent Posts',
          blogSidebarCount: 10,
          editUrl:
            'https://github.com/oralang/Ora/tree/main/website/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  markdown: {
    mermaid: true,
  },

  themes: ['@docusaurus/theme-mermaid'],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/ora-social-card.jpg',
    navbar: {
      title: 'Ora Notebook',
      logo: {
        alt: 'Ora Logo',
        src: 'img/logo.png',
        srcDark: 'img/logo.png',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/docs/examples',
          label: 'Examples',
          position: 'left',
        },
        {
          to: '/blog',
          label: 'Blog',
          position: 'left',
        },
        {
          to: '/docs/roadmap-to-asuka',
          label: 'Roadmap',
          position: 'left',
        },
        {
          to: '/docs/specs/type-system',
          label: 'Specs',
          position: 'left',
        },
        {
          to: '/docs/compiler/field-guide',
          label: 'Field Guide',
          position: 'left',
        },
        {
          href: 'https://github.com/oralang/Ora/blob/main/CONTRIBUTING.md',
          label: 'Contributing',
          position: 'right',
        },
        {
          href: 'https://github.com/oralang/Ora',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Introduction',
              to: '/docs/intro',
            },
            {
              label: 'Getting Started',
              to: '/docs/getting-started',
            },
            {
              label: 'Language Basics',
              to: '/docs/language-basics',
            },
            {
              label: 'Examples',
              to: '/docs/examples',
            },
            {
              label: 'Compiler Field Guide',
              to: '/docs/compiler/field-guide',
            },
          ],
        },
        {
          title: 'Specifications',
          items: [
            {
              label: 'Grammar',
              to: '/docs/specifications/grammar',
            },
            {
              label: 'MLIR Integration',
              to: '/docs/specifications/mlir',
            },
            {
              label: 'API Reference',
              to: '/docs/specifications/api',
            },
          ],
        },
        {
          title: 'Specs',
          items: [
            {
              label: 'Type System v0.11',
              to: '/docs/specs/type-system',
            },
            {
              label: 'ABI v0.1',
              to: '/docs/specs/abi',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              to: '/blog',
            },
            {
              label: 'Roadmap',
              to: '/docs/roadmap-to-asuka',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/oralang/Ora',
            },
            {
              label: 'Issues',
              href: 'https://github.com/oralang/Ora/issues',
            },
            {
              label: 'Contributing',
              href: 'https://github.com/oralang/Ora/blob/main/CONTRIBUTING.md',
            },
          ],
        },
      ],
      copyright: `Copyright © 2025 Ora Language Project.`,
    },
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: false,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
