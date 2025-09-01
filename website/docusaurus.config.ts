import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Ora Development Notebook',
  tagline: 'Experimental smart contract language research and development',
  favicon: 'img/favicon.ico',

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
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/oralang/Ora/tree/main/website/',
        },

        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/ora-social-card.jpg',
    navbar: {
      title: 'Ora Notebook',
      logo: {
        alt: 'Ora Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          type: 'dropdown',
          label: 'Technical Specs',
          position: 'left',
          items: [
            {
              to: '/docs/specifications/',
              label: 'Overview',
            },
            {
              to: '/docs/specifications/grammar',
              label: 'Grammar',
            },
            {
              to: '/docs/specifications/mlir',
              label: 'MLIR',
            },
            {
              to: '/docs/specifications/formal-verification',
              label: 'Formal Verification',
            },
            {
              to: '/docs/specifications/api',
              label: 'API',
            },
          ],
        },
        {
          to: '/api-docs/',
          label: 'API Reference',
          position: 'left',
        },
        {
          href: 'https://github.com/oralang/Ora',
          label: 'Source Code',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Development Notebook',
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
              label: 'Asuka Release',
              to: '/docs/asuka-release',
            },
            {
              label: 'Language Basics',
              to: '/docs/language-basics',
            },
            {
              label: 'Examples',
              to: '/docs/examples',
            },
          ],
        },
        {
          title: 'Technical Documentation',
          items: [
            {
              label: 'Specifications',
              to: '/docs/specifications/',
            },
            {
              label: 'API Reference',
              to: '/api-docs/',
            },
            {
              label: 'Grammar Definition',
              to: '/docs/specifications/grammar',
            },
            {
              label: 'MLIR Integration',
              to: '/docs/specifications/mlir',
            },
          ],
        },
        {
          title: 'Development',
          items: [
            {
              label: 'Source Code',
              href: 'https://github.com/oralang/Ora',
            },
            {
              label: 'Issues & Bugs',
              href: 'https://github.com/oralang/Ora/issues',
            },
            {
              label: 'Discussions',
              href: 'https://github.com/oralang/Ora/discussions',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Ora Language Project. Experimental research project - not for production use.`,
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
