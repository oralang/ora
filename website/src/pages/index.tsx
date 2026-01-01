import type { ReactNode } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        {/* Status Badge */}
        <div className={styles.statusBadge}>
          <span className={styles.badge}>Pre-ASUKA Alpha</span>
          <span className={styles.badge}>Contributors Welcome</span>
        </div>

        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>

        <p className={styles.heroDescription}>
          An experimental smart contract language with explicit semantics,
          memory regions, and clean compilation pipeline targeting EVM via sensei-ir.
        </p>

        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Get Started →
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            to="/blog">
            Read the Blog
          </Link>
        </div>

        {/* Quick Stats */}
        <div className={styles.stats}>
          <div className={styles.stat}>
            <div className={styles.statValue}>79%</div>
            <div className={styles.statLabel}>Success Rate</div>
          </div>
          <div className={styles.stat}>
            <div className={styles.statValue}>76/96</div>
            <div className={styles.statLabel}>Examples Pass</div>
          </div>
          <div className={styles.stat}>
            <div className={styles.statValue}>81</div>
            <div className={styles.statLabel}>MLIR Operations</div>
          </div>
        </div>
      </div>
    </header>
  );
}

function FeatureList() {
  const features = [
    {
      title: '✅ What Works Now',
      items: [
        'Full lexer and parser (76/96 examples pass - 79% success rate)',
        'Complete type checking and semantic analysis',
        'Storage, memory, and transient storage operations',
        'Switch statements and control flow (if/else)',
        'Structs, enums, and custom types',
        'Complete MLIR lowering with 81 operations',
        'Arithmetic operations (add, sub, mul, div, rem)',
        'Map operations and memory management',
      ],
    },
    {
      title: '🚧 In Development',
      items: [
        'Complete sensei-ir (SIR) lowering and EVM code generation',
        'For loops with capture syntax',
        'Enhanced error handling (try-catch improvements)',
        'Type inference improvements',
        'Full formal verification framework',
      ],
    },
    {
      title: '📋 Planned for ASUKA',
      items: [
        'Comprehensive standard library',
        'Advanced optimization passes',
        'IDE integration (LSP)',
        'Package manager',
        '50+ working examples',
        'Language specification v1.0',
      ],
    },
  ];

  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.featureGrid}>
          {features.map((feature, idx) => (
            <div key={idx} className={styles.featureCard}>
              <h3>{feature.title}</h3>
              <ul>
                {feature.items.map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function CodeExample() {
  return (
    <section className={styles.codeSection}>
      <div className="container">
        <h2>Experience Ora</h2>
        <p>Clean, explicit syntax for smart contracts</p>
        <div className={styles.codeContainer}>
          <pre className={styles.codeBlock}>
            <code className="language-ora">{`contract SimpleToken {
    storage var total_supply: u256;
    storage var balances: map[address, u256];
    
    log Transfer(sender: address, recipient: address, amount: u256);
    
    pub fn transfer(to: address, amount: u256) -> !bool
        requires(balances[std.transaction.sender] >= amount)
        requires(to != std.constants.ZERO_ADDRESS)
    {
        balances[std.transaction.sender] -= amount;
        balances[to] += amount;
        @lock(balances[to]);
        
        log Transfer(std.transaction.sender, to, amount);
        return true;
    }
}`}</code>
          </pre>
        </div>
        <div className={styles.codeActions}>
          <Link to="/docs/examples" className="button button--primary">
            See More Examples
          </Link>
          <Link to="/docs/getting-started" className="button button--outline button--primary">
            Try It Yourself
          </Link>
        </div>
      </div>
    </section>
  );
}

function ContributeSection() {
  return (
    <section className={styles.contribute}>
      <div className="container">
        <h2>Join the Development</h2>
        <p>Ora is in active development toward the ASUKA release</p>

        <div className={styles.contributeGrid}>
          <div className={styles.contributeCard}>
            <h3>🐛 Report Issues</h3>
            <p>Found a bug or have a feature request?</p>
            <Link href="https://github.com/oralang/Ora/issues">
              Open an Issue →
            </Link>
          </div>

          <div className={styles.contributeCard}>
            <h3>📝 Improve Docs</h3>
            <p>Help us make documentation better</p>
            <Link href="https://github.com/oralang/Ora/blob/main/CONTRIBUTING.md">
              Contributing Guide →
            </Link>
          </div>

          <div className={styles.contributeCard}>
            <h3>💬 Discuss</h3>
            <p>Share ideas and ask questions</p>
            <Link href="https://github.com/oralang/Ora/discussions">
              Join Discussion →
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title="Home"
      description="Ora - Pre-ASUKA Alpha smart contract language for EVM with explicit semantics and formal verification">
      <HomepageHeader />
      <main>
        <FeatureList />
        <CodeExample />
        <ContributeSection />
      </main>
    </Layout>
  );
}
