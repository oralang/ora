import type { ReactNode } from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title} Development Notebook
        </Heading>
        <p className="hero__subtitle">An experimental smart contract language with formal verification capabilities</p>

        {/* Experimental Status Banner */}
        <div className={styles.statusBanner}>
          <div className={styles.experimentalWarning}>
            <strong>⚠️ EXPERIMENTAL PROJECT</strong>
            <span>Ora is in active development and is NOT ready for production use. This is an open notebook documenting the language design and implementation progress.</span>
          </div>
        </div>

        {/* Implementation Status */}
        <div className={styles.implementationStatus}>
          <div className={styles.statusGrid}>
            <div className={styles.statusItem}>
              <span className={styles.statusIcon}>✅</span>
              <strong>Core Compilation</strong>
              <span>Lexical → Syntax → Semantic → HIR → Yul</span>
            </div>
            <div className={styles.statusItem}>
              <span className={styles.statusIcon}>🚧</span>
              <strong>Formal Verification</strong>
              <span>In active development</span>
            </div>
            <div className={styles.statusItem}>
              <span className={styles.statusIcon}>🚧</span>
              <strong>Advanced Safety</strong>
              <span>Memory safety, overflow protection</span>
            </div>
          </div>
        </div>

        {/* Quick Language Overview */}
        <div className={styles.codeExample}>
          <div className={styles.codeHeader}>
            <span>Current Language Sample</span>
            <small>Subject to change</small>
          </div>
          <pre className={styles.codeBlock}>
            <code className="language-ora">{`contract SimpleToken {
    storage var total_supply: u256;
    storage var balances: map[address, u256];
    
    log Transfer(from: address, to: address, amount: u256);
    
    pub fn transfer(to: address, amount: u256) -> bool {
        requires(balances[std.transaction.sender] >= amount);
        requires(to != std.constants.ZERO_ADDRESS);
        
        @lock(balances[to]);
        balances from std.transaction.sender -> to : amount;
        
        log Transfer(std.transaction.sender, to, amount);
        return true;
    }
}`}</code>
          </pre>
        </div>

        {/* Development Navigation */}
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Language Reference
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            to="/docs/examples">
            Implementation Examples
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            to="/docs/specifications/">
            Technical Specifications
          </Link>
          <Link
            className="button button--outline button--secondary button--lg"
            href="https://github.com/oralang/Ora">
            Source Code
          </Link>
        </div>

        {/* Current Progress */}
        <div className={styles.progressSection}>
          <h3>Current Development Focus</h3>
          <div className={styles.progressGrid}>
            <div className={styles.progressItem}>
              <strong>📝 Grammar Definition</strong>
              <span>Refining syntax and semantic rules</span>
            </div>
            <div className={styles.progressItem}>
              <strong>🔧 HIR Implementation</strong>
              <span>High-level IR for analysis passes</span>
            </div>
            <div className={styles.progressItem}>
              <strong>✅ Formal Verification</strong>
              <span>Mathematical proof system design</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Ora development notebook: An experimental smart contract language with formal verification capabilities. Documentation of language design and implementation progress.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
