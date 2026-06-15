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
    <header className={clsx('hero', styles.heroBanner)}>
      <div className={clsx('container', styles.heroInner)}>
        <div className={styles.heroLeft}
        >
          <div className={styles.statusBadge}>
            <span className={styles.badge}>Asuka v0.2</span>
            <span className={styles.badge}>Proof-Carrying Contracts</span>
            <span className={styles.badge}>Contributors Welcome</span>
          </div>

          <Heading as="h1" className={styles.heroTitle}>
            <span className={styles.oraGradient}>Ora</span> Asuka v0.2
          </Heading>
          <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>

          <p className={styles.heroDescription}>
            Smart contract language and compiler with explicit regions,
            first-class Result values, SMT verification reports, and an
            inspectable Ora MLIR to Sensei-IR pipeline.
          </p>

          <div className={styles.buttons}>
            <Link
              className="button button--secondary button--lg"
              to="/docs/intro">
              Start Here
            </Link>
            <Link
              className="button button--outline button--secondary button--lg"
              to="/docs/research/research-snapshot">
              Research Snapshot
            </Link>
            <Link
              className="button button--outline button--secondary button--lg"
              href="https://github.com/oralang/Ora">
              GitHub
            </Link>
          </div>
        </div>

        <div className={styles.heroRight}>
          <div className={styles.heroLogo}>
            <img src="/img/logo-round.png" alt="Ora Logo" className={styles.logoImage} />
          </div>
          <div className={styles.heroCard}>
            <div className={styles.heroCardTitle}>Pipeline</div>
            <div className={styles.pipeline}
            >
              Tokens → AST → Typed AST → Ora MLIR → Sensei-IR → EVM
            </div>
            <div className={styles.heroCardList}>
              <div>Regions and effects are explicit</div>
              <div>Refinements become guards or proofs</div>
              <div>Result and ADT carriers stay explicit</div>
              <div>SMT reports explain proof trust</div>
            </div>
            <div className={styles.heroCardLinks}>
              <Link to="/docs/specifications/mlir">MLIR Spec</Link>
              <span>•</span>
              <Link to="/docs/specifications/sensei-ir">Sensei-IR</Link>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function FocusAreas() {
  const areas = [
    {
      title: 'Language',
      items: [
        'Region-aware types and effects',
        'Refinement types and Result/error unions',
        'Unified ADTs and exhaustive matching',
      ],
      link: '/docs/language-basics',
      linkLabel: 'Language Basics',
    },
    {
      title: 'Compiler',
      items: [
        'Ora MLIR lowering and verification ops',
        'Sensei-IR backend integration',
        'MLIR optimization, CFG, metrics, and Z3',
      ],
      link: '/docs/compiler/field-guide',
      linkLabel: 'Field Guide',
    },
    {
      title: 'Research',
      items: [
        'Type system spec v0.11 PDF',
        'Comptime, SMT, and proof trust boundaries',
        'Refinement strategy and guard semantics',
      ],
      link: '/docs/research/research-snapshot',
      linkLabel: 'Snapshot',
    },
  ];

  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <h2>Focus Areas</h2>
          <p>Practical docs and research artifacts, side by side.</p>
        </div>
        <div className={styles.featureGrid}>
          {areas.map((area, idx) => (
            <div key={idx} className={styles.featureCard}>
              <h3>{area.title}</h3>
              <ul>
                {area.items.map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
              <Link to={area.link} className={styles.cardLink}>
                {area.linkLabel} →
              </Link>
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
        <div className={styles.sectionHeader}>
          <h2>Ora in Practice</h2>
          <p>Explicit errors, verification clauses, and predictable behavior.</p>
        </div>
        <div className={styles.codeContainer}>
          <pre className={styles.codeBlock}>
            <code className="language-ora">{`error InsufficientBalance(required: u256, available: u256);

contract Vault {
    storage var balances: map<NonZeroAddress, u256>;

    pub fn withdraw(amount: MinValue<u256, 1>)
        -> Result<u256, InsufficientBalance>
        requires balances[std.msg.sender()] >= amount
        ensures_ok(balances[std.msg.sender()] == old(balances[std.msg.sender()]) - amount)
    {
        let sender: NonZeroAddress = std.msg.sender();
        let current: u256 = balances[sender];
        if (current < amount) {
            return Err(InsufficientBalance(amount, current));
        }
        balances[sender] -= amount;
        return Ok(balances[sender]);
    }
}`}</code>
          </pre>
        </div>
        <div className={styles.codeActions}>
          <Link to="/docs/examples" className="button button--primary">
            Examples
          </Link>
          <Link to="/docs/getting-started" className="button button--outline button--primary">
            Try the Compiler
          </Link>
        </div>
      </div>
    </section>
  );
}

function ResearchStrip() {
  return (
    <section className={styles.researchStrip}>
      <div className="container">
        <div className={styles.researchGrid}>
          <div>
            <h2>Research Backbone</h2>
            <p>
              Formal specs, implementation baselines, and proof-boundary docs
              are linked directly to source artifacts.
            </p>
          </div>
          <div className={styles.researchLinks}>
            <Link to="/docs/research/type-system">Type System</Link>
            <Link to="/docs/research/comptime">Comptime</Link>
            <Link to="/docs/research/smt-verification">SMT Verification</Link>
            <Link to="/docs/research/refinement-types">Refinement Types</Link>
            <Link to="/docs/specs/type-system">Type System PDF</Link>
          </div>
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
        <p>Focused contributions: tests, docs, and compiler work.</p>

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
            <p>Help us keep docs aligned with compiler reality.</p>
            <Link href="https://github.com/oralang/Ora/blob/main/CONTRIBUTING.md">
              Contributing Guide →
            </Link>
          </div>

          <div className={styles.contributeCard}>
            <h3>💬 Discuss</h3>
            <p>Share ideas and ask questions.</p>
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
      description="Ora - smart contract language with explicit semantics, refinements, and verification-first compilation">
      <HomepageHeader />
      <main>
        <FocusAreas />
        <CodeExample />
        <ResearchStrip />
        <ContributeSection />
      </main>
    </Layout>
  );
}
