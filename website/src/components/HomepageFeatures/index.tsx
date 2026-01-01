import type { ReactNode } from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Development Status',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        <strong>Core Pipeline:</strong> Lexical analysis → Syntax analysis → Semantic analysis → HIR → MLIR (81 operations) → sensei-ir (SIR) → EVM Bytecode is functional.<br />
        <strong>Success Rate:</strong> 79% (76/96 examples passing). <strong>In Progress:</strong> sensei-ir lowering, for loops, enhanced error handling.
      </>
    ),
  },
  {
    title: 'Language Design',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        <strong>Error Handling:</strong> Zig-style <code>!T</code> error unions with explicit error declarations.<br />
        <strong>Memory Regions:</strong> Clear distinction between <code>storage</code>, <code>immutable</code>, and compile-time constants.<br />
        <strong>Safety:</strong> Explicit memory management and type safety by design.
      </>
    ),
  },
  {
    title: 'Implementation Notes',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        <strong>Built with Zig:</strong> Leveraging Zig's compile-time capabilities for meta-programming.<br />
        <strong>sensei-ir Backend:</strong> Compiles to sensei-ir (SIR), a bespoke EVM IR for optimal bytecode generation.<br />
        <strong>Formal Methods:</strong> Exploring mathematical proof systems for contract verification.
      </>
    ),
  },
];

function Feature({ title, Svg, description }: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <div className={styles.featureDescription}>{description}</div>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="text--center margin-bottom--lg">
          <Heading as="h2">Development Overview</Heading>
          <p>Current implementation status and design decisions</p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
