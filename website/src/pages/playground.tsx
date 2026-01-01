import React from 'react';
import Layout from '@theme/Layout';
import Playground from '@site/src/components/Playground';

export default function PlaygroundPage(): JSX.Element {
  return (
    <Layout
      title="Ora Playground"
      description="Try Ora syntax in your browser - Interactive code editor with syntax validation">
      <div style={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
        <Playground />
      </div>
    </Layout>
  );
}

