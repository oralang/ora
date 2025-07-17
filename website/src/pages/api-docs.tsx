import React, { useEffect } from 'react';
import Layout from '@theme/Layout';

export default function ApiDocsRedirect(): React.ReactElement {
    useEffect(() => {
        // Redirect to the API documentation
        window.location.href = '/api-docs/index.html';
    }, []);

    return (
        <Layout
            title="API Reference"
            description="Ora compiler and standard library API documentation">
            <div style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                height: '50vh',
                flexDirection: 'column',
                textAlign: 'center'
            }}>
                <h1>Redirecting to API Documentation...</h1>
                <p>
                    If you are not automatically redirected, please{' '}
                    <a href="/api-docs/index.html">click here</a>.
                </p>
            </div>
        </Layout>
    );
} 