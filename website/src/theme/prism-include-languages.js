import oraGrammar from './prism-ora';

export default function prismIncludeLanguages(PrismObject) {
    // Load our custom Ora language grammar
    oraGrammar(PrismObject);
} 