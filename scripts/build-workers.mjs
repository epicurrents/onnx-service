/**
 * Standalone worker bundles — the escape hatch from inlining.
 *
 * `dist/` inlines the worker as a Blob, which requires `worker-src blob:` in the consumer's content
 * security policy. A consumer that cannot grant it serves these files instead and registers a
 * URL-based factory, which takes precedence over the inlined default.
 *
 * Each worker is bundled separately because IIFE output cannot be code-split, and self-contained
 * because a worker resolves no bare specifiers of its own.
 * @package    epicurrents/onnx-service
 * @copyright  2026 Sampsa Lohi
 * @license    Apache-2.0
 */
import { build } from 'vite'
import { ALIASES, abs, minifyWorkerOutput } from '../vite.shared.mjs'

// This package's worker sits beside the sources rather than in a workers directory.
const WORKERS = ['onnx']

for (const name of WORKERS) {
    await build({
        configFile: false,
        logLevel: 'warn',
        build: {
            // Not Vite's library mode: it inlines every asset as a data URI, and the ONNX runtime's
            // WebAssembly files are 48 MB of them. They are fetched at run time instead, from the
            // path the host passes in the setup commission.
            assetsInlineLimit: 0,
            minify: false,
            outDir: abs('./umd'),
            emptyOutDir: false,
            target: 'esnext',
            rollupOptions: {
                input: abs(`./src/${name}.worker.ts`),
                output: {
                    entryFileNames: `${name}.worker.js`,
                    format: 'iife',
                    inlineDynamicImports: true,
                    name: 'EpiCWorker',
                },
            },
        },
        plugins: [minifyWorkerOutput()],
        resolve: {
            alias: ALIASES,
        },
    })
    console.log(`built umd/${name}.worker.js`)
}
