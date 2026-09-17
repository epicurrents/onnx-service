/**
 * Library build — emits the ESM `dist/` that every consumer imports.
 *
 * Module structure is preserved one-to-one with `src/`, so a consumer's bundler can still
 * tree-shake at module granularity. Type declarations are emitted separately by `build:types`;
 * this build emits JavaScript only.
 *
 * The worker is an entry of its own rather than being inlined the way the readers inline theirs:
 * the ONNX runtime fetches its WebAssembly at run time from a path the host supplies, so a worker
 * baked into a source string would carry 24 MB of it as base64 for nothing.
 * @package    epicurrents/onnx-service
 * @copyright  2026 Sampsa Lohi
 * @license    Apache-2.0
 */
import { defineConfig } from 'vite'
import { ALIASES, abs, externalDependencies } from './vite.shared.mjs'

export default defineConfig({
    build: {
        lib: {
            entry: {
                'index': abs('./src/index.ts'),
                'onnx.worker': abs('./src/onnx.worker.ts'),
            },
            formats: ['es'],
        },
        minify: false,
        outDir: abs('./dist'),
        emptyOutDir: true,
        target: 'esnext',
        rollupOptions: {
            external: externalDependencies,
            output: {
                preserveModules: true,
                preserveModulesRoot: abs('./src'),
                entryFileNames: '[name].js',
            },
        },
    },
    resolve: {
        alias: ALIASES,
    },
})
