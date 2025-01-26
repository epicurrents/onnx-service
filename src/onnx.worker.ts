/**
 * Epicurrents generic ONNX worker.
 * @package    @epicurrents/onnx-service
 * @copyright  2024 Sampsa Lohi
 * @license    Apache-2.0
 */

import * as ort from 'onnxruntime-web'
import { Log } from 'scoped-event-log'
import { type WorkerMessage } from '@epicurrents/core/dist/types'
import { type OnnxRunProgress, type OnnxRunResponse } from './types'
import { sleep, validateCommissionProps } from '@epicurrents/core/dist/util'

ort.env.wasm.wasmPaths = typeof __webpack_public_path__ === 'string' ? __webpack_public_path__ : ''

const SCOPE = "onnx.worker"

// Example parameters.
const DIMENSIONS = [] as number[]
const INPUT = 'imageinput'
const OUTPUT = 'softmax'

const RUN = {
    continue: false,
    results: [] as {
        success: boolean|null
        value: (ort.InferenceSession.OnnxValueMapType | null)
    }[],
    sampleIdx: 0,
    samples: [] as Float32Array[],
    session: null as ort.InferenceSession | null,
}

onmessage = async (message: WorkerMessage) => {
    if (!message?.data?.action) {
        return
    }
    const action = message.data.action
    if (action === 'load-model') {
        const data = validateCommissionProps(
            message.data as WorkerMessage['data'] & {
                dimensions: number[]
                path: string
            },
            {
                dimensions: 'Array',
                path: 'String',
            }
        )
        if (!data) {
            return
        }
        if (await loadModel(data.path, data.dimensions)) {
            postMessage({
                action: action,
                rn: message.data.rn,
                success: true,
            })
        } else {
            postMessage({
                action: action,
                rn: message.data.rn,
                success: false,
            })
        }
    } else if (action === 'cancel') {
        const result = cancelRun()
        if (!result) {
            postMessage({
                action: action,
                reason: `There was no running session to pause.`,
                rn: message.data.rn,
                success: false,
            })
        } else {
            postMessage({
                action: action,
                rn: message.data.rn,
                success: true,
            })
        }
    } else if (action === 'pause') {
        const result = pause()
        if (!result) {
            postMessage({
                action: action,
                reason: `There was no running session to pause.`,
                rn: message.data.rn,
                success: false,
            })
        } else {
            postMessage({
                action: action,
                rn: message.data.rn,
                success: true,
            })
        }
    } else if (action === 'resume') {
        const result = resume()
        if (!result) {
            postMessage({
                action: action,
                reason: `There was no paused session to resume.`,
                rn: message.data.rn,
                success: false,
            })
        } else {
            postMessage({
                action: action,
                rn: message.data.rn,
                success: true,
            })
        }
    } else if (action === 'run') {
        const data = validateCommissionProps(
            message.data as WorkerMessage['data'] & { samples: Float32Array[] },
            {
                samples: 'Array',
            }
        )
        if (!data) {
            return
        }
        const result = await run(data.samples)
        if (result === null) {
            postMessage({
                action: action,
                reason: `Session has not been prepared yet.`,
                rn: message.data.rn,
                success: false,
            })
        } else {
            postMessage({
                action: action,
                results: result.map(pred => pred?.value ? pred.value[OUTPUT].data : null),
                rn: message.data.rn,
                success: true,
            } as OnnxRunResponse)
        }
    } else if (action === 'setup-worker') {
        const data = validateCommissionProps(
            message.data as WorkerMessage['data'] & { path: string },
            {
                path: 'String',
            }
        )
        if (!data) {
            return
        }
        // Update the default path.
        ort.env.wasm.wasmPaths = data.path
        postMessage({
            action: action,
            rn: message.data.rn,
            success: true,
        })
    }
}

/**
 * Cancel the ongoing run and clear results.
 */
const cancelRun = () => {
    if (!RUN.session || !RUN.continue) {
        return false
    }
    resetRun()
    RUN.results = []
    return true
}

/**
 * Load the given model using the given Tensor dimensions.
 * @param path - Path to the ONNX model.
 * @param dimensions - Dimensions of the input Tensor this model uses.
 * @returns Promise that resolves with the success of the loading operation.
 */
const loadModel = async (path: string, dimensions: number[]) => {

    try {
        RUN.session = await ort.InferenceSession.create(path)
        DIMENSIONS.splice(0, DIMENSIONS.length, ...dimensions)
        Log.debug(`ONNX model loaded, input names for model are: ${RUN.session.inputNames.join(', ')}.`, SCOPE)
        return true
    } catch (e: unknown) {
        Log.error(`Failed to load ONNX model.`, SCOPE, e as Error)
        return false
    }
}

/**
 * Pause an ongoing run. The run will be paused after the currently active inference is complete.
 * @returns Success of the operation.
 */
const pause = () => {
    if (!RUN.session || !RUN.continue) {
        return false
    }
    RUN.continue = false
    return true
}

/**
 * Reset run parameters.
 */
const resetRun = () => {
    RUN.continue = false
    RUN.results = []
    RUN.sampleIdx = 0
    RUN.samples = []
}

/**
 * Resume a paused run.
 * @returns Success of the operation.
 */
const resume = () => {
    if (!RUN.session || RUN.continue) {
        return false
    }
    RUN.continue = true
    run([])
    return true
}

/**
 * Run inference with the loaded model on the given set of `samples`. A `progress` event is sent after each completed sample.
 * @param samples - Array of Float32Array samples to process with the model. If continuing a paused run, give empty array.
 * @returns An array of values (corresponding to each input sample) returned by the model or null if the run fails.\
 *          An empty array is returned if no session is active.
 */
const run = async (samples: Float32Array[]) => {
    if (RUN.session) {
        if (!RUN.sampleIdx) {
            // This is a new run, set results and samples; otherwise continue from where we left.
            RUN.results = new Array(samples.length).fill({ success: null, value: null })
            RUN.samples = samples
        }
        for (; RUN.sampleIdx<samples.length; RUN.sampleIdx++) {
            if (!RUN.continue) {
                // Pause the run and return results so far.
                return RUN.results
            }
            const i = RUN.sampleIdx
            const sample = samples[i]
            // Create an input tensor from the sample.
            const data = new ort.Tensor(sample, DIMENSIONS)
            try {
                const result = await RUN.session.run({ [INPUT]: data })
                RUN.results[i].value = result
                RUN.results[i].success = true
            } catch (e: unknown) {
                Log.error(`Running inference on the given sample #${i} failed.`, SCOPE, e as Error)
                RUN.results[i].success = false
            }
            // Report progress as the number of samples that habe been processed.
            postMessage({
                action: 'progress',
                complete: i + 1,
                success: RUN.results.map(r => r.success),
            } as { action: 'progress' } & OnnxRunProgress)
            // Wait for an instant for a possible pause request to be handled.
            await sleep(10)
        }
    }
    // Save run results before resetting it.
    const results = RUN.results.splice(0)
    // Reset run parameters.
    resetRun()
    // Also clear the results after returning them.
    return results
}