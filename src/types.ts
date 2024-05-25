/**
 * ONNX service types.
 * @package    epicurrents/onnx-service
 * @copyright  2024 Sampsa Lohi
 * @license    Apache-2.0
 */

import { AssetService } from "@epicurrents/core/dist/types"
import { SetupWorkerResponse, WorkerCommissionResponse } from "@epicurrents/core/dist/types/service"

/**
 * Returned value is `true` if loading was successful, `false` otherwise.
 */
export type LoadModelResponse = boolean
/**
 * Progress of a run operation on an array of samples.
 * - `complete`: Number of samples that have been completed.
 * - `success`: An array of one of the following values for each sample:
 *   - true means inference run was successful
 *   - false means there was an error
 *   - null means the sample hasn't been processed yet.
 */
export type OnnxRunProgress = {
    complete: number
    success: (boolean|null)[]
}
export type OnnxRunResponse = WorkerCommissionResponse & {
    /**
     * Array on inference run results for each of the input samples.
     */
    results: OnnxRunResults
}
/**
 * A array of ONNX model inference results for each of the given samples.
 */
export type OnnxRunResults = (Float32Array | string[] | Uint8Array | Int8Array | Uint16Array | Int16Array | Int32Array | BigInt64Array | Float64Array | Uint32Array | BigUint64Array | null)[]
/**
 * Service class for interacting with an ONNX model.
 */
export interface OnnxService extends AssetService {
    /** Returns a promise that resolves after the model is done loading (or immediately if no loading is underway). */
    modelLoading: Promise<boolean>
    /** Is there a run in progress. */
    runInProgress: boolean
    /** Current run progress as a fraction. */
    runProgress: number
    /**
     * Cancel the running model, clearing all progress and results.
     * @returns Promise that resolves with the success of the operation.
     */
    cancelRun (): Promise<boolean>
    /**
     * Load the ONNX model from the given path.
     * @return Promise that fulfills with result of the operation when loading is complete.
     */
    loadModel (path: string): Promise<LoadModelResponse>
    /**
     * Pause the running model. Pause takes effect after the currently active sample has been processed.
     * @param duration - Optional duration of the pause in milliseconds; run is automatically resumed after this time if not zero or empty.
     * @returns Promise that resolves with success of the operation once the run has bene paused.
     */
    pauseRun (duration?: number): Promise<boolean>
    /**
     * Resume the running model.
     * @returns Promise that resolves with the success of the operation.
     */
    resumeRun (): Promise<boolean>
    /**
     * Load ONNX Runtime Web and prepare the web worker for loading a model.
     * @param config - Optional configuration:
     *                 * `rootPath`: URL path to the directory holding ONNX scripts and binaries.
     * @return Promise that fulfills with result of the operation when peparation is complete.
     */
    setupWorker (config?: { rootPath?: string }): Promise<SetupWorkerResponse>
    /**
     * Reset all progress-related parameters, including the target.
     * Any active runs will be stopped.
     */
    resetProgress (): void
    /**
     * Run the active model.
     * @param samples - Samples to run inference on.
     * @returns Final result of the run as an array of individual results for each sample.
     */
    run (samples: Float32Array[]): Promise<OnnxRunResults>
}
export type OnnxServiceReject = (reason: string) => void
export type OnnxServiceResolve = (result: unknown) => void