/**
 * ONNX service. This class communicates with a worker running a ONNX model.
 * @package    epicurrents/onnx-service
 * @copyright  2024 Sampsa Lohi
 * @license    Apache-2.0
 */

import { GenericService } from '@epicurrents/core'
import { type SetupWorkerResponse, type WorkerResponse } from '@epicurrents/core/dist/types/service'
import { Log } from 'scoped-ts-log'
import {
    type LoadModelResponse,
    type OnnxRunResponse,
    type OnnxRunResults,
    type OnnxService,
} from '#types'

const SCOPE = 'OnnxService'

type LoadingState = 'error' | 'loaded' | 'loading' | 'not_loaded'

export default abstract class GenericOnnxService extends GenericService implements OnnxService {
    protected _callbacks: ((...results: unknown[]) => unknown)[] = []
    protected _modelState: LoadingState = 'not_loaded'
    protected _progress = {
        complete: 0,
        target: 0,
    }
    protected _runInProgress = false

    constructor (config?: { rootPath?: string }) {
        if (!window.__EPICURRENTS__?.RUNTIME) {
            Log.error(`Reference to core application runtime was not found.`, SCOPE)
        }
        const overrideWorker = window.__EPICURRENTS__?.RUNTIME?.WORKERS.get('onnx')
        const worker = overrideWorker ? overrideWorker() : new Worker(
            new URL(
                /* webpackChunkName: 'onnx.worker' */
                `./onnx.worker`,
                import.meta.url
            ),
            { type: 'module' }
        )
        Log.registerWorker(worker)
        super(SCOPE, worker)
        worker.addEventListener('message', this.handleWorkerResponse.bind(this))
        // Set up a map for initialization waiters.
        this._initWaiters('init')
        this.setupWorker(config)
    }

    get initialSetup () {
        if (!this._waiters.get('init')) {
            return Promise.resolve(true)
        }
        return this.awaitAction('init')
    }

    get modelLoading () {
        if (!this._waiters.get('load')) {
            return Promise.resolve(true)
        }
        return this.awaitAction('load')
    }

    get modelState () {
        return this._modelState
    }
    protected set modelState (value: LoadingState) {
        const prevState = this._modelState
        this._modelState = value
        this.onPropertyUpdate('model-state', value, prevState)
    }

    get runInProgress () {
        return this._runInProgress
    }
    protected set runInProgress (value: boolean) {
        if (value === this._runInProgress) {
            return
        }
        this._runInProgress = value
        this.onPropertyUpdate('run-in-progress', value)
    }

    get runProgress () {
        return this._progress.target ? this._progress.complete/this._progress.target : 0
    }

    async cancelRun () {
        if (!this._runInProgress) {
            return false
        }
        this.runInProgress = false
        this._notifyWaiters('run', false)
        Log.debug(`Cancelling the ongoing ONNX run.`, SCOPE)
        const commission = this._commissionWorker('cancel')
        const response = await commission.promise as boolean
        this.resetProgress()
        return response
    }

    handleWorkerResponse (message: WorkerResponse) {
        const data = message.data
        if (!data || !data.action) {
            return false
        }
        const commission = this._getCommissionForMessage(message)
        if (commission) {
            if (data.action === 'load-model') {
                if (data.success) {
                    commission.resolve(data.success as LoadModelResponse)
                } else if (commission.reject) {
                    commission.reject(data.error as string)
                }
                return true
            } else if (data.action === 'pause') {
                commission.resolve(data.success)
                return true
            } else if (data.action === 'progress') {
                this._progress.complete = data.complete as number
                return true
            } else if (data.action === 'resume') {
                commission.resolve(data.success)
                return true
            } else if (data.action === 'run') {
                if (data.success) {
                    commission.resolve(((data as OnnxRunResponse).results))
                } else if (commission.reject) {
                    commission.reject(data.error as string)
                }
                return true
            } else {
                return super._handleWorkerCommission(message)
            }
        }
        return false
    }

    async loadModel (path: string) {
        this.modelState = 'loading'
        // Set up a map for model loading waiters.
        this._initWaiters('load')
        const commission = this._commissionWorker(
            'load-model',
            new Map([
                ['path', path],
            ])
        )
        try {
            const response = await commission.promise as Promise<LoadModelResponse>
            this.modelState = 'loaded'
            // Notify possible waiters that loading is done.
            this._notifyWaiters('load', true)
            return response
        } catch (e: any) {
            this.modelState = 'error'
            this._notifyWaiters('load', false)
            return false
        }
    }

    async setupWorker (config?: { rootPath?: string }): Promise<SetupWorkerResponse> {
        const scriptPath = typeof __webpack_public_path__ === 'string' && __webpack_public_path__
                           ? __webpack_public_path__
                           : window.location.pathname
        // Default to a folder called 'onnx' in the asset path or root (HTML) file path.
        // WebWorker doesn't have access to window.location, so we have to do this here.
        const onnxDir = config?.rootPath || scriptPath.substring(0, scriptPath.lastIndexOf('/')) + '/onnx'
        const commission = this._commissionWorker(
            'setup-worker',
            new Map([
                ['path', onnxDir],
            ])
        )
        try {
            const response = await commission.promise as SetupWorkerResponse
            // Notify possible waiters that setup is done.
            this._notifyWaiters('init', true)
            return response
        } catch (e: any) {
            this._notifyWaiters('init', false)
            return false
        }
    }

    async pauseRun (duration?: number) {
        if (!this._runInProgress) {
            return false
        }
        this._initWaiters('pause')
        this.runInProgress = false
        Log.debug(`Pausing the ONNX run${ duration ? ` for ${duration} ms` : ''}.`, SCOPE)
        if (duration) {
            setTimeout(() => {
                this.resumeRun()
            }, duration)
        }
        const commission = this._commissionWorker('pause')
        const response = await commission.promise as boolean
        return response
    }

    resetProgress () {
        if (this._runInProgress) {
            this.cancelRun()
        }
        this._progress.complete = 0
        this._progress.target = 0
        for (const upd of this._actionWatchers) {
            // Inform all progress action watchers of the reset
            if (upd.actions.includes('run')) {
                upd.handler(
                    { ...this._progress, action: 'run', update: 'progress' }
                )
            }
        }
        this.onPropertyUpdate('progress')
    }

    async resumeRun () {
        if (this._runInProgress) {
            return false
        }
        this.runInProgress = true
        this._notifyWaiters('pause', true)
        Log.debug(`Resuming the ONNX run.`, SCOPE)
        const commission = this._commissionWorker('resume')
        const response = await commission.promise as boolean
        return response
    }

    async run (samples: Float32Array[]) {
        if (this._runInProgress) {
            Log.error(`There is already a run in progress.`, SCOPE)
            return []
        }
        if (this._modelState === 'loading') {
            await this.modelLoading
        } else if (this._modelState !== 'loaded') {
            Log.error(`Cannot run inference before a model has been successfully loaded.`, SCOPE)
            return []
        }
        if (!samples.length) {
            Log.error(`Cannot run inference on an empty set of samples.`, SCOPE)
            return []
        }
        this.resetProgress()
        this._progress.target = samples.length
        this.runInProgress = true
        // Measure time for debug.
        const startTime = Date.now()
        Log.debug(`Starting a new ONNX run.`, SCOPE)
        this._initWaiters('run')
        const commission = this._commissionWorker(
            'run',
            new Map([
                ['samples', samples],
            ])
        )
        const response = await commission.promise as OnnxRunResults
        this.runInProgress = false
        this._notifyWaiters('run', true)
        Log.debug(`ONNX run complete in ${((Date.now() - startTime)/1000).toFixed(1)} seconds.`, SCOPE)
        return response
    }
}
