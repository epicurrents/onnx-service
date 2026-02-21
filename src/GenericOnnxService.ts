/**
 * ONNX service. This class communicates with a worker running a ONNX model.
 * @package    epicurrents/onnx-service
 * @copyright  2024 Sampsa Lohi
 * @license    Apache-2.0
 */

import { GenericService } from '@epicurrents/core'
import { type SetupWorkerResponse, type WorkerResponse } from '@epicurrents/core/dist/types/service'
import { Log } from 'scoped-event-log'
import {
    type LoadingState,
    type LoadModelResponse,
    type OnnxRunResponse,
    type OnnxRunResults,
    type OnnxService,
} from '#types'

const SCOPE = 'OnnxService'

export default abstract class GenericOnnxService extends GenericService implements OnnxService {
    protected _callbacks: ((...results: unknown[]) => unknown)[] = []
    protected _isRunInProgress = false
    protected _modelState: LoadingState = 'not_loaded'
    protected _progress = {
        complete: 0,
        target: 0,
    }

    constructor (config?: { rootPath?: string }) {
        if (!window.__EPICURRENTS__?.RUNTIME) {
            Log.error(`Reference to core application runtime was not found.`, SCOPE)
        }
        const overrideWorker = window.__EPICURRENTS__?.RUNTIME?.getWorkerOverride('onnx')
        const worker = overrideWorker ? overrideWorker : new Worker(
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

    get initialSetupPromise () {
        if (!this._waiters.get('init')) {
            return Promise.resolve(true)
        }
        return this.awaitAction('init') as Promise<boolean>
    }

    get isRunInProgress () {
        return this._isRunInProgress
    }
    protected set isRunInProgress (value: boolean) {
        if (value === this._isRunInProgress) {
            return
        }
        this._setPropertyValue('isRunInProgress', value)
    }

    get modelLoadedPromise () {
        if (!this._waiters.get('load')) {
            return Promise.resolve(true)
        }
        return this.awaitAction('load') as Promise<boolean>
    }

    get modelState () {
        return this._modelState
    }
    protected set modelState (value: LoadingState) {
        this._setPropertyValue('modelState', value)
    }

    get runProgress () {
        return this._progress.target ? this._progress.complete/this._progress.target : 0
    }

    async cancelRun () {
        if (!this._isRunInProgress) {
            return false
        }
        this.isRunInProgress = false
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
        } catch (e: unknown) {
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
        } catch (e: unknown) {
            this._notifyWaiters('init', false)
            return false
        }
    }

    async pauseRun (duration?: number) {
        if (!this._isRunInProgress) {
            return false
        }
        this._initWaiters('pause')
        this.isRunInProgress = false
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
        if (this._isRunInProgress) {
            this.cancelRun()
        }
        const prevState = this.runProgress
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
        this.dispatchPropertyChangeEvent('runProgress', this.runProgress, prevState)
    }

    async resumeRun () {
        if (this._isRunInProgress) {
            return false
        }
        this.isRunInProgress = true
        this._notifyWaiters('pause', true)
        Log.debug(`Resuming the ONNX run.`, SCOPE)
        const commission = this._commissionWorker('resume')
        const response = await commission.promise as boolean
        return response
    }

    async run (samples: Float32Array[]) {
        if (this._isRunInProgress) {
            Log.error(`There is already a run in progress.`, SCOPE)
            return []
        }
        if (this._modelState === 'loading') {
            await this.modelLoadedPromise
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
        this.isRunInProgress = true
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
        this.isRunInProgress = false
        this._notifyWaiters('run', true)
        Log.debug(`ONNX run complete in ${((Date.now() - startTime)/1000).toFixed(1)} seconds.`, SCOPE)
        return response
    }
}
