/**
 * Global property type declarations.
 * @package    @epicurrents/onnx-service
 * @copyright  2024 Sampsa Lohi
 * @license    Apache-2.0
 */

/* eslint-disable */
/**
 * Epicurrents properties available in the global scope.
 */
type EpicurrentsGlobal = {
    /**
     * Master event bus for broadcasting application events.
     */
    EVENT_BUS: import('#events/types').ScopedEventBus | null
    /**
     * Runtime state manager of the initiated application (null before initiation).
     * @remarks
     * This property is required by the core application.
     */
    RUNTIME: import('@epicurrents/core/dist/types/application').StateManager | null
}
declare global {
    /** Path where WebPack serves its public assets (js) from. */
    let __webpack_public_path__: string
    interface Window {
        /**
         * Runtime state manager of the initiated application. Having the runtime accessible in the window object is a
         * workaround for cases, where different modules may implement different versions of the core package and thus
         * the imported `SETTINGS` may not point to the same object.
         *
         * If the `core` dependency of all the modules point to the same package at the time of compilation, the
         * imported runtime `state` will also point to the same object and stay synchronized between the modules.
         * See documentation for more details.
         */
        __EPICURRENTS__: EpicurrentsGlobal
    }
}
export {} // Guarantees the global declaration to work.
