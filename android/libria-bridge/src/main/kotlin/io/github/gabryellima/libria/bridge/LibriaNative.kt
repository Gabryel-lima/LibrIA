package io.github.gabryellima.libria.bridge

/**
 * Ponte JNI para `liblibria_jni.so` (native/libria_jni).
 *
 * Os métodos são registrados em `JNI_OnLoad` via `RegisterNatives`: se uma
 * assinatura aqui divergir da do C++, o carregamento falha com
 * [UnsatisfiedLinkError] em vez de estourar na primeira chamada.
 * Não use diretamente: a API pública é [LibriaPipeline] e [WristRelativeFeatures].
 */
internal object LibriaNative {
    const val LIBRARY_NAME = "libria_jni"

    init {
        System.loadLibrary(LIBRARY_NAME)
    }

    @JvmStatic
    external fun nativeCreate(
        doubles: DoubleArray,
        ints: IntArray,
        emitPartials: Boolean,
        unknownLabel: String,
        labelIndices: IntArray,
        labelNames: Array<String>,
        sequenceLength: Int,
        featureDim: Int,
        temporal: TemporalPredictor,
        static: StaticPredictor?,
    ): Long

    @JvmStatic external fun nativeDestroy(handle: Long)

    @JvmStatic external fun nativeProcessFrame(
        handle: Long,
        features: FloatArray?,
        timestampSeconds: Double,
        handPresent: Boolean,
    ): SignToken?

    @JvmStatic external fun nativeReset(handle: Long)

    @JvmStatic external fun nativeIsSigning(handle: Long): Boolean

    @JvmStatic external fun nativeLastEnergy(handle: Long): Double

    @JvmStatic external fun nativeWristRelative(landmarks: FloatArray, out: FloatArray)

    @JvmStatic external fun nativeDefaultDoubles(): DoubleArray

    @JvmStatic external fun nativeDefaultInts(): IntArray

    @JvmStatic external fun nativeDefaultEmitPartials(): Boolean

    @JvmStatic external fun nativeDefaultUnknownLabel(): String
}
