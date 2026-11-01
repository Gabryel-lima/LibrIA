package io.github.gabryellima.libria.bridge

/**
 * Espelho de `TEMPORAL_PIPELINE_CONFIG` (config/settings.py) e de
 * `libria::PipelineConfig`. Os padrões são conferidos nos testes contra o
 * núcleo nativo, que por sua vez é conferido contra o Python.
 */
data class PipelineConfig(
    val motionSmoothing: Double = 0.6,
    val motionStartThreshold: Double = 0.012,
    val motionEndThreshold: Double = 0.006,
    val minStartFrames: Int = 2,
    val minEndFrames: Int = 5,
    val minSegmentFrames: Int = 8,
    val minDurationSeconds: Double = 0.2,
    val maxDurationSeconds: Double = 4.0,
    val maxAbsentFrames: Int = 5,
    val smoothingWindow: Int = 5,
    val duplicateWindowSeconds: Double = 1.0,
    val emitPartialTokens: Boolean = true,
    val partialIntervalFrames: Int = 5,
    val temporalConfidenceThreshold: Double = 0.85,
    val staticConfidenceThreshold: Double = 0.75,
    val staticIntervalFrames: Int = 20,
    val staticCooldownSeconds: Double = 0.5,
    val unknownLabel: String = "DESCONHECIDO",
) {
    // A ordem dos arrays é a de DoubleField/IntField em native/libria_jni/src/libria_jni.cpp.
    internal fun nativeDoubles(): DoubleArray = doubleArrayOf(
        motionSmoothing,
        motionStartThreshold,
        motionEndThreshold,
        minDurationSeconds,
        maxDurationSeconds,
        duplicateWindowSeconds,
        temporalConfidenceThreshold,
        staticConfidenceThreshold,
        staticCooldownSeconds,
    )

    internal fun nativeInts(): IntArray = intArrayOf(
        minStartFrames,
        minEndFrames,
        minSegmentFrames,
        maxAbsentFrames,
        smoothingWindow,
        partialIntervalFrames,
        staticIntervalFrames,
    )

    companion object {
        /** Os padrões compilados no núcleo C++, lidos pelo JNI. */
        fun nativeDefaults(): PipelineConfig {
            val d = LibriaNative.nativeDefaultDoubles()
            val i = LibriaNative.nativeDefaultInts()
            return PipelineConfig(
                motionSmoothing = d[0],
                motionStartThreshold = d[1],
                motionEndThreshold = d[2],
                minDurationSeconds = d[3],
                maxDurationSeconds = d[4],
                duplicateWindowSeconds = d[5],
                temporalConfidenceThreshold = d[6],
                staticConfidenceThreshold = d[7],
                staticCooldownSeconds = d[8],
                minStartFrames = i[0],
                minEndFrames = i[1],
                minSegmentFrames = i[2],
                maxAbsentFrames = i[3],
                smoothingWindow = i[4],
                partialIntervalFrames = i[5],
                staticIntervalFrames = i[6],
                emitPartialTokens = LibriaNative.nativeDefaultEmitPartials(),
                unknownLabel = LibriaNative.nativeDefaultUnknownLabel(),
            )
        }
    }
}
