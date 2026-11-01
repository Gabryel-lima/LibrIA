package io.github.gabryellima.libria.bridge

/**
 * Pipeline temporal do núcleo nativo: movimento, segmentação, suavização,
 * deduplicação e fallback estático, com paridade bit a bit com o Python.
 *
 * Não é thread-safe: use um pipeline por thread. Libera memória nativa em
 * [close]; depois disso, qualquer chamada lança [IllegalStateException].
 *
 * @param labelMap índice da classe do modelo temporal -> rótulo.
 * @param staticPredictor opcional; sem ele, não há fallback estático.
 */
class LibriaPipeline(
    temporalPredictor: TemporalPredictor,
    labelMap: Map<Int, String>,
    val sequenceLength: Int = 30,
    val featureDim: Int = WristRelativeFeatures.DIMENSION,
    staticPredictor: StaticPredictor? = null,
    val config: PipelineConfig = PipelineConfig(),
) : AutoCloseable {

    private var handle: Long

    init {
        val entries = labelMap.entries.sortedBy { it.key }
        handle = LibriaNative.nativeCreate(
            config.nativeDoubles(),
            config.nativeInts(),
            config.emitPartialTokens,
            config.unknownLabel,
            IntArray(entries.size) { entries[it].key },
            Array(entries.size) { entries[it].value },
            sequenceLength,
            featureDim,
            temporalPredictor,
            staticPredictor,
        )
    }

    /**
     * Processa um quadro. `features == null` (ou `handPresent == false`)
     * significa mão ausente. Devolve um token quando houver o que reportar.
     */
    fun processFrame(features: FloatArray?, timestampSeconds: Double, handPresent: Boolean = true): SignToken? =
        LibriaNative.nativeProcessFrame(requireOpen(), features, timestampSeconds, handPresent && features != null)

    val isSigning: Boolean get() = LibriaNative.nativeIsSigning(requireOpen())

    val lastEnergy: Double get() = LibriaNative.nativeLastEnergy(requireOpen())

    fun reset() = LibriaNative.nativeReset(requireOpen())

    val isClosed: Boolean get() = handle == 0L

    override fun close() {
        val current = handle
        handle = 0L
        if (current != 0L) LibriaNative.nativeDestroy(current)
    }

    private fun requireOpen(): Long {
        check(handle != 0L) { "LibriaPipeline já foi fechado" }
        return handle
    }
}
