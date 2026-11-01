package io.github.gabryellima.libria.bridge

/**
 * Modelo temporal: recebe `frames x featureDim` floats contíguos e devolve a
 * distribuição de probabilidade por classe.
 *
 * Chamado na mesma thread de [LibriaPipeline.processFrame]. O array recebido é
 * um buffer reutilizado: copie-o se precisar guardar o conteúdo. Uma exceção
 * lançada aqui chega intacta a quem chamou `processFrame`.
 */
fun interface TemporalPredictor {
    fun predict(sequence: FloatArray, frames: Int, featureDim: Int): DoubleArray
}

/** Resposta do modelo estático para um quadro. */
data class StaticPrediction(val label: String, val confidence: Double)

/**
 * Modelo estático: recebe as features de um quadro (buffer reutilizado, como em
 * [TemporalPredictor]).
 */
fun interface StaticPredictor {
    fun predict(features: FloatArray): StaticPrediction
}
