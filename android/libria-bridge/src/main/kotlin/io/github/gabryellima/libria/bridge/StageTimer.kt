package io.github.gabryellima.libria.bridge

/** Estatísticas de uma etapa, em milissegundos. */
data class StageStats(val count: Int, val p50Ms: Double, val p95Ms: Double, val maxMs: Double)

/**
 * Benchmark por etapa (captura, MediaPipe, features, núcleo...): guarda as
 * últimas [capacity] durações de cada etapa e calcula percentis.
 *
 * A F2 do plano exige medir antes de otimizar: é isto que diz onde está o
 * gargalo real. Thread-safe (a câmera grava, a interface lê).
 */
class StageTimer(private val capacity: Int = 120) {
    init {
        require(capacity > 0) { "capacity deve ser maior que zero" }
    }

    private class Ring(capacity: Int) {
        val values = LongArray(capacity)
        var next = 0
        var size = 0
    }

    private val stages = LinkedHashMap<String, Ring>()

    @Synchronized
    fun record(stage: String, nanos: Long) {
        require(nanos >= 0) { "duração negativa para $stage" }
        val ring = stages.getOrPut(stage) { Ring(capacity) }
        ring.values[ring.next] = nanos
        ring.next = (ring.next + 1) % capacity
        if (ring.size < capacity) ring.size++
    }

    /** Mede o bloco e grava a duração. */
    inline fun <T> measure(stage: String, clock: () -> Long = System::nanoTime, block: () -> T): T {
        val start = clock()
        try {
            return block()
        } finally {
            record(stage, clock() - start)
        }
    }

    @Synchronized
    fun snapshot(): Map<String, StageStats> = stages.mapValues { (_, ring) ->
        val sorted = ring.values.copyOf(ring.size).also { it.sort() }
        StageStats(
            count = sorted.size,
            p50Ms = nearestRank(sorted, 0.50) / 1e6,
            p95Ms = nearestRank(sorted, 0.95) / 1e6,
            maxMs = sorted.last() / 1e6,
        )
    }

    @Synchronized
    fun clear() = stages.clear()

    private fun nearestRank(sorted: LongArray, percentile: Double): Double {
        // Percentil pelo método nearest-rank: o menor valor com pelo menos
        // `percentile` das amostras abaixo ou igual a ele.
        val rank = kotlin.math.ceil(percentile * sorted.size).toInt().coerceIn(1, sorted.size)
        return sorted[rank - 1].toDouble()
    }
}
