package io.github.gabryellima.libria.bridge

/**
 * Quadros por segundo numa janela deslizante de tempo.
 * O relógio é injetado (nanossegundos) para ser testável.
 */
class FpsMeter(private val windowNanos: Long = 1_000_000_000L) {
    init {
        require(windowNanos > 0) { "windowNanos deve ser maior que zero" }
    }

    private val timestamps = ArrayDeque<Long>()

    /** Registra um quadro e devolve o FPS atual (0 até haver dois quadros). */
    @Synchronized
    fun tick(nowNanos: Long): Double {
        timestamps.addLast(nowNanos)
        while (timestamps.size > 1 && nowNanos - timestamps.first() > windowNanos) {
            timestamps.removeFirst()
        }
        if (timestamps.size < 2) return 0.0
        val span = timestamps.last() - timestamps.first()
        return if (span <= 0) 0.0 else (timestamps.size - 1) * 1e9 / span
    }
}
