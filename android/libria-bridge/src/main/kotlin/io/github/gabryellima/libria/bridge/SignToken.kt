package io.github.gabryellima.libria.bridge

/** Ordem igual a `libria::TokenSource` (o JNI passa o ordinal). */
enum class TokenSource { STATIC, TEMPORAL }

/** Ordem igual a `libria::TokenState` (o JNI passa o ordinal). */
enum class TokenState { PARTIAL, FINAL, REJECTED }

/**
 * Unidade reconhecida, espelho de `src/inference/sign_token.py`.
 *
 * `PARTIAL` é hipótese durante o sinal; `FINAL`, sinal aceito; `REJECTED`, sinal
 * encerrado abaixo do limiar (rótulo DESCONHECIDO).
 */
data class SignToken(
    val label: String,
    val confidence: Double,
    val startTime: Double,
    val endTime: Double,
    val source: TokenSource,
    val state: TokenState,
    val frameCount: Int,
) {
    val durationSeconds: Double get() = (endTime - startTime).coerceAtLeast(0.0)

    companion object {
        /** Usado pelo JNI para construir o token. Não faz parte da API pública estável. */
        @JvmStatic
        fun fromNative(
            label: String,
            confidence: Double,
            startTime: Double,
            endTime: Double,
            source: Int,
            state: Int,
            frameCount: Int,
        ): SignToken = SignToken(
            label = label,
            confidence = confidence,
            startTime = startTime,
            endTime = endTime,
            source = TokenSource.entries[source],
            state = TokenState.entries[state],
            frameCount = frameCount,
        )
    }
}
