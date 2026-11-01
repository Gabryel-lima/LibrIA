package io.github.gabryellima.libria.bridge

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

class BridgeErrorHandlingTest {
    private val labels = mapOf(0 to "J", 1 to "Z")

    /** Movimento contínuo: abre um segmento e força chamadas ao preditor temporal. */
    private fun drive(pipeline: LibriaPipeline, frames: Int = 60) {
        val frame = FloatArray(3)
        for (i in 0 until frames) {
            frame[0] += 0.05f
            pipeline.processFrame(frame, i / 30.0)
        }
    }

    @Test
    fun `excecao do preditor Kotlin atravessa o C++ intacta`() {
        val failure = IllegalStateException("modelo indisponível")
        val pipeline = LibriaPipeline({ _, _, _ -> throw failure }, labels, sequenceLength = 4, featureDim = 3,
            config = PipelineConfig(partialIntervalFrames = 1))
        pipeline.use {
            val thrown = assertFailsWith<IllegalStateException> { drive(it) }
            assertSame(failure, thrown)
            // O pipeline continua utilizável depois da exceção.
            it.reset()
            assertEquals(false, it.isSigning)
        }
    }

    @Test
    fun `distribuicao vazia vira IllegalStateException`() {
        LibriaPipeline({ _, _, _ -> DoubleArray(0) }, labels, sequenceLength = 4, featureDim = 3,
            config = PipelineConfig(partialIntervalFrames = 1)).use {
            val thrown = assertFailsWith<IllegalStateException> { drive(it) }
            assertTrue("vazia" in thrown.message.orEmpty(), thrown.message)
        }
    }

    @Test
    fun `features com tamanho errado e configuracao invalida viram IllegalArgumentException`() {
        LibriaPipeline({ _, _, _ -> doubleArrayOf(0.5, 0.5) }, labels, featureDim = 3).use {
            assertFailsWith<IllegalArgumentException> { it.processFrame(FloatArray(5), 0.0) }
        }
        assertFailsWith<IllegalArgumentException> {
            LibriaPipeline({ _, _, _ -> doubleArrayOf(1.0) }, labels, config = PipelineConfig(partialIntervalFrames = 0))
        }
        assertFailsWith<IllegalArgumentException> {
            LibriaPipeline({ _, _, _ -> doubleArrayOf(1.0) }, labels,
                config = PipelineConfig(motionEndThreshold = 0.5, motionStartThreshold = 0.1))
        }
    }

    @Test
    fun `uso depois de close falha e close duplo e seguro`() {
        val pipeline = LibriaPipeline({ _, _, _ -> doubleArrayOf(1.0) }, labels)
        pipeline.close()
        pipeline.close()
        assertTrue(pipeline.isClosed)
        assertFailsWith<IllegalStateException> { pipeline.processFrame(null, 0.0) }
        assertFailsWith<IllegalStateException> { pipeline.lastEnergy }
    }

    @Test
    fun `muitos pipelines criados e fechados nao vazam referencias`() {
        // Com -Xcheck:jni, um vazamento de referência global ou local aparece como aviso/erro da JVM.
        repeat(500) {
            LibriaPipeline({ _, _, _ -> doubleArrayOf(0.9, 0.1) }, labels, sequenceLength = 4, featureDim = 3,
                staticPredictor = { StaticPrediction("A", 0.9) }, config = PipelineConfig(partialIntervalFrames = 1))
                .use { p -> drive(p, frames = 20) }
        }
    }

    @Test
    fun `preditor estatico com rotulo minusculo e normalizado`() {
        var calls = 0
        LibriaPipeline({ _, _, _ -> doubleArrayOf(0.5, 0.5) }, labels, featureDim = 3,
            staticPredictor = { calls++; StaticPrediction(" b ", 0.9) }).use { p ->
            val still = floatArrayOf(0.1f, 0.2f, 0.3f)
            val token = (0 until 20).mapNotNull { p.processFrame(still, it / 30.0) }.firstOrNull()
            assertNotNull(token)
            assertEquals("B", token.label)
            assertEquals(TokenSource.STATIC, token.source)
            assertEquals(1, calls)
        }
    }
}
