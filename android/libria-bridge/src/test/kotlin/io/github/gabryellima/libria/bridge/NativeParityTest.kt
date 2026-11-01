package io.github.gabryellima.libria.bridge

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Paridade Kotlin -> JNI -> C++ com o pipeline Python, nas mesmas fixtures do
 * teste nativo. Valida a ponte: arrays, strings, enums, callbacks e o retorno
 * dos tokens. Comparação exata, bit a bit.
 */
class NativeParityTest {

    @Test
    fun `features wrist_relative pelo JNI batem bit a bit`() {
        Fixtures.array("features").forEachIndexed { case, item ->
            val landmarks = item.obj["landmarks"]!!.arr.flatMap { point -> point.arr.map { it.exactFloat } }.toFloatArray()
            val expected = item.obj["expected"]!!.arr.toFloatArray()
            val actual = WristRelativeFeatures.compute(landmarks)
            for (k in expected.indices) {
                assertEquals(expected[k].toRawBits(), actual[k].toRawBits(), "caso $case, k=$k: ${actual[k]} != ${expected[k]}")
            }
        }
    }

    @Test
    fun `todos os cenarios do pipeline batem com o Python`() {
        val scenarios = Fixtures.array("pipeline")
        assertTrue(scenarios.size >= 10, "fixtures sem cenários suficientes")
        scenarios.forEach { runScenario(it.obj) }
    }

    private fun runScenario(scenario: kotlinx.serialization.json.JsonObject) {
        val name = scenario["name"]!!.text
        val dim = scenario["feature_dim"]!!.int
        val frames: List<FloatArray?> = scenario["frames"]!!.arr.map { frame ->
            val f = frame.obj["f"]!!
            if (f.isNull) null else f.arr.toFloatArray()
        }
        val timestamps = scenario["frames"]!!.arr.map { it.obj["t"]!!.num }
        val labelMap = scenario["label_map"]!!.obj.mapKeys { it.key.toInt() }.mapValues { it.value.text }

        val temporalCalls = scenario["temporal_calls"]!!.arr
        var temporalIndex = 0
        val temporal = TemporalPredictor { sequence, count, featureDim ->
            check(temporalIndex < temporalCalls.size) { "$name: chamada temporal a mais" }
            val call = temporalCalls[temporalIndex].obj
            val expectedFrames = call["input_frames"]!!.arr
            check(count == expectedFrames.size && featureDim == dim) { "$name: forma da entrada temporal" }
            expectedFrames.forEachIndexed { step, index ->
                val source = frames[index.int]!!
                for (k in 0 until dim) {
                    check(sequence[step * dim + k].toRawBits() == source[k].toRawBits()) {
                        "$name: chamada temporal $temporalIndex, passo $step difere do quadro ${index.int}"
                    }
                }
            }
            temporalIndex++
            call["output"]!!.arr.map { it.num }.toDoubleArray()
        }

        val staticCalls = scenario["static_calls"]!!.arr
        var staticIndex = 0
        val static = if (scenario["has_static_predictor"]!!.flag) {
            StaticPredictor { features ->
                check(staticIndex < staticCalls.size) { "$name: chamada estática a mais" }
                val call = staticCalls[staticIndex].obj
                val source = frames[call["input_frames"]!!.int]!!
                check(features.size == dim && (0 until dim).all { features[it].toRawBits() == source[it].toRawBits() }) {
                    "$name: chamada estática $staticIndex recebeu o quadro errado"
                }
                staticIndex++
                val output = call["output"]!!.obj
                StaticPrediction(output["label"]!!.text, output["confidence"]!!.num)
            }
        } else {
            null
        }

        LibriaPipeline(
            temporalPredictor = temporal,
            labelMap = labelMap,
            sequenceLength = scenario["sequence_length"]!!.int,
            featureDim = dim,
            staticPredictor = static,
            config = configFrom(scenario["config"]!!.obj),
        ).use { pipeline ->
            val expected = scenario["expected"]!!.arr
            frames.forEachIndexed { f, frame ->
                val token = pipeline.processFrame(frame, timestamps[f], handPresent = frame != null)
                val exp = expected[f].obj
                assertEquals(exp["energy"]!!.num, pipeline.lastEnergy, "$name quadro $f: energia")
                assertEquals(exp["signing"]!!.flag, pipeline.isSigning, "$name quadro $f: isSigning")
                val expToken = exp["token"]!!
                if (expToken.isNull) {
                    assertNull(token, "$name quadro $f: C++ emitiu token e o Python não")
                } else {
                    val t = expToken.obj
                    val actual = token ?: error("$name quadro $f: Python emitiu token e o C++ não")
                    assertEquals(t["label"]!!.text, actual.label, "$name quadro $f: rótulo")
                    assertEquals(t["confidence"]!!.num, actual.confidence, "$name quadro $f: confiança")
                    assertEquals(t["start_time"]!!.num, actual.startTime, "$name quadro $f: início")
                    assertEquals(t["end_time"]!!.num, actual.endTime, "$name quadro $f: fim")
                    assertEquals(t["source"]!!.text, actual.source.name.lowercase(), "$name quadro $f: origem")
                    assertEquals(t["state"]!!.text, actual.state.name.lowercase(), "$name quadro $f: estado")
                    assertEquals(t["frame_count"]!!.int, actual.frameCount, "$name quadro $f: frameCount")
                }
            }
        }
        assertEquals(temporalCalls.size, temporalIndex, "$name: número de chamadas temporais")
        assertEquals(staticCalls.size, staticIndex, "$name: número de chamadas estáticas")
    }
}
