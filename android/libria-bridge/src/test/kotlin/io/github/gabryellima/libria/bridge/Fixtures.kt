package io.github.gabryellima.libria.bridge

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File

/** Leitura de native/fixtures/parity_fixtures.json (gerado do pipeline Python). */
object Fixtures {
    val root: JsonObject by lazy {
        val path = System.getProperty("libria.fixtures") ?: error("propriedade libria.fixtures ausente")
        Json.parseToJsonElement(File(path).readText()).jsonObject.also {
            check(it["format_version"]!!.jsonPrimitive.int == 1) { "format_version não suportado" }
        }
    }

    fun array(key: String): JsonArray = root[key]!!.jsonArray
}

val JsonElement.obj: JsonObject get() = jsonObject
val JsonElement.arr: JsonArray get() = jsonArray
val JsonElement.num: Double get() = jsonPrimitive.double
val JsonElement.int: Int get() = jsonPrimitive.int
val JsonElement.flag: Boolean get() = jsonPrimitive.boolean
val JsonElement.text: String get() = jsonPrimitive.content
val JsonElement.isNull: Boolean get() = this is JsonNull

/** float32 gravado como double na fixture; confere que a conversão é exata. */
val JsonElement.exactFloat: Float
    get() {
        val value = num
        val result = value.toFloat()
        check(result.toDouble() == value) { "valor da fixture não é float32 exato: $value" }
        return result
    }

fun JsonArray.toFloatArray(): FloatArray = FloatArray(size) { this[it].exactFloat }

fun configFrom(json: JsonObject): PipelineConfig = PipelineConfig(
    motionSmoothing = json["motion_smoothing"]!!.num,
    motionStartThreshold = json["motion_start_threshold"]!!.num,
    motionEndThreshold = json["motion_end_threshold"]!!.num,
    minStartFrames = json["min_start_frames"]!!.int,
    minEndFrames = json["min_end_frames"]!!.int,
    minSegmentFrames = json["min_segment_frames"]!!.int,
    minDurationSeconds = json["min_duration_seconds"]!!.num,
    maxDurationSeconds = json["max_duration_seconds"]!!.num,
    maxAbsentFrames = json["max_absent_frames"]!!.int,
    smoothingWindow = json["smoothing_window"]!!.int,
    duplicateWindowSeconds = json["duplicate_window_seconds"]!!.num,
    emitPartialTokens = json["emit_partial_tokens"]!!.flag,
    partialIntervalFrames = json["partial_interval_frames"]!!.int,
    temporalConfidenceThreshold = json["temporal_confidence_threshold"]!!.num,
    staticConfidenceThreshold = json["static_confidence_threshold"]!!.num,
    staticIntervalFrames = json["static_interval_frames"]!!.int,
    staticCooldownSeconds = json["static_cooldown_seconds"]!!.num,
)
