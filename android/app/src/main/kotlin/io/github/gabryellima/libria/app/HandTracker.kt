package io.github.gabryellima.libria.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import io.github.gabryellima.libria.bridge.FpsMeter
import io.github.gabryellima.libria.bridge.LibriaPipeline
import io.github.gabryellima.libria.bridge.SignToken
import io.github.gabryellima.libria.bridge.StageStats
import io.github.gabryellima.libria.bridge.StageTimer
import io.github.gabryellima.libria.bridge.TemporalPredictor
import io.github.gabryellima.libria.bridge.WristRelativeFeatures
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.util.concurrent.ConcurrentHashMap

/** O que a tela mostra. `landmarks`: 21 pontos (x, y, z) normalizados na imagem analisada. */
data class TrackerState(
    val landmarks: FloatArray? = null,
    val imageWidth: Int = 0,
    val imageHeight: Int = 0,
    val fps: Double = 0.0,
    val energy: Double = 0.0,
    val signing: Boolean = false,
    val lastToken: SignToken? = null,
    val stages: Map<String, StageStats> = emptyMap(),
    val error: String? = null,
)

/**
 * Câmera -> MediaPipe Hand Landmarker -> features (C++) -> pipeline (C++).
 *
 * F2: ainda sem modelo de Libras. O pipeline roda com um preditor uniforme, o que
 * exercita movimento e segmentação no aparelho real (sinais encerrados saem como
 * DESCONHECIDO). O objetivo desta fase é medir onde o tempo é gasto.
 *
 * Etapas medidas (StageTimer): "conversao" (ImageProxy -> Bitmap rotacionado),
 * "mediapipe" (da submissão ao resultado), "features" e "nucleo".
 */
class HandTracker(context: Context) : AutoCloseable {

    private val _state = MutableStateFlow(TrackerState())
    val state: StateFlow<TrackerState> = _state

    private val timer = StageTimer()
    private val fpsMeter = FpsMeter()
    private val submitted = ConcurrentHashMap<Long, Long>()  // timestamp do quadro (ms) -> nanoTime da submissão
    private var lastFrameTimeMs = 0L

    private val raw = FloatArray(WristRelativeFeatures.DIMENSION)
    private val features = FloatArray(WristRelativeFeatures.DIMENSION)

    // O pipeline é de uma thread só: processFrame (thread do MediaPipe) e close (thread
    // principal) são serializados por este lock.
    private val pipelineLock = Any()
    private val pipeline = LibriaPipeline(
        temporalPredictor = UNIFORM_TEMPORAL,
        labelMap = mapOf(0 to "J", 1 to "Z"),
    )

    private val landmarker: HandLandmarker = HandLandmarker.createFromOptions(
        context,
        HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath(MODEL_ASSET)
                    .setDelegate(Delegate.CPU)
                    .build(),
            )
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumHands(1)
            // Mesmos limiares do Python (INFERENCE_CONFIG e mp.solutions.hands).
            .setMinHandDetectionConfidence(0.3f)
            .setMinTrackingConfidence(0.5f)
            .setResultListener(this::onResult)
            .setErrorListener { error -> _state.value = _state.value.copy(error = error.message) }
            .build(),
    )

    /** Chamado na thread de análise do CameraX, um quadro por vez. */
    fun analyze(image: ImageProxy) {
        val bitmap = try {
            timer.measure("conversao") { image.toBitmap().rotated(image.imageInfo.rotationDegrees) }
        } finally {
            image.close()
        }
        // Sem espelhar: o Python entrega ao MediaPipe o quadro como a câmera fornece.
        // Só o desenho na tela é espelhado (câmera frontal). Hipótese a confirmar na F2.
        val frameTimeMs = nextFrameTimeMs()
        submitted[frameTimeMs] = System.nanoTime()
        try {
            landmarker.detectAsync(BitmapImageBuilder(bitmap).build(), frameTimeMs)
        } catch (closed: IllegalStateException) {
            // Quadro que chegou durante o encerramento da tela, depois de close().
            submitted.remove(frameTimeMs)
        }
    }

    private fun onResult(result: HandLandmarkerResult, input: MPImage) {
        val now = System.nanoTime()
        submitted.remove(result.timestampMs())?.let { timer.record("mediapipe", now - it) }

        val hand = result.landmarks().firstOrNull()
        if (hand != null) {
            hand.forEachIndexed { i, point ->
                raw[3 * i] = point.x()
                raw[3 * i + 1] = point.y()
                raw[3 * i + 2] = point.z()
            }
            timer.measure("features") { WristRelativeFeatures.compute(raw, features) }
        }

        val (token, energy, signing) = synchronized(pipelineLock) {
            if (pipeline.isClosed) return
            val token = timer.measure("nucleo") {
                pipeline.processFrame(if (hand != null) features else null, result.timestampMs() / 1000.0)
            }
            Triple(token, pipeline.lastEnergy, pipeline.isSigning)
        }

        val previous = _state.value
        _state.value = previous.copy(
            landmarks = hand?.let { raw.copyOf() },
            imageWidth = input.width,
            imageHeight = input.height,
            fps = fpsMeter.tick(now),
            energy = energy,
            signing = signing,
            lastToken = token ?: previous.lastToken,
            stages = timer.snapshot(),
        )
    }

    /** O MediaPipe exige timestamps estritamente crescentes no modo LIVE_STREAM. */
    private fun nextFrameTimeMs(): Long {
        val now = SystemClock.uptimeMillis()
        lastFrameTimeMs = if (now > lastFrameTimeMs) now else lastFrameTimeMs + 1
        return lastFrameTimeMs
    }

    override fun close() {
        landmarker.close()
        synchronized(pipelineLock) { pipeline.close() }
    }

    private fun Bitmap.rotated(degrees: Int): Bitmap {
        if (degrees == 0) return this
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    companion object {
        const val MODEL_ASSET = "hand_landmarker.task"

        /** Sem modelo de Libras na F2: distribuição uniforme entre J e Z. */
        private val UNIFORM_TEMPORAL = TemporalPredictor { _, _, _ -> doubleArrayOf(0.5, 0.5) }
    }
}
