package io.github.gabryellima.libria.app

import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import io.github.gabryellima.libria.bridge.OverlayMapper
import java.util.Locale
import java.util.concurrent.Executors

/** Pré-visualização da câmera frontal, landmarks por cima e painel de medições. */
@Composable
fun CameraScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val tracker = remember { HandTracker(context) }
    val state by tracker.state.collectAsState()
    val previewView = remember { PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER } }

    // O rastreador vive enquanto a tela existir; a câmera pode ser religada sem recriá-lo.
    DisposableEffect(tracker) {
        onDispose { tracker.close() }
    }

    DisposableEffect(lifecycleOwner) {
        val executor = Executors.newSingleThreadExecutor()
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { it.setAnalyzer(executor, tracker::analyze) }
            provider.unbindAll()
            provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            if (providerFuture.isDone) providerFuture.get().unbindAll()
            executor.shutdown()
        }
    }

    Box(Modifier.fillMaxSize()) {
        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())

        Canvas(Modifier.fillMaxSize()) {
            val points = state.landmarks ?: return@Canvas
            if (state.imageWidth <= 0 || state.imageHeight <= 0) return@Canvas
            for (i in 0 until points.size / 3) {
                val p = OverlayMapper.map(
                    points[3 * i], points[3 * i + 1],
                    state.imageWidth, state.imageHeight,
                    size.width.toInt(), size.height.toInt(),
                    mirror = true,  // a pré-visualização da câmera frontal é espelhada
                )
                drawCircle(Color.Cyan, radius = 8f, center = Offset(p.x, p.y))
            }
        }

        Column(
            Modifier
                .padding(12.dp)
                .background(Color(0x99000000))
                .padding(8.dp),
        ) {
            val line: @Composable (String) -> Unit = { text -> Text(text, color = Color.White, fontSize = 13.sp) }
            line(String.format(Locale.US, "FPS %.1f   energia %.4f", state.fps, state.energy))
            line(if (state.signing) "● sinal em execução" else "○ mão parada / ausente")
            state.lastToken?.let { line("último token: ${it.label} (${it.state.name.lowercase()}, ${"%.2f".format(Locale.US, it.confidence)})") }
            state.stages.forEach { (stage, stats) ->
                line(String.format(Locale.US, "%-10s p50 %5.1f ms  p95 %5.1f ms", stage, stats.p50Ms, stats.p95Ms))
            }
            state.error?.let { line("erro: $it") }
            line("F2: sem modelo de Libras (preditor uniforme)")
        }
    }
}
