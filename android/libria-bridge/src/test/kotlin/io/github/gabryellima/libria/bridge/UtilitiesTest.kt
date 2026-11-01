package io.github.gabryellima.libria.bridge

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class UtilitiesTest {
    @Test
    fun `StageTimer calcula percentis nearest-rank e respeita a janela`() {
        val timer = StageTimer(capacity = 10)
        (1..20).forEach { timer.record("mediapipe", it * 1_000_000L) }  // só os 10 últimos: 11..20 ms
        val stats = timer.snapshot().getValue("mediapipe")
        assertEquals(10, stats.count)
        assertEquals(15.0, stats.p50Ms)
        assertEquals(20.0, stats.p95Ms)
        assertEquals(20.0, stats.maxMs)
        assertFailsWith<IllegalArgumentException> { timer.record("x", -1) }
    }

    @Test
    fun `StageTimer measure usa o relogio injetado`() {
        val timer = StageTimer()
        var now = 0L
        val result = timer.measure("features", clock = { now }) { now += 3_000_000L; 42 }
        assertEquals(42, result)
        assertEquals(3.0, timer.snapshot().getValue("features").p50Ms)
    }

    @Test
    fun `FpsMeter mede quadros na janela deslizante`() {
        val meter = FpsMeter(windowNanos = 1_000_000_000L)
        assertEquals(0.0, meter.tick(0))
        var fps = 0.0
        for (i in 1..60) fps = meter.tick(i * 33_333_333L)  // ~30 FPS por 2 s
        assertEquals(30.0, fps, 0.05)
    }

    @Test
    fun `OverlayMapper em FILL_CENTER corta o excesso e espelha`() {
        // Imagem 480x640 (retrato) numa view 1080x1920: escala 2.25 x 3.0 -> usa 3.0,
        // largura 1440 e corte de 180 px de cada lado.
        val center = OverlayMapper.map(0.5f, 0.5f, 480, 640, 1080, 1920, mirror = false)
        assertEquals(540f, center.x, 1e-3f)
        assertEquals(960f, center.y, 1e-3f)
        val left = OverlayMapper.map(0f, 0f, 480, 640, 1080, 1920, mirror = false)
        assertEquals(-180f, left.x, 1e-3f)
        val mirrored = OverlayMapper.map(0f, 0f, 480, 640, 1080, 1920, mirror = true)
        assertEquals(1260f, mirrored.x, 1e-3f)
    }
}
