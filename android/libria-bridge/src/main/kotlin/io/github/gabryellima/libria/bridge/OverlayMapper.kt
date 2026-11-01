package io.github.gabryellima.libria.bridge

/** Ponto em pixels da view. */
data class ViewPoint(val x: Float, val y: Float)

/**
 * Converte coordenadas normalizadas da imagem analisada (0..1) para pixels de
 * uma view em modo FILL_CENTER (escala até cobrir e corta o excesso), como o
 * `PreviewView` do CameraX usa por padrão.
 *
 * `mirror = true` espelha na horizontal: a pré-visualização da câmera frontal é
 * espelhada, mas a imagem enviada ao MediaPipe não (mesmo comportamento da
 * coleta em Python, que não espelha o quadro).
 */
object OverlayMapper {
    fun map(
        x: Float,
        y: Float,
        imageWidth: Int,
        imageHeight: Int,
        viewWidth: Int,
        viewHeight: Int,
        mirror: Boolean,
    ): ViewPoint {
        require(imageWidth > 0 && imageHeight > 0 && viewWidth > 0 && viewHeight > 0) { "dimensões devem ser positivas" }
        val scale = maxOf(viewWidth.toFloat() / imageWidth, viewHeight.toFloat() / imageHeight)
        val offsetX = (viewWidth - imageWidth * scale) / 2f
        val offsetY = (viewHeight - imageHeight * scale) / 2f
        val px = x * imageWidth * scale + offsetX
        val py = y * imageHeight * scale + offsetY
        return ViewPoint(if (mirror) viewWidth - px else px, py)
    }
}
