package io.github.gabryellima.libria.bridge

/** Landmarks da mão -> features `wrist_relative` (utils/helpers.py), no núcleo nativo. */
object WristRelativeFeatures {
    const val LANDMARKS = 21
    const val DIMENSION = LANDMARKS * 3

    /**
     * @param landmarks 21 pontos (x, y, z) intercalados, 63 floats.
     * @param out destino de 63 floats (pode ser o próprio `landmarks`).
     */
    fun compute(landmarks: FloatArray, out: FloatArray = FloatArray(DIMENSION)): FloatArray {
        require(landmarks.size == DIMENSION && out.size == DIMENSION) { "landmarks e out devem ter $DIMENSION floats" }
        LibriaNative.nativeWristRelative(landmarks, out)
        return out
    }
}
