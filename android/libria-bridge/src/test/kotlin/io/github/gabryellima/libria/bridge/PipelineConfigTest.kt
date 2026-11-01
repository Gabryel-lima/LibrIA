package io.github.gabryellima.libria.bridge

import kotlin.test.Test
import kotlin.test.assertEquals

class PipelineConfigTest {
    @Test
    fun `padroes do Kotlin batem com os compilados no nucleo C++`() {
        // Também valida a ordem dos arrays de configuração do JNI.
        assertEquals(PipelineConfig.nativeDefaults(), PipelineConfig())
    }

    @Test
    fun `padroes do Kotlin batem com config settings py`() {
        val staticHold = Fixtures.array("pipeline").first { it.obj["name"]!!.text == "static_hold" }
        assertEquals(configFrom(staticHold.obj["config"]!!.obj), PipelineConfig())
    }
}
