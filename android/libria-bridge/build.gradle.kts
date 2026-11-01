import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// API Kotlin do núcleo nativo. Módulo JVM puro (sem dependência do Android),
// usado pelo :app. Nos testes, a biblioteca JNI é compilada para o host a
// partir de native/libria_jni e carregada pela JVM com -Xcheck:jni.

plugins {
    id("org.jetbrains.kotlin.jvm")
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

dependencies {
    testImplementation(kotlin("test"))
    testImplementation(platform("org.junit:junit-bom:5.11.4"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
    testImplementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
}

val repoRoot = rootDir.resolve("..").canonicalFile
val nativeJniDir = repoRoot.resolve("native/libria_jni")
val nativeCoreDir = repoRoot.resolve("native/libria_core")
val hostNativeBuildDir = layout.buildDirectory.dir("native-host")

// Compila native/libria_jni (+ núcleo) para a JVM do host.
val buildHostNative by tasks.registering(Exec::class) {
    description = "Compila a biblioteca JNI para o host, para os testes na JVM"
    inputs.dir(nativeJniDir.resolve("src"))
    inputs.file(nativeJniDir.resolve("CMakeLists.txt"))
    inputs.dir(nativeCoreDir.resolve("src"))
    inputs.dir(nativeCoreDir.resolve("include"))
    inputs.file(nativeCoreDir.resolve("CMakeLists.txt"))
    outputs.dir(hostNativeBuildDir)
    val buildDir = hostNativeBuildDir.get().asFile.absolutePath
    commandLine(
        "bash", "-c",
        "cmake -S '${nativeJniDir.absolutePath}' -B '$buildDir' -DCMAKE_BUILD_TYPE=Release -DLIBRIA_WERROR=ON " +
            "&& cmake --build '$buildDir'",
    )
}

tasks.test {
    useJUnitPlatform()
    dependsOn(buildHostNative)
    systemProperty("java.library.path", hostNativeBuildDir.get().asFile.absolutePath)
    systemProperty("libria.fixtures", repoRoot.resolve("native/fixtures/parity_fixtures.json").absolutePath)
    // Validação extra do uso da API JNI pela JVM (referências, exceções pendentes).
    jvmArgs("-Xcheck:jni")
    testLogging { events("failed"); exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL }
}
