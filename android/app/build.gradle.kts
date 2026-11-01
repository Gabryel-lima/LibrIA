import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import java.net.URI

// App Android da fase F2: câmera, MediaPipe Hand Landmarker, overlay, FPS e
// benchmark por etapa, com o núcleo C++ via JNI (native/libria_jni).
//
// ATENÇÃO: escrito sem acesso ao Android SDK nem ao Google Maven (bloqueados no
// ambiente em que foi criado). Nunca foi compilado. Ver android/README.md.

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "io.github.gabryellima.libria.app"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "io.github.gabryellima.libria"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0-f2"
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
        externalNativeBuild {
            cmake { arguments += listOf("-DLIBRIA_WERROR=ON") }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("../../native/libria_jni/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures { compose = true }

    // O MediaPipe lê o modelo direto do APK.
    androidResources { noCompress += "task" }
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

dependencies {
    implementation(project(":libria-bridge"))

    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")

    val camerax = "1.4.1"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // Mesma versão do MediaPipe Python fixada em requirements.txt (0.10.14).
    implementation("com.google.mediapipe:tasks-vision:0.10.14")
}

// O modelo do MediaPipe (~7,8 MB) não é versionado: é baixado no primeiro build.
val handLandmarkerUrl =
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
val handLandmarkerFile = layout.projectDirectory.file("src/main/assets/hand_landmarker.task").asFile

val downloadHandLandmarker by tasks.registering {
    description = "Baixa o modelo hand_landmarker.task do MediaPipe para os assets"
    outputs.file(handLandmarkerFile)
    onlyIf { !handLandmarkerFile.exists() }
    doLast {
        handLandmarkerFile.parentFile.mkdirs()
        URI(handLandmarkerUrl).toURL().openStream().use { input ->
            handLandmarkerFile.outputStream().use { output -> input.copyTo(output) }
        }
    }
}

tasks.named("preBuild") { dependsOn(downloadHandLandmarker) }
