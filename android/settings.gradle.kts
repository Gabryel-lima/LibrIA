// Projeto Android do LibrIA (plano: .github/plans/android_plan.md, fase F2).
//
// Módulos:
//   :libria-bridge  Kotlin/JVM puro: API do núcleo C++ via JNI, benchmark por
//                   etapa, FPS e mapeamento do overlay. Testado na JVM do host.
//   :app            App Android (CameraX + MediaPipe + Compose). Precisa do
//                   Android SDK e do NDK; sem eles, fica fora do build.

pluginManagement {
    repositories {
        // Só os grupos do Google vão ao repositório google(): o resto não depende dele.
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
    plugins {
        // Versões do app escolhidas sem acesso ao Google Maven (ver android/README.md).
        id("com.android.application") version "8.7.3"
        id("org.jetbrains.kotlin.android") version "2.0.21"
        id("org.jetbrains.kotlin.plugin.compose") version "2.0.21"
        id("org.jetbrains.kotlin.jvm") version "2.0.21"
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
    }
}

rootProject.name = "libria-android"
include(":libria-bridge")

fun androidSdkAvailable(): Boolean {
    if (!System.getenv("ANDROID_HOME").isNullOrBlank() || !System.getenv("ANDROID_SDK_ROOT").isNullOrBlank()) {
        return true
    }
    val localProperties = file("local.properties")
    return localProperties.isFile && localProperties.readLines().any { it.trim().startsWith("sdk.dir=") }
}

if (androidSdkAvailable()) {
    include(":app")
} else {
    logger.lifecycle("LibrIA: Android SDK não encontrado (ANDROID_HOME ou local.properties); :app fica fora do build.")
}
