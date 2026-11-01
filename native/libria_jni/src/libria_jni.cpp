// Ponte JNI: io.github.gabryellima.libria.bridge.LibriaNative <-> libria::core.
//
// Contrato com o Kotlin:
// - Os métodos são registrados em JNI_OnLoad (RegisterNatives). Se uma
//   assinatura divergir, o carregamento falha com UnsatisfiedLinkError em vez
//   de estourar na primeira chamada.
// - Os preditores Kotlin são chamados de volta durante processFrame, na mesma
//   thread. Uma exceção lançada por eles atravessa o C++ e chega intacta a
//   quem chamou processFrame.
// - Cada pipeline é de uma thread só, como o TemporalPipeline do C++.

#include <jni.h>

#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "libria/features.h"
#include "libria/temporal_pipeline.h"

namespace {

constexpr const char* kNativeClass = "io/github/gabryellima/libria/bridge/LibriaNative";
constexpr const char* kSignTokenClass = "io/github/gabryellima/libria/bridge/SignToken";
constexpr const char* kTemporalPredictorClass = "io/github/gabryellima/libria/bridge/TemporalPredictor";
constexpr const char* kStaticPredictorClass = "io/github/gabryellima/libria/bridge/StaticPredictor";
constexpr const char* kStaticPredictionClass = "io/github/gabryellima/libria/bridge/StaticPrediction";

// Layout dos arrays de configuração (o mesmo de PipelineConfig.kt).
enum DoubleField {
    kMotionSmoothing, kMotionStartThreshold, kMotionEndThreshold, kMinDurationSeconds,
    kMaxDurationSeconds, kDuplicateWindowSeconds, kTemporalConfidenceThreshold,
    kStaticConfidenceThreshold, kStaticCooldownSeconds, kDoubleFieldCount
};
enum IntField {
    kMinStartFrames, kMinEndFrames, kMinSegmentFrames, kMaxAbsentFrames, kSmoothingWindow,
    kPartialIntervalFrames, kStaticIntervalFrames, kIntFieldCount
};

struct Cache {
    jclass sign_token = nullptr;
    jmethodID sign_token_from_native = nullptr;
    jmethodID temporal_predict = nullptr;
    jmethodID static_predict = nullptr;
    jfieldID static_label = nullptr;
    jfieldID static_confidence = nullptr;
    jclass illegal_argument = nullptr;
    jclass illegal_state = nullptr;
};
Cache g_cache;

// Uma exceção Java já está pendente: só desempilhar o C++ até a borda JNI.
struct JavaExceptionPending {};

void throw_java(JNIEnv* env, jclass type, const std::string& message) {
    if (!env->ExceptionCheck()) env->ThrowNew(type, message.c_str());
}

struct NativePipeline {
    JNIEnv* env = nullptr;           // válido só durante a chamada nativa corrente
    jobject temporal = nullptr;      // referência global
    jobject static_predictor = nullptr;  // referência global ou nulo
    jfloatArray sequence_buffer = nullptr;  // referência global, reutilizada
    jfloatArray features_buffer = nullptr;  // referência global, reutilizada
    std::vector<float> frame;        // cópia do quadro recebido, reutilizada
    std::unique_ptr<libria::TemporalPipeline> pipeline;

    void release(JNIEnv* e) {
        if (temporal) e->DeleteGlobalRef(temporal);
        if (static_predictor) e->DeleteGlobalRef(static_predictor);
        if (sequence_buffer) e->DeleteGlobalRef(sequence_buffer);
        if (features_buffer) e->DeleteGlobalRef(features_buffer);
        temporal = static_predictor = nullptr;
        sequence_buffer = features_buffer = nullptr;
    }
};

NativePipeline* from_handle(JNIEnv* env, jlong handle) {
    auto* native = reinterpret_cast<NativePipeline*>(static_cast<std::intptr_t>(handle));
    if (native == nullptr) throw_java(env, g_cache.illegal_state, "pipeline já foi fechado");
    return native;
}

jfloatArray new_global_float_array(JNIEnv* env, std::size_t size) {
    jfloatArray local = env->NewFloatArray(static_cast<jsize>(size));
    if (local == nullptr) throw JavaExceptionPending{};  // OutOfMemoryError pendente
    auto global = static_cast<jfloatArray>(env->NewGlobalRef(local));
    env->DeleteLocalRef(local);
    return global;
}

libria::PipelineConfig read_config(JNIEnv* env, jdoubleArray doubles, jintArray ints, jboolean emit_partials,
                                   jstring unknown_label) {
    if (env->GetArrayLength(doubles) != kDoubleFieldCount || env->GetArrayLength(ints) != kIntFieldCount) {
        throw std::invalid_argument("layout dos arrays de configuração não confere com o núcleo nativo");
    }
    double d[kDoubleFieldCount];
    jint i[kIntFieldCount];
    env->GetDoubleArrayRegion(doubles, 0, kDoubleFieldCount, d);
    env->GetIntArrayRegion(ints, 0, kIntFieldCount, i);

    libria::PipelineConfig config;
    config.motion_smoothing = d[kMotionSmoothing];
    config.motion_start_threshold = d[kMotionStartThreshold];
    config.motion_end_threshold = d[kMotionEndThreshold];
    config.min_duration_seconds = d[kMinDurationSeconds];
    config.max_duration_seconds = d[kMaxDurationSeconds];
    config.duplicate_window_seconds = d[kDuplicateWindowSeconds];
    config.temporal_confidence_threshold = d[kTemporalConfidenceThreshold];
    config.static_confidence_threshold = d[kStaticConfidenceThreshold];
    config.static_cooldown_seconds = d[kStaticCooldownSeconds];
    config.min_start_frames = i[kMinStartFrames];
    config.min_end_frames = i[kMinEndFrames];
    config.min_segment_frames = i[kMinSegmentFrames];
    config.max_absent_frames = i[kMaxAbsentFrames];
    config.smoothing_window = i[kSmoothingWindow];
    config.partial_interval_frames = i[kPartialIntervalFrames];
    config.static_interval_frames = i[kStaticIntervalFrames];
    config.emit_partial_tokens = emit_partials == JNI_TRUE;

    const char* label = env->GetStringUTFChars(unknown_label, nullptr);
    if (label == nullptr) throw JavaExceptionPending{};
    config.unknown_label = label;
    env->ReleaseStringUTFChars(unknown_label, label);
    return config;
}

std::map<int, std::string> read_label_map(JNIEnv* env, jintArray indices, jobjectArray names) {
    const jsize count = env->GetArrayLength(indices);
    if (env->GetArrayLength(names) != count) {
        throw std::invalid_argument("labelMap: índices e nomes com tamanhos diferentes");
    }
    std::vector<jint> keys(static_cast<std::size_t>(count));
    env->GetIntArrayRegion(indices, 0, count, keys.data());
    std::map<int, std::string> result;
    for (jsize k = 0; k < count; ++k) {
        auto name = static_cast<jstring>(env->GetObjectArrayElement(names, k));
        const char* chars = env->GetStringUTFChars(name, nullptr);
        if (chars == nullptr) throw JavaExceptionPending{};
        result[keys[static_cast<std::size_t>(k)]] = chars;
        env->ReleaseStringUTFChars(name, chars);
        env->DeleteLocalRef(name);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Métodos nativos
// ---------------------------------------------------------------------------

jlong native_create(JNIEnv* env, jclass, jdoubleArray doubles, jintArray ints, jboolean emit_partials,
                    jstring unknown_label, jintArray label_indices, jobjectArray label_names,
                    jint sequence_length, jint feature_dim, jobject temporal, jobject static_predictor) {
    auto native = std::make_unique<NativePipeline>();
    try {
        if (temporal == nullptr) throw std::invalid_argument("temporalPredictor é obrigatório");
        if (sequence_length <= 0 || feature_dim <= 0) {
            throw std::invalid_argument("sequenceLength e featureDim devem ser maiores que zero");
        }
        const libria::PipelineConfig config = read_config(env, doubles, ints, emit_partials, unknown_label);
        std::map<int, std::string> labels = read_label_map(env, label_indices, label_names);

        const auto seq = static_cast<std::size_t>(sequence_length);
        const auto dim = static_cast<std::size_t>(feature_dim);
        native->temporal = env->NewGlobalRef(temporal);
        if (static_predictor != nullptr) native->static_predictor = env->NewGlobalRef(static_predictor);
        native->sequence_buffer = new_global_float_array(env, seq * dim);
        native->features_buffer = new_global_float_array(env, dim);
        native->frame.resize(dim);

        NativePipeline* h = native.get();
        libria::TemporalPredictor temporal_fn = [h](const float* sequence, std::size_t frames, std::size_t width,
                                                    std::vector<double>& probabilities) {
            JNIEnv* e = h->env;
            e->SetFloatArrayRegion(h->sequence_buffer, 0, static_cast<jsize>(frames * width), sequence);
            auto result = static_cast<jdoubleArray>(e->CallObjectMethod(
                h->temporal, g_cache.temporal_predict, h->sequence_buffer, static_cast<jint>(frames),
                static_cast<jint>(width)));
            if (e->ExceptionCheck()) throw JavaExceptionPending{};
            if (result == nullptr) {
                throw_java(e, g_cache.illegal_state, "TemporalPredictor.predict devolveu null");
                throw JavaExceptionPending{};
            }
            const jsize n = e->GetArrayLength(result);
            probabilities.resize(static_cast<std::size_t>(n));
            e->GetDoubleArrayRegion(result, 0, n, probabilities.data());
            e->DeleteLocalRef(result);
        };

        libria::StaticPredictor static_fn;
        if (native->static_predictor != nullptr) {
            static_fn = [h](const float* features, std::size_t width) {
                JNIEnv* e = h->env;
                e->SetFloatArrayRegion(h->features_buffer, 0, static_cast<jsize>(width), features);
                jobject result = e->CallObjectMethod(h->static_predictor, g_cache.static_predict, h->features_buffer);
                if (e->ExceptionCheck()) throw JavaExceptionPending{};
                if (result == nullptr) {
                    throw_java(e, g_cache.illegal_state, "StaticPredictor.predict devolveu null");
                    throw JavaExceptionPending{};
                }
                auto label = static_cast<jstring>(e->GetObjectField(result, g_cache.static_label));
                const double confidence = e->GetDoubleField(result, g_cache.static_confidence);
                libria::StaticPrediction prediction;
                prediction.confidence = confidence;
                if (label != nullptr) {
                    const char* chars = e->GetStringUTFChars(label, nullptr);
                    if (chars == nullptr) throw JavaExceptionPending{};
                    prediction.label = chars;
                    e->ReleaseStringUTFChars(label, chars);
                    e->DeleteLocalRef(label);
                }
                e->DeleteLocalRef(result);
                return prediction;
            };
        }

        native->pipeline = std::make_unique<libria::TemporalPipeline>(
            std::move(temporal_fn), std::move(labels), seq, dim, std::move(static_fn), config);
        return static_cast<jlong>(reinterpret_cast<std::intptr_t>(native.release()));
    } catch (const JavaExceptionPending&) {
    } catch (const std::invalid_argument& error) {
        throw_java(env, g_cache.illegal_argument, error.what());
    } catch (const std::bad_alloc&) {
        throw_java(env, g_cache.illegal_state, "memória insuficiente no núcleo nativo");
    } catch (const std::exception& error) {
        throw_java(env, g_cache.illegal_state, error.what());
    }
    native->release(env);
    return 0;
}

void native_destroy(JNIEnv* env, jclass, jlong handle) {
    auto* native = reinterpret_cast<NativePipeline*>(static_cast<std::intptr_t>(handle));
    if (native == nullptr) return;
    native->release(env);
    delete native;
}

jobject native_process_frame(JNIEnv* env, jclass, jlong handle, jfloatArray features, jdouble timestamp,
                             jboolean hand_present) {
    NativePipeline* native = from_handle(env, handle);
    if (native == nullptr) return nullptr;
    try {
        const float* frame = nullptr;
        if (features != nullptr) {
            if (static_cast<std::size_t>(env->GetArrayLength(features)) != native->frame.size()) {
                throw std::invalid_argument("features com tamanho diferente de featureDim");
            }
            env->GetFloatArrayRegion(features, 0, static_cast<jsize>(native->frame.size()), native->frame.data());
            frame = native->frame.data();
        }
        native->env = env;
        const std::optional<libria::SignToken> token =
            native->pipeline->process_frame(frame, timestamp, hand_present == JNI_TRUE);
        native->env = nullptr;
        if (!token.has_value()) return nullptr;

        jstring label = env->NewStringUTF(token->label.c_str());
        if (label == nullptr) return nullptr;
        jobject result = env->CallStaticObjectMethod(
            g_cache.sign_token, g_cache.sign_token_from_native, label, token->confidence, token->start_time,
            token->end_time, static_cast<jint>(token->source), static_cast<jint>(token->state),
            static_cast<jint>(token->frame_count));
        env->DeleteLocalRef(label);
        return result;
    } catch (const JavaExceptionPending&) {
    } catch (const std::invalid_argument& error) {
        throw_java(env, g_cache.illegal_argument, error.what());
    } catch (const std::exception& error) {
        throw_java(env, g_cache.illegal_state, error.what());
    }
    native->env = nullptr;
    return nullptr;
}

void native_reset(JNIEnv* env, jclass, jlong handle) {
    if (NativePipeline* native = from_handle(env, handle)) native->pipeline->reset();
}

jboolean native_is_signing(JNIEnv* env, jclass, jlong handle) {
    NativePipeline* native = from_handle(env, handle);
    return native != nullptr && native->pipeline->is_signing() ? JNI_TRUE : JNI_FALSE;
}

jdouble native_last_energy(JNIEnv* env, jclass, jlong handle) {
    NativePipeline* native = from_handle(env, handle);
    return native != nullptr ? native->pipeline->last_energy() : 0.0;
}

void native_wrist_relative(JNIEnv* env, jclass, jfloatArray landmarks, jfloatArray out) {
    constexpr auto n = static_cast<jsize>(libria::kFeatureDimension);
    if (env->GetArrayLength(landmarks) != n || env->GetArrayLength(out) != n) {
        throw_java(env, g_cache.illegal_argument, "landmarks e out devem ter 63 floats (21 x 3)");
        return;
    }
    float buffer[libria::kFeatureDimension];
    env->GetFloatArrayRegion(landmarks, 0, n, buffer);
    libria::wrist_relative_features(buffer, buffer);
    env->SetFloatArrayRegion(out, 0, n, buffer);
}

jdoubleArray native_default_doubles(JNIEnv* env, jclass) {
    const libria::PipelineConfig c;
    double d[kDoubleFieldCount];
    d[kMotionSmoothing] = c.motion_smoothing;
    d[kMotionStartThreshold] = c.motion_start_threshold;
    d[kMotionEndThreshold] = c.motion_end_threshold;
    d[kMinDurationSeconds] = c.min_duration_seconds;
    d[kMaxDurationSeconds] = c.max_duration_seconds;
    d[kDuplicateWindowSeconds] = c.duplicate_window_seconds;
    d[kTemporalConfidenceThreshold] = c.temporal_confidence_threshold;
    d[kStaticConfidenceThreshold] = c.static_confidence_threshold;
    d[kStaticCooldownSeconds] = c.static_cooldown_seconds;
    jdoubleArray result = env->NewDoubleArray(kDoubleFieldCount);
    if (result != nullptr) env->SetDoubleArrayRegion(result, 0, kDoubleFieldCount, d);
    return result;
}

jintArray native_default_ints(JNIEnv* env, jclass) {
    const libria::PipelineConfig c;
    jint i[kIntFieldCount];
    i[kMinStartFrames] = c.min_start_frames;
    i[kMinEndFrames] = c.min_end_frames;
    i[kMinSegmentFrames] = c.min_segment_frames;
    i[kMaxAbsentFrames] = c.max_absent_frames;
    i[kSmoothingWindow] = c.smoothing_window;
    i[kPartialIntervalFrames] = c.partial_interval_frames;
    i[kStaticIntervalFrames] = c.static_interval_frames;
    jintArray result = env->NewIntArray(kIntFieldCount);
    if (result != nullptr) env->SetIntArrayRegion(result, 0, kIntFieldCount, i);
    return result;
}

jboolean native_default_emit_partials(JNIEnv*, jclass) {
    return libria::PipelineConfig{}.emit_partial_tokens ? JNI_TRUE : JNI_FALSE;
}

jstring native_default_unknown_label(JNIEnv* env, jclass) {
    return env->NewStringUTF(libria::PipelineConfig{}.unknown_label.c_str());
}

jclass global_class(JNIEnv* env, const char* name) {
    jclass local = env->FindClass(name);
    if (local == nullptr) return nullptr;
    auto global = static_cast<jclass>(env->NewGlobalRef(local));
    env->DeleteLocalRef(local);
    return global;
}

}  // namespace

extern "C" JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
    JNIEnv* env = nullptr;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) return JNI_ERR;

    g_cache.sign_token = global_class(env, kSignTokenClass);
    jclass temporal = env->FindClass(kTemporalPredictorClass);
    jclass static_predictor = env->FindClass(kStaticPredictorClass);
    jclass static_prediction = env->FindClass(kStaticPredictionClass);
    g_cache.illegal_argument = global_class(env, "java/lang/IllegalArgumentException");
    g_cache.illegal_state = global_class(env, "java/lang/IllegalStateException");
    if (!g_cache.sign_token || !temporal || !static_predictor || !static_prediction || !g_cache.illegal_argument ||
        !g_cache.illegal_state) {
        return JNI_ERR;
    }

    const std::string token_sig = std::string("(Ljava/lang/String;DDDIII)L") + kSignTokenClass + ";";
    g_cache.sign_token_from_native = env->GetStaticMethodID(g_cache.sign_token, "fromNative", token_sig.c_str());
    g_cache.temporal_predict = env->GetMethodID(temporal, "predict", "([FII)[D");
    const std::string static_sig = std::string("([F)L") + kStaticPredictionClass + ";";
    g_cache.static_predict = env->GetMethodID(static_predictor, "predict", static_sig.c_str());
    g_cache.static_label = env->GetFieldID(static_prediction, "label", "Ljava/lang/String;");
    g_cache.static_confidence = env->GetFieldID(static_prediction, "confidence", "D");
    if (!g_cache.sign_token_from_native || !g_cache.temporal_predict || !g_cache.static_predict ||
        !g_cache.static_label || !g_cache.static_confidence) {
        return JNI_ERR;
    }

    const std::string create_sig = std::string("([D[IZLjava/lang/String;[I[Ljava/lang/String;IIL") +
                                   kTemporalPredictorClass + ";L" + kStaticPredictorClass + ";)J";
    const std::string process_sig = std::string("(J[FDZ)L") + kSignTokenClass + ";";
    const JNINativeMethod methods[] = {
        {const_cast<char*>("nativeCreate"), const_cast<char*>(create_sig.c_str()), reinterpret_cast<void*>(native_create)},
        {const_cast<char*>("nativeDestroy"), const_cast<char*>("(J)V"), reinterpret_cast<void*>(native_destroy)},
        {const_cast<char*>("nativeProcessFrame"), const_cast<char*>(process_sig.c_str()), reinterpret_cast<void*>(native_process_frame)},
        {const_cast<char*>("nativeReset"), const_cast<char*>("(J)V"), reinterpret_cast<void*>(native_reset)},
        {const_cast<char*>("nativeIsSigning"), const_cast<char*>("(J)Z"), reinterpret_cast<void*>(native_is_signing)},
        {const_cast<char*>("nativeLastEnergy"), const_cast<char*>("(J)D"), reinterpret_cast<void*>(native_last_energy)},
        {const_cast<char*>("nativeWristRelative"), const_cast<char*>("([F[F)V"), reinterpret_cast<void*>(native_wrist_relative)},
        {const_cast<char*>("nativeDefaultDoubles"), const_cast<char*>("()[D"), reinterpret_cast<void*>(native_default_doubles)},
        {const_cast<char*>("nativeDefaultInts"), const_cast<char*>("()[I"), reinterpret_cast<void*>(native_default_ints)},
        {const_cast<char*>("nativeDefaultEmitPartials"), const_cast<char*>("()Z"), reinterpret_cast<void*>(native_default_emit_partials)},
        {const_cast<char*>("nativeDefaultUnknownLabel"), const_cast<char*>("()Ljava/lang/String;"), reinterpret_cast<void*>(native_default_unknown_label)},
    };
    jclass native_class = env->FindClass(kNativeClass);
    if (native_class == nullptr) return JNI_ERR;
    if (env->RegisterNatives(native_class, methods, sizeof(methods) / sizeof(methods[0])) != JNI_OK) return JNI_ERR;
    env->DeleteLocalRef(native_class);
    env->DeleteLocalRef(temporal);
    env->DeleteLocalRef(static_predictor);
    env->DeleteLocalRef(static_prediction);
    return JNI_VERSION_1_6;
}
