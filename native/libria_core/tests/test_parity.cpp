// Paridade com o pipeline Python, a partir de native/fixtures/parity_fixtures.json
// (gerado por scripts/export_parity_fixtures.py).
//
// A comparação é EXATA (bit a bit): o núcleo reproduz as operações do NumPy em
// float32/float64, e uma diferença de arredondamento perto de um limiar mudaria
// uma decisão. Se algum dia for preciso tolerância, isso deve ser uma decisão
// explícita, não um afrouxamento silencioso.

#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "libria/features.h"
#include "libria/hybrid.h"
#include "libria/motion_detector.h"
#include "libria/quantization.h"
#include "libria/temporal_buffer.h"
#include "libria/temporal_pipeline.h"
#include "mini_json.h"
#include "test_framework.h"

namespace {

const mini_json::Value& fixtures() {
    static const mini_json::Value value = [] {
        const std::string& path = testing::state().fixture_path;
        if (path.empty()) throw std::runtime_error("informe o caminho das fixtures como argumento");
        mini_json::Value parsed = mini_json::parse_file(path);
        if (parsed["format_version"].num() != 1) throw std::runtime_error("format_version não suportado");
        return parsed;
    }();
    return value;
}

// Converte um número da fixture (float32 gravado como double) para float,
// conferindo que a conversão é exata.
float as_float(const mini_json::Value& value) {
    const double number = value.num();
    const float result = static_cast<float>(number);
    if (static_cast<double>(result) != number) {
        throw std::runtime_error("valor da fixture não é float32 exato");
    }
    return result;
}

std::vector<float> as_floats(const mini_json::Value& array) {
    std::vector<float> out;
    out.reserve(array.size());
    for (const auto& item : array.array) out.push_back(as_float(item));
    return out;
}

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

}  // namespace

TEST_CASE(features_wrist_relative) {
    int index = 0;
    for (const auto& item : fixtures()["features"].array) {
        std::vector<float> landmarks;
        for (const auto& point : item["landmarks"].array) {
            for (const auto& coordinate : point.array) landmarks.push_back(as_float(coordinate));
        }
        EXPECT(landmarks.size() == libria::kFeatureDimension);
        const std::vector<float> expected = as_floats(item["expected"]);
        float actual[libria::kFeatureDimension];
        libria::wrist_relative_features(landmarks.data(), actual);
        for (std::size_t k = 0; k < libria::kFeatureDimension; ++k) {
            EXPECT(same_bits(actual[k], expected[k]),
                   "caso " << index << " k=" << k << ": " << actual[k] << " != " << expected[k]);
        }
        ++index;
    }
}

TEST_CASE(features_in_place) {
    // `landmarks` e `out` podem ser o mesmo buffer.
    const auto& item = fixtures()["features"][0];
    std::vector<float> buffer;
    for (const auto& point : item["landmarks"].array) {
        for (const auto& coordinate : point.array) buffer.push_back(as_float(coordinate));
    }
    libria::wrist_relative_features(buffer.data(), buffer.data());
    const std::vector<float> expected = as_floats(item["expected"]);
    for (std::size_t k = 0; k < libria::kFeatureDimension; ++k) {
        EXPECT(same_bits(buffer[k], expected[k]), "k=" << k);
    }
}

TEST_CASE(resample_indices_match_numpy) {
    std::vector<std::size_t> indices;
    for (const auto& item : fixtures()["resample"].array) {
        const auto frames = static_cast<std::size_t>(item["frames"].num());
        const auto target = static_cast<std::size_t>(item["target"].num());
        libria::resample_indices(frames, target, indices);
        const auto& expected = item["indices"];
        EXPECT(indices.size() == expected.size(), "frames=" << frames << " target=" << target);
        for (std::size_t i = 0; i < indices.size() && i < expected.size(); ++i) {
            EXPECT(indices[i] == static_cast<std::size_t>(expected[i].num()),
                   "frames=" << frames << " target=" << target << " i=" << i << ": " << indices[i]
                             << " != " << expected[i].num());
        }
    }
}

TEST_CASE(motion_detector_energy) {
    int index = 0;
    for (const auto& item : fixtures()["motion"].array) {
        libria::MotionDetector detector(item["smoothing"].num());
        const auto& frames = item["frames"].array;
        for (std::size_t f = 0; f < frames.size(); ++f) {
            double energy;
            if (frames[f].is_null()) {
                energy = detector.update(nullptr, 0);
            } else {
                const std::vector<float> features = as_floats(frames[f]);
                energy = detector.update(features.data(), features.size());
            }
            const double expected = item["energy"][f].num();
            EXPECT(energy == expected, "caso " << index << " quadro " << f << ": " << energy << " != " << expected);
            EXPECT(detector.has_reading() == item["has_reading"][f].flag(), "caso " << index << " quadro " << f);
        }
        ++index;
    }
}

TEST_CASE(hybrid_decision) {
    auto read = [](const mini_json::Value& value, const char* source) -> std::optional<libria::EmbeddedPrediction> {
        if (value.is_null()) return std::nullopt;
        return libria::EmbeddedPrediction{value["token"].str(), value["confidence"].num(), source};
    };
    int index = 0;
    for (const auto& item : fixtures()["hybrid_decision"].array) {
        std::vector<std::string> priority;
        for (const auto& label : item["temporal_priority_classes"].array) priority.push_back(label.str());
        const auto chosen = libria::choose_embedded_prediction(
            read(item["static"], "static"), read(item["temporal"], "temporal"),
            item["static_threshold"].num(), item["temporal_threshold"].num(), priority);
        const auto& expected = item["expected"];
        EXPECT(chosen.has_value() == !expected.is_null(), "caso " << index);
        if (chosen.has_value() && !expected.is_null()) {
            EXPECT(chosen->token == expected["token"].str(), "caso " << index);
            EXPECT(chosen->confidence == expected["confidence"].num(), "caso " << index);
            EXPECT(chosen->source == expected["source"].str(), "caso " << index);
        }
        ++index;
    }
}

TEST_CASE(quantization_int8) {
    int index = 0;
    for (const auto& item : fixtures()["quantization"].array) {
        const std::vector<float> input = as_floats(item["input"]);
        const libria::QuantizationParams in_params{static_cast<float>(item["input_scale"].num()),
                                                   static_cast<std::int32_t>(item["input_zero_point"].num())};
        std::vector<std::int8_t> quantized(input.size());
        libria::quantize_int8(input.data(), input.size(), in_params, quantized.data());
        const auto& expected_q = item["quantized"];
        EXPECT(expected_q.size() == quantized.size(), "caso " << index);
        for (std::size_t i = 0; i < quantized.size() && i < expected_q.size(); ++i) {
            EXPECT(quantized[i] == static_cast<std::int8_t>(expected_q[i].num()),
                   "caso " << index << " i=" << i << ": " << int(quantized[i]) << " != " << expected_q[i].num());
        }

        std::vector<std::int8_t> raw;
        for (const auto& v : item["raw_output"].array) raw.push_back(static_cast<std::int8_t>(v.num()));
        const libria::QuantizationParams out_params{static_cast<float>(item["output_scale"].num()),
                                                    static_cast<std::int32_t>(item["output_zero_point"].num())};
        std::vector<float> probabilities(raw.size());
        libria::dequantize_normalize_int8(raw.data(), raw.size(), out_params, probabilities.data());
        const std::vector<float> expected_p = as_floats(item["probabilities"]);
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            EXPECT(same_bits(probabilities[i], expected_p[i]),
                   "caso " << index << " i=" << i << ": " << probabilities[i] << " != " << expected_p[i]);
        }
        ++index;
    }
}

// --------------------------------------------------------------------------
// Pipeline temporal
// --------------------------------------------------------------------------

namespace {

const std::set<std::string> kKnownConfigKeys = {
    "duplicate_window_seconds", "emit_partial_tokens",  "max_absent_frames",
    "max_duration_seconds",     "min_duration_seconds", "min_end_frames",
    "min_segment_frames",       "min_start_frames",     "motion_end_threshold",
    "motion_smoothing",         "motion_start_threshold", "partial_interval_frames",
    "smoothing_window",         "static_confidence_threshold", "static_cooldown_seconds",
    "static_interval_frames",   "temporal_confidence_threshold",
};

libria::PipelineConfig read_config(const mini_json::Value& c) {
    libria::PipelineConfig config;
    config.duplicate_window_seconds = c["duplicate_window_seconds"].num();
    config.emit_partial_tokens = c["emit_partial_tokens"].flag();
    config.max_absent_frames = static_cast<int>(c["max_absent_frames"].num());
    config.max_duration_seconds = c["max_duration_seconds"].num();
    config.min_duration_seconds = c["min_duration_seconds"].num();
    config.min_end_frames = static_cast<int>(c["min_end_frames"].num());
    config.min_segment_frames = static_cast<int>(c["min_segment_frames"].num());
    config.min_start_frames = static_cast<int>(c["min_start_frames"].num());
    config.motion_end_threshold = c["motion_end_threshold"].num();
    config.motion_smoothing = c["motion_smoothing"].num();
    config.motion_start_threshold = c["motion_start_threshold"].num();
    config.partial_interval_frames = static_cast<int>(c["partial_interval_frames"].num());
    config.smoothing_window = static_cast<int>(c["smoothing_window"].num());
    config.static_confidence_threshold = c["static_confidence_threshold"].num();
    config.static_cooldown_seconds = c["static_cooldown_seconds"].num();
    config.static_interval_frames = static_cast<int>(c["static_interval_frames"].num());
    config.temporal_confidence_threshold = c["temporal_confidence_threshold"].num();
    return config;
}

}  // namespace

TEST_CASE(pipeline_config_keys_are_known) {
    // Uma chave nova em TEMPORAL_PIPELINE_CONFIG precisa ser portada para o
    // núcleo; este teste impede que ela seja ignorada em silêncio.
    for (const auto& scenario : fixtures()["pipeline"].array) {
        std::set<std::string> keys;
        for (const auto& entry : scenario["config"].object) keys.insert(entry.first);
        for (const auto& key : keys) {
            EXPECT(kKnownConfigKeys.count(key) == 1, "chave nova no Python: " << key);
        }
        for (const auto& key : kKnownConfigKeys) {
            EXPECT(keys.count(key) == 1, "chave sumiu do Python: " << key);
        }
    }
}

TEST_CASE(pipeline_default_config_matches_python_settings) {
    // O cenário static_hold usa a configuração do Python sem overrides.
    for (const auto& scenario : fixtures()["pipeline"].array) {
        if (scenario["name"].str() != "static_hold") continue;
        const libria::PipelineConfig from_python = read_config(scenario["config"]);
        const libria::PipelineConfig defaults;
#define SAME(field) EXPECT(defaults.field == from_python.field, #field " difere de config/settings.py")
        SAME(duplicate_window_seconds); SAME(emit_partial_tokens); SAME(max_absent_frames);
        SAME(max_duration_seconds); SAME(min_duration_seconds); SAME(min_end_frames);
        SAME(min_segment_frames); SAME(min_start_frames); SAME(motion_end_threshold);
        SAME(motion_smoothing); SAME(motion_start_threshold); SAME(partial_interval_frames);
        SAME(smoothing_window); SAME(static_confidence_threshold); SAME(static_cooldown_seconds);
        SAME(static_interval_frames); SAME(temporal_confidence_threshold);
#undef SAME
        return;
    }
    EXPECT(false, "cenário static_hold não encontrado");
}

TEST_CASE(pipeline_scenarios) {
    for (const auto& scenario : fixtures()["pipeline"].array) {
        const std::string name = scenario["name"].str();
        const auto dim = static_cast<std::size_t>(scenario["feature_dim"].num());
        const auto sequence_length = static_cast<std::size_t>(scenario["sequence_length"].num());

        // Quadros do cenário (vazio = mão ausente).
        std::vector<std::vector<float>> frames;
        std::vector<double> timestamps;
        for (const auto& frame : scenario["frames"].array) {
            timestamps.push_back(frame["t"].num());
            frames.push_back(frame["f"].is_null() ? std::vector<float>{} : as_floats(frame["f"]));
        }

        std::map<int, std::string> label_map;
        for (const auto& entry : scenario["label_map"].object) {
            label_map[std::stoi(entry.first)] = entry.second.str();
        }

        // Preditores roteirizados: conferem a entrada recebida contra a gravada
        // no Python e devolvem a mesma resposta.
        const auto& temporal_calls = scenario["temporal_calls"].array;
        std::size_t temporal_index = 0;
        auto temporal = [&](const float* sequence, std::size_t count, std::size_t feature_dim,
                            std::vector<double>& probabilities) {
            EXPECT(temporal_index < temporal_calls.size(), name << ": chamada temporal a mais");
            if (temporal_index >= temporal_calls.size()) { probabilities.assign(1, 0.0); return; }
            const auto& call = temporal_calls[temporal_index];
            const auto& expected_frames = call["input_frames"];
            EXPECT(count == expected_frames.size() && feature_dim == dim, name << ": forma da entrada temporal");
            for (std::size_t i = 0; i < count && i < expected_frames.size(); ++i) {
                const auto& source = frames[static_cast<std::size_t>(expected_frames[i].num())];
                EXPECT(std::memcmp(sequence + i * feature_dim, source.data(), feature_dim * sizeof(float)) == 0,
                       name << ": chamada temporal " << temporal_index << ", passo " << i
                            << " difere do quadro " << expected_frames[i].num());
            }
            probabilities.clear();
            for (const auto& p : call["output"].array) probabilities.push_back(p.num());
            ++temporal_index;
        };

        const auto& static_calls = scenario["static_calls"].array;
        std::size_t static_index = 0;
        libria::StaticPredictor static_predictor;
        if (scenario["has_static_predictor"].flag()) {
            static_predictor = [&](const float* features, std::size_t feature_dim) {
                EXPECT(static_index < static_calls.size(), name << ": chamada estática a mais");
                if (static_index >= static_calls.size()) return libria::StaticPrediction{"", 0.0};
                const auto& call = static_calls[static_index];
                const auto& source = frames[static_cast<std::size_t>(call["input_frames"].num())];
                EXPECT(feature_dim == dim && std::memcmp(features, source.data(), dim * sizeof(float)) == 0,
                       name << ": chamada estática " << static_index << " recebeu o quadro errado");
                ++static_index;
                return libria::StaticPrediction{call["output"]["label"].str(), call["output"]["confidence"].num()};
            };
        }

        libria::TemporalPipeline pipeline(temporal, label_map, sequence_length, dim, static_predictor,
                                          read_config(scenario["config"]));

        const auto& expected = scenario["expected"].array;
        for (std::size_t f = 0; f < frames.size(); ++f) {
            const bool present = !frames[f].empty();
            const auto token = pipeline.process_frame(present ? frames[f].data() : nullptr, timestamps[f], present);
            const auto& exp = expected[f];

            EXPECT(pipeline.last_energy() == exp["energy"].num(),
                   name << " quadro " << f << ": energia " << pipeline.last_energy() << " != " << exp["energy"].num());
            EXPECT(pipeline.is_signing() == exp["signing"].flag(), name << " quadro " << f << ": is_signing");

            const auto& exp_token = exp["token"];
            EXPECT(token.has_value() == !exp_token.is_null(),
                   name << " quadro " << f << ": token " << (token ? "emitido" : "ausente") << " no C++, "
                        << (exp_token.is_null() ? "ausente" : "emitido") << " no Python");
            if (token.has_value() && !exp_token.is_null()) {
                EXPECT(token->label == exp_token["label"].str(),
                       name << " quadro " << f << ": rótulo " << token->label << " != " << exp_token["label"].str());
                EXPECT(token->confidence == exp_token["confidence"].num(), name << " quadro " << f << ": confiança");
                EXPECT(token->start_time == exp_token["start_time"].num(), name << " quadro " << f << ": início");
                EXPECT(token->end_time == exp_token["end_time"].num(), name << " quadro " << f << ": fim");
                EXPECT(std::string(libria::to_string(token->source)) == exp_token["source"].str(),
                       name << " quadro " << f << ": origem");
                EXPECT(std::string(libria::to_string(token->state)) == exp_token["state"].str(),
                       name << " quadro " << f << ": estado " << libria::to_string(token->state));
                EXPECT(token->frame_count == static_cast<int>(exp_token["frame_count"].num()),
                       name << " quadro " << f << ": frame_count " << token->frame_count);
            }
        }
        EXPECT(temporal_index == temporal_calls.size(),
               name << ": " << temporal_index << " chamadas temporais, Python fez " << temporal_calls.size());
        EXPECT(static_index == static_calls.size(),
               name << ": " << static_index << " chamadas estáticas, Python fez " << static_calls.size());
    }
}
