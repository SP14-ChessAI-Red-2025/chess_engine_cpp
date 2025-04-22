#include "chess_cpp/nnue_evaluator.hpp"
#include "chess_cpp/chess_rules.hpp"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <array>
#include <memory>
#include <iostream>
#include <filesystem>

// Define a simple logging macro
#define LOG_INFO(msg) std::cout << "[INFO] " << msg << std::endl
#define LOG_WARN(msg) std::cerr << "[WARN] " << msg << std::endl

namespace chess::ai {

    // --- Constants matching Python feature extraction ---
    constexpr int TOTAL_FEATURES = 768;
    constexpr int FEATURES_PER_COLOR = TOTAL_FEATURES / 2;

    inline int piece_type_to_index(chess::piece_type pt) {
        int type_val = static_cast<int>(pt);
        return (type_val >= 1 && type_val <= 6) ? type_val - 1 : -1;
    }

    // PImpl Idiom for ONNX Runtime details
    struct NNUEEvaluator::OrtImpl {
        Ort::Env env;
        Ort::SessionOptions session_options;
        std::unique_ptr<Ort::Session> session;

        OrtImpl(const std::string& model_path)
            : env(ORT_LOGGING_LEVEL_WARNING, "chess_nnue_evaluator") {
            session_options.SetIntraOpNumThreads(0);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            try {
                // Attempt CUDA Execution Provider
                OrtCUDAProviderOptions cuda_options{};
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                initialize_session(model_path);
                LOG_INFO("ONNX Runtime session initialized with CUDA Execution Provider.");
            } catch (const Ort::Exception& e) {
                LOG_WARN("CUDA Execution Provider unavailable. Falling back to CPU.");

                // Create a new session options object for CPU fallback
                Ort::SessionOptions cpu_session_options;
                cpu_session_options.SetIntraOpNumThreads(0);
                cpu_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

                // Initialize the session with the new options
                initialize_session_with_options(model_path, cpu_session_options);
                LOG_INFO("ONNX Runtime session initialized with CPU Execution Provider.");
            }
        }

        void initialize_session(const std::string& model_path) {
            std::filesystem::path model_path_fs(model_path);
            #ifdef _WIN32
                std::wstring w_model_path = model_path_fs.wstring();
                session = std::make_unique<Ort::Session>(env, w_model_path.c_str(), session_options);
            #else
                session = std::make_unique<Ort::Session>(env, model_path_fs.c_str(), session_options);
            #endif
        }

        void initialize_session_with_options(const std::string& model_path, Ort::SessionOptions& options) {
            std::filesystem::path model_path_fs(model_path);
            #ifdef _WIN32
                std::wstring w_model_path = model_path_fs.wstring();
                session = std::make_unique<Ort::Session>(env, w_model_path.c_str(), options);
            #else
                session = std::make_unique<Ort::Session>(env, model_path_fs.c_str(), options);
            #endif
        }
    };

    NNUEEvaluator::NNUEEvaluator(const std::string& model_path)
        : input_node_names_{"input_indices", "input_offsets"},
          output_node_names_{"output"} {
        ort_impl_ = std::make_unique<OrtImpl>(model_path);
        memory_info_cpu_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    }

    NNUEEvaluator::~NNUEEvaluator() = default;

    std::vector<int64_t> NNUEEvaluator::boardToFeaturesIndices(const chess::board_state& board) const {
        std::vector<int64_t> indices;
        indices.reserve(32);

        bool white_king_found = false, black_king_found = false;

        for (int square = 0; square < 64; ++square) {
            const auto& piece = board.pieces[square / 8][square % 8];
            if (piece.type == piece_type::none) continue;

            int p_idx = piece_type_to_index(piece.type);
            if (p_idx == -1) continue;

            int64_t feature_index = static_cast<int64_t>(p_idx) * 64 + square;
            if (piece.piece_player == player::white) {
                indices.push_back(feature_index);
                if (piece.type == piece_type::king) white_king_found = true;
            } else {
                indices.push_back(feature_index + FEATURES_PER_COLOR);
                if (piece.type == piece_type::king) black_king_found = true;
            }

            // Break early if both kings are found
            if (white_king_found && black_king_found) break;
        }

        return (white_king_found && black_king_found) ? indices : std::vector<int64_t>{};
    }

    double NNUEEvaluator::evaluate(const chess::board_state& board) {
        static thread_local std::vector<int64_t> feature_indices;
        static thread_local std::vector<int64_t> offsets_data = {0};

        feature_indices.clear();
        feature_indices = boardToFeaturesIndices(board);

        if (feature_indices.empty()) return 0.0;

        std::array<int64_t, 1> indices_shape = {static_cast<int64_t>(feature_indices.size())};
        Ort::Value indices_tensor = Ort::Value::CreateTensor<int64_t>(
            *memory_info_cpu_, feature_indices.data(), feature_indices.size(), indices_shape.data(), indices_shape.size());

        std::array<int64_t, 1> offsets_shape = {1};
        Ort::Value offsets_tensor = Ort::Value::CreateTensor<int64_t>(
            *memory_info_cpu_, offsets_data.data(), offsets_data.size(), offsets_shape.data(), offsets_shape.size());

        std::array<Ort::Value, 2> input_tensors = {std::move(indices_tensor), std::move(offsets_tensor)};
        auto output_tensors = ort_impl_->session->Run(
            Ort::RunOptions{nullptr}, input_node_names_.data(), input_tensors.data(), input_tensors.size(),
            output_node_names_.data(), output_node_names_.size());

        auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        if (type_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || type_info.GetElementCount() != 1) {
            LOG_WARN("Unexpected ONNX output tensor shape or type.");
            return 0.0;
        }

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        double score = static_cast<double>(output_data[0]);
        return (board.current_player == player::black) ? -score : score;
    }

} // namespace chess::ai
