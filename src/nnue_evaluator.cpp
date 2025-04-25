// src/nnue_evaluator.cpp (Includes logging macros)
#include "chess_cpp/nnue_evaluator.hpp"
#include "chess_cpp/chess_rules.hpp" // Uses board_state, piece_type, player
#include <onnxruntime_cxx_api.h>   // Main ORT header
#include <vector>
#include <string>
#include <stdexcept>
#include <array>
#include <memory>                   // For std::unique_ptr
#include <iostream>                 // For std::cout, std::cerr
#include <vector>                   // For std::vector
#include <cstdint>                  // For int64_t
#include <filesystem>               // For path handling

// Simple logging macros
#define LOG_INFO(msg) std::cout << "[INFO] " << msg << std::endl
#define LOG_WARN(msg) std::cerr << "[WARN] " << msg << std::endl
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl

namespace chess::ai {

    // --- Constants ---
    constexpr int TOTAL_FEATURES = 768;
    constexpr int FEATURES_PER_COLOR = TOTAL_FEATURES / 2;

    // Helper to get index (0-5) for piece type (1-6) or -1 for none
    inline int piece_type_to_index(chess::piece_type pt) {
        int type_val = static_cast<int>(pt);
        return (type_val >= 1 && type_val <= 6) ? type_val - 1 : -1;
    }


    // --- PImpl Definition ---
    struct NNUEEvaluator::OrtImpl {
        Ort::Env env; // ORT Environment (must outlive session)
        Ort::SessionOptions session_options; // Session options
        std::unique_ptr<Ort::Session> session; // ONNX Session

        // Constructor: Initialize environment, options, and session
        OrtImpl(const std::string& model_path)
            : env(ORT_LOGGING_LEVEL_WARNING, "chess_nnue_evaluator") // Create env first
        {
            session_options.SetIntraOpNumThreads(0); // Use default thread settings
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            bool cuda_available = false;
            // Check CUDA availability (wrap in try-catch as AppendExecutionProvider throws)
            try {
                OrtCUDAProviderOptions cuda_options{};
                // Can configure options like device_id, gpu_mem_limit here if needed
                // cuda_options.device_id = 0;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                cuda_available = true; // Assume success if no exception
                LOG_INFO("Attempting to initialize ONNX Runtime session with CUDA Execution Provider.");
            } catch (const Ort::Exception& e) {
                LOG_WARN("Failed to append CUDA Execution Provider: " + std::string(e.what()) + ". Falling back to CPU.");
                // If CUDA fails, SessionOptions might be in a state that prevents Session creation later.
                // Best practice is often to recreate SessionOptions for CPU only.
                session_options = Ort::SessionOptions(); // Reset options
                session_options.SetIntraOpNumThreads(0);
                session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            } catch (...) {
                 LOG_WARN("Unknown error appending CUDA provider. Falling back to CPU.");
                 session_options = Ort::SessionOptions(); // Reset options
            }

            // Create the session using the configured options (either CUDA or CPU)
            try {
                // Convert path for Windows if necessary
                std::filesystem::path model_path_fs(model_path);
                #ifdef _WIN32
                    std::wstring w_model_path = model_path_fs.wstring();
                    session = std::make_unique<Ort::Session>(env, w_model_path.c_str(), session_options);
                #else
                    session = std::make_unique<Ort::Session>(env, model_path_fs.c_str(), session_options);
                #endif
                LOG_INFO(std::string("ONNX Runtime session initialized successfully using ") + (cuda_available ? "CUDA." : "CPU."));
            } catch (const Ort::Exception& e) {
                LOG_ERROR("Failed to create ONNX Runtime session: " + std::string(e.what()));
                // Re-throw a standard exception to indicate failure
                throw std::runtime_error("ONNX Runtime session creation failed: " + std::string(e.what()));
            } catch (...) {
                LOG_ERROR("Unknown error creating ONNX Runtime session.");
                throw std::runtime_error("Unknown error during ONNX Runtime session creation.");
            }
        }
    };


    // --- NNUEEvaluator Class Member Function Definitions ---

    // Constructor: Create the implementation object
    NNUEEvaluator::NNUEEvaluator(const std::string& model_path)
        : input_node_names_{input_node_name_indices_, input_node_name_offsets_}, // Initialize vector from members
          output_node_names_{output_node_name_}
    {
        try {
             ort_impl_ = std::make_unique<OrtImpl>(model_path); // Create the PImpl instance
             memory_info_cpu_ = std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
        } catch (...) {
             // Constructor of OrtImpl might throw, ensure NNUEEvaluator constructor propagates it
             // No need to catch here if OrtImpl constructor handles logging and throws std::runtime_error
             throw; // Re-throw exception from OrtImpl constructor
        }
    }

    // Destructor: Default is sufficient due to unique_ptr managing OrtImpl
    NNUEEvaluator::~NNUEEvaluator() = default;

    // boardToFeaturesIndices: Convert board state to NNUE input indices
    std::vector<int64_t> NNUEEvaluator::boardToFeaturesIndices(const chess::board_state& board) const {
        std::vector<int64_t> indices;
        indices.reserve(32); // Reserve space for typical number of pieces

        bool white_king_found = false;
        bool black_king_found = false;

        // Iterate through squares 0-63
        for (int square = 0; square < 64; ++square) {
            // Calculate rank and file from square index
            int rank = square / 8;
            int file = square % 8;
            const auto& piece = board.pieces[rank][file];

            if (piece.type == piece_type::none) continue; // Skip empty squares

            int p_idx = piece_type_to_index(piece.type);
            if (p_idx == -1) { // Should not happen with valid piece types
                 LOG_WARN("Invalid piece type encountered in boardToFeaturesIndices.");
                 continue;
            }

            // Calculate base feature index: piece_index * 64 + square_index
            int64_t feature_index = static_cast<int64_t>(p_idx) * 64 + square;

            // Add offset for black pieces
            if (piece.piece_player == player::white) {
                indices.push_back(feature_index);
                if (piece.type == piece_type::king) white_king_found = true;
            } else { // Black piece
                indices.push_back(feature_index + FEATURES_PER_COLOR); // Add offset
                if (piece.type == piece_type::king) black_king_found = true;
            }
        }

        // Return empty vector if a king is missing (invalid position for evaluation)
        if (!white_king_found || !black_king_found) {
            // LOG_WARN("Evaluation skipped: King missing from board state."); // Can be noisy
            return {}; // Return empty vector
        }

        return indices;
    }


    // evaluate: Perform NNUE inference
    double NNUEEvaluator::evaluate(const chess::board_state& board) {
        // --- Prepare Input Tensors ---
        std::vector<int64_t> feature_indices = boardToFeaturesIndices(board);

        // Handle invalid positions (e.g., missing king) where indices are empty
        if (feature_indices.empty()) {
             // LOG_WARN("NNUE Evaluation skipped due to empty feature indices (invalid board?).");
             return 0.0; // Return neutral score for invalid positions
        }

        // Indices Tensor
        std::array<int64_t, 1> indices_shape = {static_cast<int64_t>(feature_indices.size())};
        Ort::Value indices_tensor = Ort::Value::CreateTensor<int64_t>(
            *memory_info_cpu_, feature_indices.data(), feature_indices.size(), indices_shape.data(), indices_shape.size());

        // Offsets Tensor (for batch size 1)
        std::vector<int64_t> offsets_data = {0}; // Start at index 0
        std::array<int64_t, 1> offsets_shape = {static_cast<int64_t>(offsets_data.size())};
        Ort::Value offsets_tensor = Ort::Value::CreateTensor<int64_t>(
            *memory_info_cpu_, offsets_data.data(), offsets_data.size(), offsets_shape.data(), offsets_shape.size());

        // --- Run Inference ---
        std::array<Ort::Value, 2> input_tensors = {std::move(indices_tensor), std::move(offsets_tensor)};
        std::vector<Ort::Value> output_tensors; // Use vector for output
        try {
             output_tensors = ort_impl_->session->Run(
                 Ort::RunOptions{nullptr},
                 input_node_names_.data(),  // Pointer to array of const char*
                 input_tensors.data(),      // Pointer to array of Ort::Value
                 input_tensors.size(),      // Number of inputs
                 output_node_names_.data(), // Pointer to array of const char*
                 output_node_names_.size()  // Number of outputs (expecting 1)
             );
        } catch (const Ort::Exception& e) {
            LOG_ERROR("ONNX Runtime inference failed: " + std::string(e.what()));
            throw std::runtime_error("ONNX Runtime inference failed: " + std::string(e.what())); // Re-throw
        } catch(...) {
             LOG_ERROR("Unknown error during ONNX Runtime inference.");
             throw std::runtime_error("Unknown error during ONNX inference.");
        }


        // --- Process Output Tensor ---
        if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
             LOG_ERROR("Invalid or missing output tensor from ONNX Runtime.");
             throw std::runtime_error("Invalid output tensor from NNUE model.");
        }

        auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        if (type_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || type_info.GetElementCount() != 1) {
             LOG_ERROR("Unexpected NNUE output tensor shape or type.");
             throw std::runtime_error("Unexpected NNUE output tensor format.");
        }

        // Get the result
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        double raw_score = static_cast<double>(output_data[0]);

        // Adjust score based on whose turn it is
        // NNUE models typically evaluate from White's perspective.
        // If it's Black's turn, the evaluation needs to be flipped.
        // return (board.current_player == player::black) ? -raw_score : raw_score;
        // Assuming the model outputs score relative to the CURRENT player to move:
        return raw_score; // No flip needed if model outputs relative score
    }

} // namespace chess::ai