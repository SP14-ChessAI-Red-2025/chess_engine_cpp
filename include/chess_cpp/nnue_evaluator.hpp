#pragma once

#include "chess_rules.hpp" // For chess::board_state
#include <vector>
#include <string>
#include <memory> // For std::unique_ptr
#include <cstdint> // For int64_t

// Forward declare ONNX Runtime types to avoid including heavy header here
namespace Ort {
    struct Env;
    struct Session;
    struct SessionOptions;
    struct AllocatorWithDefaultOptions;
    struct MemoryInfo; // Forward declare MemoryInfo
    struct Value;     // Forward declare Value
}

namespace chess::ai {

    /**
     * @brief Evaluates chess positions using an NNUE ONNX model.
     * Assumes the model takes sparse ('feature_indices', 'offsets') inputs.
     */
    class NNUEEvaluator {
    public:
        /**
         * @brief Constructs the evaluator and loads the ONNX model.
         * @param model_path Filesystem path to the .onnx model file.
         * @throws std::runtime_error if the model cannot be loaded or Ort::Env fails.
         */
        explicit NNUEEvaluator(const std::string& model_path);

        /**
         * @brief Destructor.
         */
        ~NNUEEvaluator();

        // Disable copy and move semantics for simplicity with unique_ptr member
        NNUEEvaluator(const NNUEEvaluator&) = delete;
        NNUEEvaluator& operator=(const NNUEEvaluator&) = delete;
        NNUEEvaluator(NNUEEvaluator&&) = delete;
        NNUEEvaluator& operator=(NNUEEvaluator&&) = delete;


        /**
         * @brief Evaluates the given board state using the loaded NNUE model.
         * @param board The current board state.
         * @return The evaluation score (e.g., centipawns) from the perspective of the current player.
         * Positive means advantage for the current player, negative means disadvantage.
         * @throws std::runtime_error if ONNX inference fails.
         */
        double evaluate(const chess::board_state& board);

    private:
        /**
         * @brief Converts a board state into the feature indices expected by the NNUE model.
         * Replicates the logic from the Python `extract_features_indices` function.
         * Assumes a 768-feature space (6 piece types * 2 colors * 64 squares).
         * @param board The board state to convert.
         * @return A vector<int64_t> containing the active feature indices.
         */
        std::vector<int64_t> boardToFeaturesIndices(const chess::board_state& board) const;

        // ONNX Runtime environment and session management using PImpl idiom
        // to hide ONNX headers from this header file.
        struct OrtImpl;
        std::unique_ptr<OrtImpl> ort_impl_;

        // Store input/output node names based on the export script
        const char* input_node_name_indices_ = "feature_indices";
        const char* input_node_name_offsets_ = "offsets";
        const char* output_node_name_ = "evaluation"; // Corrected output name
        std::vector<const char*> input_node_names_;
        std::vector<const char*> output_node_names_;

        // Cache memory info
        std::unique_ptr<Ort::MemoryInfo> memory_info_cpu_;

    };

} // namespace chess::ai
