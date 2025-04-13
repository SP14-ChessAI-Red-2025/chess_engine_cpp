#pragma once

#include "chess_rules.hpp" // For chess::board_state
#include <vector>
#include <string>
#include <memory> // For std::unique_ptr

// Forward declare ONNX Runtime types to avoid including heavy header here
namespace Ort {
    struct Env;
    struct Session;
    struct SessionOptions;
    struct AllocatorWithDefaultOptions;
}

namespace chess::ai {

    /**
     * @brief Evaluates chess positions using an NNUE ONNX model.
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
         * @brief Converts a board state into the feature vector expected by the NNUE model.
         * Assumes a 768-feature vector (12 piece types * 64 squares).
         * @param board The board state to convert.
         * @return A flat vector<float> containing the features.
         */
        std::vector<float> boardToFeatures(const chess::board_state& board) const;

        // ONNX Runtime environment and session management using PImpl idiom
        // to hide ONNX headers from this header file.
        struct OrtImpl;
        std::unique_ptr<OrtImpl> ort_impl_;

        // Store input/output node names (assuming standard names, adjust if needed)
        const char* input_node_name_ = "input";
        const char* output_node_name_ = "output";
        std::vector<const char*> input_node_names_;
        std::vector<const char*> output_node_names_;
    };

} // namespace chess::ai