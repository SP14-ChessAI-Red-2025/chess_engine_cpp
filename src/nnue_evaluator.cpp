#include "chess_cpp/nnue_evaluator.hpp"
#include "chess_cpp/chess_rules.hpp"
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <array>
#include <memory>
#include <iostream>

// #include <vector> // Potentially implicitly included

namespace chess::ai {

    // PImpl Idiom for ONNX Runtime details
    struct NNUEEvaluator::OrtImpl {
        Ort::Env env;
        Ort::SessionOptions session_options;
        std::unique_ptr<Ort::Session> session;

        OrtImpl(const std::string& model_path)
            : env(ORT_LOGGING_LEVEL_WARNING, "chess_nnue_evaluator")
        {

            // Let ONNX Runtime manage CPU threads automatically
            session_options.SetIntraOpNumThreads(0);

            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            // Attempt to enable CUDA Execution Provider
            OrtCUDAProviderOptions cuda_options{};
            session_options.AppendExecutionProvider_CUDA(cuda_options);

            try {
                #ifdef _WIN32
                    std::wstring w_model_path(model_path.begin(), model_path.end());
                    session = std::make_unique<Ort::Session>(env, w_model_path.c_str(), session_options);
                #else
                    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
                #endif

                 if (!session) {
                    throw std::runtime_error("Failed to create ONNX Runtime session for model: " + model_path);
                }
                std::cout << "[INFO] ONNX Runtime session created." << std::endl;

            } catch (const Ort::Exception& ort_exc) {
                 std::cerr << "[WARNING] ONNX Runtime exception during session creation (CUDA might be unavailable or model incompatible): "
                           << ort_exc.what() << std::endl;
                 // Attempt to create session again without CUDA (fallback to CPU)
                 session_options = Ort::SessionOptions(); // Reset options
                 session_options.SetIntraOpNumThreads(0); // Set CPU threads again
                 session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
                 try {
                     #ifdef _WIN32
                         std::wstring w_model_path(model_path.begin(), model_path.end());
                         session = std::make_unique<Ort::Session>(env, w_model_path.c_str(), session_options);
                     #else
                         session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
                     #endif
                     if (!session) {
                        throw std::runtime_error("Failed to create fallback ONNX Runtime CPU session for model: " + model_path);
                     }
                     std::cout << "[INFO] Created ONNX Runtime session using CPU fallback." << std::endl;
                 } catch (const Ort::Exception& fallback_exc) {
                      throw std::runtime_error("Failed to create even fallback CPU ONNX session: " + std::string(fallback_exc.what()));
                 }
            }
        }
    };


    NNUEEvaluator::NNUEEvaluator(const std::string& model_path)
        : input_node_names_{input_node_name_}, output_node_names_{output_node_name_}
    {
        try {
             ort_impl_ = std::make_unique<OrtImpl>(model_path);
        } catch (const Ort::Exception& e) {
            throw std::runtime_error("Failed to initialize ONNX Runtime Environment/Session: " + std::string(e.what()));
        } catch (const std::runtime_error& e) {
             throw;
        } catch (...) {
             throw std::runtime_error("Unknown error during NNUEEvaluator initialization.");
        }
    }

    NNUEEvaluator::~NNUEEvaluator() = default;


    std::vector<float> NNUEEvaluator::boardToFeatures(const chess::board_state& board) const {
        std::vector<float> features(768, 0.0f); // Assuming 768 features
        for (int r = 0; r < 8; ++r) {
            for (int f = 0; f < 8; ++f) {
                const auto& piece = board.pieces[r][f];
                if (piece.type != piece_type::none) {
                    int piece_index = static_cast<int>(piece.type) - 1;
                    if (piece.piece_player == player::black) {
                        piece_index += 6;
                    }
                    int square_index = r * 8 + f;
                    // Simple one-hot encoding: feature = square * 12 + piece_index
                    int feature_index = square_index * 12 + piece_index;
                    if (feature_index >= 0 && feature_index < 768) {
                        features[feature_index] = 1.0f;
                    } else {
                        std::cerr << "[ERROR] Calculated feature index out of bounds: " << feature_index << std::endl;
                    }
                }
            }
        }
        return features;
    }

    double NNUEEvaluator::evaluate(const chess::board_state& board) {
        auto features = boardToFeatures(board);

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 2> input_shape = {1, static_cast<int64_t>(features.size())}; // Batch size 1

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, features.data(), features.size(), input_shape.data(), input_shape.size());

        std::vector<Ort::Value> output_tensors;

        try {
             if (!ort_impl_ || !ort_impl_->session) {
                 throw std::runtime_error("ONNX Runtime session is not initialized.");
             }

             output_tensors = ort_impl_->session->Run(Ort::RunOptions{nullptr},
                                            input_node_names_.data(), &input_tensor, 1,
                                            output_node_names_.data(), 1);

            if (output_tensors.empty() || !output_tensors[0].IsTensor()) {
                 throw std::runtime_error("ONNX inference did not return a valid tensor.");
            }

            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            double evaluation_score = static_cast<double>(output_data[0]);

            // Adjust score perspective if needed
            if (board.current_player == player::black) {
                return -evaluation_score;
            } else {
                return evaluation_score;
            }

        } catch (const Ort::Exception& e) {
            throw std::runtime_error(std::string("ONNX Runtime inference failed: ") + e.what());
        } catch (const std::runtime_error& e) {
             throw;
        }
    }

} // namespace chess::ai