#include <chess_cpp/chess_rules.hpp>
#include <chess_cpp/chess_ai.hpp>
#include <chess_cpp/nnue_evaluator.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <cstddef>
#include <exception>

int main(int argc, char* argv[]) { // Allow command-line argument for model path
    auto board = chess::board_state::initial_board_state();
    chess::previous_board_states game_history;
    std::vector<chess::chess_move> moves;

    try {
        moves = chess::get_valid_moves(board);
    } catch (const std::exception& e) {
         std::cerr << "Error getting initial valid moves: " << e.what() << std::endl;
         return 1;
    } catch (...) {
         std::cerr << "Unknown error getting initial valid moves." << std::endl;
         return 1;
    }

    std::cout << "Initial num_moves: " << moves.size() << std::endl;

    // --- Get model path from command line or use a default ---
    std::string model_path;
    if (argc > 1) {
        model_path = argv[1];
        std::cout << "Using model path from command line: " << model_path << std::endl;
    } else {
        std::cerr << "Error: No model path provided. Usage: " << argv[0] << " <path_to_model.onnx>" << std::endl;
        return 1;
    }
    // --- End model path handling ---

    try {
        chess::ai::chess_ai_state ai_player(model_path);
        std::cout << "AI Initialized successfully using model: " << model_path << std::endl;

        // Example: Make one AI move
        if (board.status == chess::game_status::normal) {
            std::cout << "Requesting AI move..." << std::endl;
            ai_player.make_move(board, 5, game_history); // Use a default difficulty for testing
            std::cout << "AI move applied. New board state status: " << static_cast<int>(board.status) << std::endl;
        }

    } catch (const std::exception& e) {
         std::cerr << "Failed to initialize or use AI state: " << e.what() << std::endl;
         return 1;
    } catch (...) {
         std::cerr << "Unknown error during AI processing." << std::endl;
         return 1;
    }

    return 0;
}