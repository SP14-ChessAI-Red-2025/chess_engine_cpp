import sys

from python.chess import *


def main():
    if len(sys.argv) < 2:
        print("Must specify library path on the command line")
        return

    library_path = sys.argv[1]

    with ChessEngine(library_path) as chess_engine:
        while True:
            valid_moves = chess_engine.get_valid_moves(chess_engine.board_state)

            print(f"{len(valid_moves)} moves found")

            for idx, move in enumerate(valid_moves):
                print(f"{idx+1}: {chess_engine.move_to_str(move)}")

            # print("Enter move number")

            move_number = int(input("Enter move number: "))

            move = valid_moves[move_number - 1]

            print(f"Applying move: {chess_engine.move_to_str(move)}")

            chess_engine.apply_move(move)

            status = chess_engine.board_state.status

            status_str = ["normal", "draw", "checkmate", "resigned"][status]

            print(f"Game status: {status_str}")

            if status == 1:
                print("Game over: draw")

                break
            elif status == 2 or status == 3:
                winner = 0 if chess_engine.board_state.current_player == 1 else 1
                winner_str = ["white", "black"][winner]

                print(f"Game over: {winner_str} wins")

                break


if __name__ == "__main__":
    main()
