#pragma once
#include <array>
#include <vector>
#include <cstdint>
#include "constants.h"
#include "timeManager.h"

enum Action : uint8_t {
    LEFT, RIGHT, DOWN, CW, CCW, DROP, SWAP, NOOP
};

// All fields are plain fixed-size arrays — no heap allocation, one contiguous
// block per field, trivially copyable.
struct Observation {
    static constexpr int BoardW       = 18;
    static constexpr int BoardH       = 24;
    static constexpr int MaxQueueSize = 7;  // maximum supported queue depth

    std::array<std::array<uint8_t, BoardW>, BoardH> board;
    std::array<std::array<uint8_t, BoardW>, BoardH> active_tetromino;
    std::array<std::array<uint8_t, Tetris::PIECE_SIZE>, Tetris::PIECE_SIZE>                       holder;
    std::array<std::array<uint8_t, Tetris::PIECE_SIZE>, MaxQueueSize * Tetris::PIECE_SIZE>        queue;
};

// StepResult carries only the scalar outputs of a step; callers read the
// updated observation directly from TetrisGame::obs.
struct StepResult {
    float reward;
    bool  terminated;
};

class TetrisGame {
public:
    TetrisGame(TimeManager::Mode m, uint8_t queue_size = 3);
    void reset();
    StepResult step(int action);
    float getReward();
    bool isGameOver();
    uint8_t getNextPiece();
    uint8_t setLastPiece(uint8_t val);
    void updateGameState();
    void updateActiveMask();
    void updateObservation();
    void applyAction(uint8_t action);
#ifndef NO_TERMINAL_LOOP
    void loop();
#endif

    // Made public for testing - consider friend class for production
    void spawnPiece();
    bool checkCollision();
    void lockPiece();
    int clearLines();
    void completeClearLines();
    int clearLine(uint8_t row);

    // general board state data
    int score;
    int scored; // points accumulated in one cycle
    bool game_over;
    int8_t queue_size;
    std::vector<int> clearing_lines;  // Lines currently being cleared (for animation)
    TimeManager tm;
    Observation obs;
    std::vector<uint8_t> queue;
    uint8_t queue_index; // circular buffer
    uint8_t holder_type;

    // current piece data
    int8_t current_x;
    int8_t current_y;
    uint8_t current_piece_type;
    uint8_t rotation; // 0-3 possible options
};
