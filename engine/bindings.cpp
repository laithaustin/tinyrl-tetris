#include <cstdint>
#include <cstring>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "tetrisGame.h"
#include "constants.h"

namespace py = pybind11;

// helper methods
// helper method to convert vec2d to numpy
template <typename T>
py::array_t<T> vec2d_to_numpy(const std::vector<std::vector<T>>& vec) {
    // first we can convert our vector to an array
    size_t rows = vec.size();
    size_t cols = vec[0].size();
    // flatten to 1d array
    py::array_t<T> arr({rows, cols});  // Create array with shape
    auto buf = arr.template mutable_unchecked<2>();    // Get mutable buffer with 2D access

    // Now copy data
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            buf(i, j) = vec[i][j];
        }
    }

    return arr;
}

// helper method to convert obs to dictionary
py::dict obs_to_dict(const Observation& obs) {
    py::dict d;
    d["board"] = vec2d_to_numpy(obs.board);
    d["active_tetromino"] = vec2d_to_numpy(obs.active_tetromino);
    d["holder"] = vec2d_to_numpy(obs.holder);
    d["queue"] = vec2d_to_numpy(obs.queue);
    return d;
}

// ═══════════════════════════════════════════════════════════════════════════
// Write-through wrapper
//
// Pre-allocates numpy arrays once at construction time.  On every step/reset
// the observation is copied from the C++ vector-of-vectors into those
// fixed buffers via memcpy (one pass, no Python object allocation).
// The same dict and the same array objects are returned on every call,
// eliminating all per-step heap pressure.
// ═══════════════════════════════════════════════════════════════════════════
struct TetrisEnvWT {
    TetrisGame game;
    uint8_t    qs;   // queue_size

    // Pre-allocated, C-contiguous numpy arrays
    py::array_t<uint8_t> np_board;
    py::array_t<uint8_t> np_active;
    py::array_t<uint8_t> np_holder;
    py::array_t<uint8_t> np_queue;

    // Raw pointers into the numpy buffers (obtained once, reused every step)
    uint8_t* board_ptr;
    uint8_t* active_ptr;
    uint8_t* holder_ptr;
    uint8_t* queue_ptr;

    // Pre-built dict that always references the same numpy objects
    py::dict obs_dict;

    TetrisEnvWT(TimeManager::Mode m, uint8_t queue_size)
        : game(m, queue_size), qs(queue_size)
    {
        // Allocate once -------------------------------------------------
        np_board  = py::array_t<uint8_t>({Observation::BoardH, Observation::BoardW});
        np_active = py::array_t<uint8_t>({Observation::BoardH, Observation::BoardW});
        np_holder = py::array_t<uint8_t>({Tetris::PIECE_SIZE,  Tetris::PIECE_SIZE});
        np_queue  = py::array_t<uint8_t>({(int)(queue_size * Tetris::PIECE_SIZE), Tetris::PIECE_SIZE});

        // Cache raw pointers (valid for the lifetime of this object)
        board_ptr  = np_board.mutable_data();
        active_ptr = np_active.mutable_data();
        holder_ptr = np_holder.mutable_data();
        queue_ptr  = np_queue.mutable_data();

        // Build the dict once (always references the same array objects)
        obs_dict["board"]            = np_board;
        obs_dict["active_tetromino"] = np_active;
        obs_dict["holder"]           = np_holder;
        obs_dict["queue"]            = np_queue;
    }

    // Write the C++ observation into the pre-allocated numpy buffers via memcpy
    inline void write_obs() {
        const Observation& obs = game.obs;

        // board and active_tetromino: BoardH rows of BoardW bytes
        for (int y = 0; y < Observation::BoardH; y++) {
            std::memcpy(board_ptr  + y * Observation::BoardW, obs.board[y].data(),            Observation::BoardW);
            std::memcpy(active_ptr + y * Observation::BoardW, obs.active_tetromino[y].data(), Observation::BoardW);
        }
        // holder: PIECE_SIZE rows of PIECE_SIZE bytes
        for (int y = 0; y < Tetris::PIECE_SIZE; y++) {
            std::memcpy(holder_ptr + y * Tetris::PIECE_SIZE, obs.holder[y].data(), Tetris::PIECE_SIZE);
        }
        // queue: queue_size*PIECE_SIZE rows of PIECE_SIZE bytes
        int queue_rows = qs * Tetris::PIECE_SIZE;
        for (int y = 0; y < queue_rows; y++) {
            std::memcpy(queue_ptr + y * Tetris::PIECE_SIZE, obs.queue[y].data(), Tetris::PIECE_SIZE);
        }
    }

    py::dict reset() {
        game.reset();
        write_obs();
        return obs_dict;
    }

    py::tuple step(int action) {
        StepResult result = game.step(action);
        write_obs();
        return py::make_tuple(obs_dict, result.reward, result.terminated, py::dict());
    }
};


PYBIND11_MODULE(tinyrl_tetris, m) {
    m.doc() = "TinyRL Tetris Python Bindings";

    // ── Original env (allocates new numpy arrays on every step) ──────────
    py::class_<TetrisGame>(m, "TetrisEnv")
        .def(py::init<TimeManager::Mode, uint8_t>(),
            py::arg("mode"), py::arg("queue_size") = 3)
        .def("reset", [](TetrisGame& self) {
            self.reset();
            return obs_to_dict(self.obs);
        })
        .def("step", [](TetrisGame& self, int action) {
            StepResult result = self.step(action);
            return py::make_tuple(
                obs_to_dict(result.obs),
                result.reward,
                result.terminated,
                py::dict()  // empty info dict
            );
        })
        .def_property_readonly("obs", [](TetrisGame& self) {
            return obs_to_dict(self.obs);
        })
        .def_readonly("score", &TetrisGame::score)
        .def_readonly("game_over", &TetrisGame::game_over);

    // ── Write-through env (zero per-step allocation) ──────────────────────
    py::class_<TetrisEnvWT>(m, "TetrisEnvWT")
        .def(py::init<TimeManager::Mode, uint8_t>(),
            py::arg("mode"), py::arg("queue_size") = 3)
        .def("reset", &TetrisEnvWT::reset)
        .def("step",  &TetrisEnvWT::step)
        .def_property_readonly("obs", [](TetrisEnvWT& self) {
            return self.obs_dict;
        })
        .def_property_readonly("score", [](TetrisEnvWT& self) {
            return self.game.score;
        })
        .def_property_readonly("game_over", [](TetrisEnvWT& self) {
            return self.game.game_over;
        });

    // Expose enums
    py::enum_<Action>(m, "Action")
        .value("LEFT", Action::LEFT)
        .value("RIGHT", Action::RIGHT)
        .value("DOWN", Action::DOWN)
        .value("CW", Action::CW)
        .value("CCW", Action::CCW)
        .value("DROP", Action::DROP)
        .value("SWAP", Action::SWAP)
        .value("NOOP", Action::NOOP)
        .export_values();

    py::enum_<TimeManager::Mode>(m, "TimeMode")
        .value("REALTIME", TimeManager::Mode::REALTIME)
        .value("STEPPED", TimeManager::Mode::SIMULATION)
        .export_values();

}
