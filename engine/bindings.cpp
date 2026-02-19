#include <cstdint>
#include <cstring>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "tetrisGame.h"
#include "constants.h"

namespace py = pybind11;

// Build a fresh dict of newly allocated numpy arrays from an Observation.
// One memcpy per field — no element-wise loops.
py::dict obs_to_dict(const Observation& obs, int qs) {
    py::dict d;

    auto board = py::array_t<uint8_t>({Observation::BoardH, Observation::BoardW});
    std::memcpy(board.mutable_data(), obs.board.data(), sizeof(obs.board));
    d["board"] = board;

    auto active = py::array_t<uint8_t>({Observation::BoardH, Observation::BoardW});
    std::memcpy(active.mutable_data(), obs.active_tetromino.data(), sizeof(obs.active_tetromino));
    d["active_tetromino"] = active;

    auto holder = py::array_t<uint8_t>({Tetris::PIECE_SIZE, Tetris::PIECE_SIZE});
    std::memcpy(holder.mutable_data(), obs.holder.data(), sizeof(obs.holder));
    d["holder"] = holder;

    int queue_bytes = qs * Tetris::PIECE_SIZE * Tetris::PIECE_SIZE;
    auto queue = py::array_t<uint8_t>({qs * Tetris::PIECE_SIZE, Tetris::PIECE_SIZE});
    std::memcpy(queue.mutable_data(), obs.queue.data(), queue_bytes);
    d["queue"] = queue;

    return d;
}

// ═══════════════════════════════════════════════════════════════════════════
// Write-through wrapper
//
// Pre-allocates numpy arrays once at construction time.  On every step/reset
// the observation is copied from the C++ flat arrays into those fixed buffers
// via 4 memcpy calls — no Python object allocation, no element-wise loops.
// The same dict and the same array objects are returned on every call.
// ═══════════════════════════════════════════════════════════════════════════
struct TetrisEnvWT {
    TetrisGame game;
    int        qs;   // queue_size

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
        np_board  = py::array_t<uint8_t>({Observation::BoardH, Observation::BoardW});
        np_active = py::array_t<uint8_t>({Observation::BoardH, Observation::BoardW});
        np_holder = py::array_t<uint8_t>({Tetris::PIECE_SIZE,  Tetris::PIECE_SIZE});
        np_queue  = py::array_t<uint8_t>({qs * Tetris::PIECE_SIZE, Tetris::PIECE_SIZE});

        board_ptr  = np_board.mutable_data();
        active_ptr = np_active.mutable_data();
        holder_ptr = np_holder.mutable_data();
        queue_ptr  = np_queue.mutable_data();

        obs_dict["board"]            = np_board;
        obs_dict["active_tetromino"] = np_active;
        obs_dict["holder"]           = np_holder;
        obs_dict["queue"]            = np_queue;
    }

    // 4 memcpy calls — one per field, exploiting the flat array layout.
    inline void write_obs() {
        const Observation& obs = game.obs;
        std::memcpy(board_ptr,  obs.board.data(),            sizeof(obs.board));
        std::memcpy(active_ptr, obs.active_tetromino.data(), sizeof(obs.active_tetromino));
        std::memcpy(holder_ptr, obs.holder.data(),           sizeof(obs.holder));
        std::memcpy(queue_ptr,  obs.queue.data(),
                    qs * Tetris::PIECE_SIZE * Tetris::PIECE_SIZE);
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


// ═══════════════════════════════════════════════════════════════════════════
// Vectorised write-through env
//
// Holds num_envs TetrisGame objects in a contiguous vector.  Pre-allocates
// stacked (num_envs, ...) numpy arrays once.  A single Python call to step()
// drives all envs through a tight C++ loop — one Python→C boundary crossing
// amortised over N environments, same pattern as PufferLib's vec_step.
// Done envs are auto-reset in-place and their obs slice overwritten.
// ═══════════════════════════════════════════════════════════════════════════
struct VecTetrisEnvWT {
    int num_envs;
    int qs;
    std::vector<TetrisGame> envs;

    // Stacked observation arrays: shape (num_envs, ...)
    py::array_t<uint8_t> np_board;
    py::array_t<uint8_t> np_active;
    py::array_t<uint8_t> np_holder;
    py::array_t<uint8_t> np_queue;
    py::array_t<float>   np_rewards;
    py::array_t<uint8_t> np_terminals;

    uint8_t* board_ptr;
    uint8_t* active_ptr;
    uint8_t* holder_ptr;
    uint8_t* queue_ptr;
    float*   rewards_ptr;
    uint8_t* terminals_ptr;

    py::dict obs_dict;

    // Per-env byte strides (flat slice size for each field)
    int board_stride;
    int active_stride;
    int holder_stride;
    int queue_stride;

    VecTetrisEnvWT(TimeManager::Mode m, uint8_t queue_size, int n)
        : num_envs(n), qs(queue_size)
    {
        envs.reserve(n);
        for (int i = 0; i < n; i++)
            envs.emplace_back(m, queue_size);

        board_stride  = Observation::BoardH * Observation::BoardW;
        active_stride = Observation::BoardH * Observation::BoardW;
        holder_stride = Tetris::PIECE_SIZE  * Tetris::PIECE_SIZE;
        queue_stride  = qs * Tetris::PIECE_SIZE * Tetris::PIECE_SIZE;

        np_board     = py::array_t<uint8_t>({n, Observation::BoardH, Observation::BoardW});
        np_active    = py::array_t<uint8_t>({n, Observation::BoardH, Observation::BoardW});
        np_holder    = py::array_t<uint8_t>({n, Tetris::PIECE_SIZE,  Tetris::PIECE_SIZE});
        np_queue     = py::array_t<uint8_t>({n, qs * Tetris::PIECE_SIZE, Tetris::PIECE_SIZE});
        np_rewards   = py::array_t<float>  ({n});
        np_terminals = py::array_t<uint8_t>({n});

        board_ptr     = np_board.mutable_data();
        active_ptr    = np_active.mutable_data();
        holder_ptr    = np_holder.mutable_data();
        queue_ptr     = np_queue.mutable_data();
        rewards_ptr   = np_rewards.mutable_data();
        terminals_ptr = np_terminals.mutable_data();

        obs_dict["board"]            = np_board;
        obs_dict["active_tetromino"] = np_active;
        obs_dict["holder"]           = np_holder;
        obs_dict["queue"]            = np_queue;
    }

    inline void write_obs_for(int i) {
        const Observation& obs = envs[i].obs;
        std::memcpy(board_ptr  + i * board_stride,  obs.board.data(),            board_stride);
        std::memcpy(active_ptr + i * active_stride, obs.active_tetromino.data(), active_stride);
        std::memcpy(holder_ptr + i * holder_stride, obs.holder.data(),           holder_stride);
        std::memcpy(queue_ptr  + i * queue_stride,  obs.queue.data(),            queue_stride);
    }

    py::dict reset() {
        for (int i = 0; i < num_envs; i++) {
            envs[i].reset();
            write_obs_for(i);
        }
        return obs_dict;
    }

    py::tuple step(py::array_t<int32_t, py::array::c_style | py::array::forcecast> actions) {
        auto act = actions.unchecked<1>();
        for (int i = 0; i < num_envs; i++) {
            StepResult result = envs[i].step(act(i));
            write_obs_for(i);
            rewards_ptr[i]   = result.reward;
            terminals_ptr[i] = result.terminated ? 1 : 0;
            if (result.terminated) {
                envs[i].reset();
                write_obs_for(i);
            }
        }
        return py::make_tuple(obs_dict, np_rewards, np_terminals);
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
            return obs_to_dict(self.obs, self.queue_size);
        })
        .def("step", [](TetrisGame& self, int action) {
            StepResult result = self.step(action);
            return py::make_tuple(
                obs_to_dict(self.obs, self.queue_size),
                result.reward,
                result.terminated,
                py::dict()
            );
        })
        .def_property_readonly("obs", [](TetrisGame& self) {
            return obs_to_dict(self.obs, self.queue_size);
        })
        .def_readonly("score", &TetrisGame::score)
        .def_readonly("game_over", &TetrisGame::game_over);

    // ── Write-through env (zero per-step allocation, 4-memcpy flush) ─────
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

    // ── Vectorised write-through env ─────────────────────────────────────
    py::class_<VecTetrisEnvWT>(m, "VecTetrisEnvWT")
        .def(py::init<TimeManager::Mode, uint8_t, int>(),
            py::arg("mode"), py::arg("queue_size") = 3, py::arg("num_envs") = 1)
        .def("reset", &VecTetrisEnvWT::reset)
        .def("step",  &VecTetrisEnvWT::step)
        .def_property_readonly("obs", [](VecTetrisEnvWT& self) {
            return self.obs_dict;
        })
        .def_readonly("num_envs", &VecTetrisEnvWT::num_envs);

    // Expose enums
    py::enum_<Action>(m, "Action")
        .value("LEFT",  Action::LEFT)
        .value("RIGHT", Action::RIGHT)
        .value("DOWN",  Action::DOWN)
        .value("CW",    Action::CW)
        .value("CCW",   Action::CCW)
        .value("DROP",  Action::DROP)
        .value("SWAP",  Action::SWAP)
        .value("NOOP",  Action::NOOP)
        .export_values();

    py::enum_<TimeManager::Mode>(m, "TimeMode")
        .value("REALTIME", TimeManager::Mode::REALTIME)
        .value("STEPPED",  TimeManager::Mode::SIMULATION)
        .export_values();
}
