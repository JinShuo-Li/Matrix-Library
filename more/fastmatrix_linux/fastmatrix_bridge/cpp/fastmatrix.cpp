#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {
thread_local std::string g_last_error;

inline int normalize_threads(int requested) {
    if (requested > 0) return requested;
    unsigned int hc = std::thread::hardware_concurrency();
    return hc == 0 ? 4 : static_cast<int>(hc);
}

void set_error(const char* msg) {
    g_last_error = msg ? msg : "unknown error";
}

void add_worker(const double* a, const double* b, double* out, std::int64_t start, std::int64_t end) {
    for (std::int64_t i = start; i < end; ++i) out[i] = a[i] + b[i];
}

void sub_worker(const double* a, const double* b, double* out, std::int64_t start, std::int64_t end) {
    for (std::int64_t i = start; i < end; ++i) out[i] = a[i] - b[i];
}

void transpose_worker(const double* src, double* dst, std::int64_t rows, std::int64_t cols,
                      std::int64_t r0, std::int64_t r1) {
    for (std::int64_t i = r0; i < r1; ++i) {
        const double* row = src + i * cols;
        for (std::int64_t j = 0; j < cols; ++j) {
            dst[j * rows + i] = row[j];
        }
    }
}

void matmul_worker(const double* a,
                   const double* bt,
                   double* c,
                   std::int64_t n,
                   std::int64_t k,
                   std::int64_t row_begin,
                   std::int64_t row_end,
                   int block_size) {
    for (std::int64_t ii = row_begin; ii < row_end; ii += block_size) {
        std::int64_t i_max = std::min<std::int64_t>(ii + block_size, row_end);
        for (std::int64_t jj = 0; jj < k; jj += block_size) {
            std::int64_t j_max = std::min<std::int64_t>(jj + block_size, k);
            for (std::int64_t pp = 0; pp < n; pp += block_size) {
                std::int64_t p_max = std::min<std::int64_t>(pp + block_size, n);
                for (std::int64_t i = ii; i < i_max; ++i) {
                    const double* a_row = a + i * n;
                    double* c_row = c + i * k;
                    for (std::int64_t j = jj; j < j_max; ++j) {
                        const double* bt_row = bt + j * n;
                        double sum = (pp == 0) ? 0.0 : c_row[j];
                        for (std::int64_t p = pp; p < p_max; ++p) {
                            sum += a_row[p] * bt_row[p];
                        }
                        c_row[j] = sum;
                    }
                }
            }
        }
    }
}

void parallel_for_flat(const double* a, const double* b, double* out,
                       std::int64_t total, int threads,
                       void (*worker)(const double*, const double*, double*, std::int64_t, std::int64_t)) {
    threads = std::max(1, std::min<int>(threads, total == 0 ? 1 : static_cast<int>(total)));
    std::vector<std::thread> pool;
    pool.reserve(threads);
    std::int64_t chunk = (total + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
        std::int64_t start = t * chunk;
        std::int64_t end = std::min(total, start + chunk);
        if (start >= end) break;
        pool.emplace_back(worker, a, b, out, start, end);
    }
    for (auto& th : pool) th.join();
}

}  // namespace

extern "C" {

const char* fm_last_error() {
    return g_last_error.c_str();
}

int fm_hardware_threads() {
    return normalize_threads(0);
}

int fm_add(const double* a, const double* b, double* out,
           std::int64_t rows, std::int64_t cols, int threads) {
    if (!a || !b || !out) {
        set_error("null pointer in fm_add");
        return -1;
    }
    if (rows < 0 || cols < 0) {
        set_error("negative dimension in fm_add");
        return -2;
    }
    parallel_for_flat(a, b, out, rows * cols, normalize_threads(threads), add_worker);
    return 0;
}

int fm_sub(const double* a, const double* b, double* out,
           std::int64_t rows, std::int64_t cols, int threads) {
    if (!a || !b || !out) {
        set_error("null pointer in fm_sub");
        return -1;
    }
    if (rows < 0 || cols < 0) {
        set_error("negative dimension in fm_sub");
        return -2;
    }
    parallel_for_flat(a, b, out, rows * cols, normalize_threads(threads), sub_worker);
    return 0;
}

int fm_transpose(const double* src, double* dst,
                 std::int64_t rows, std::int64_t cols, int threads) {
    if (!src || !dst) {
        set_error("null pointer in fm_transpose");
        return -1;
    }
    if (rows < 0 || cols < 0) {
        set_error("negative dimension in fm_transpose");
        return -2;
    }
    int real_threads = std::max(1, std::min<int>(normalize_threads(threads), rows == 0 ? 1 : static_cast<int>(rows)));
    std::vector<std::thread> pool;
    pool.reserve(real_threads);
    std::int64_t chunk = (rows + real_threads - 1) / real_threads;
    for (int t = 0; t < real_threads; ++t) {
        std::int64_t r0 = t * chunk;
        std::int64_t r1 = std::min<std::int64_t>(rows, r0 + chunk);
        if (r0 >= r1) break;
        pool.emplace_back(transpose_worker, src, dst, rows, cols, r0, r1);
    }
    for (auto& th : pool) th.join();
    return 0;
}

int fm_matmul(const double* a, const double* b, double* c,
              std::int64_t m, std::int64_t n, std::int64_t k,
              int threads, int block_size) {
    if (!a || !b || !c) {
        set_error("null pointer in fm_matmul");
        return -1;
    }
    if (m < 0 || n < 0 || k < 0) {
        set_error("negative dimension in fm_matmul");
        return -2;
    }
    if (block_size <= 0) {
        set_error("block_size must be positive");
        return -3;
    }

    std::fill(c, c + m * k, 0.0);
    std::vector<double> bt(k * n);
    for (std::int64_t i = 0; i < n; ++i) {
        const double* b_row = b + i * k;
        for (std::int64_t j = 0; j < k; ++j) bt[j * n + i] = b_row[j];
    }

    int real_threads = std::max(1, std::min<int>(normalize_threads(threads), m == 0 ? 1 : static_cast<int>(m)));
    std::vector<std::thread> pool;
    pool.reserve(real_threads);
    std::int64_t chunk = (m + real_threads - 1) / real_threads;
    for (int t = 0; t < real_threads; ++t) {
        std::int64_t row_begin = t * chunk;
        std::int64_t row_end = std::min<std::int64_t>(m, row_begin + chunk);
        if (row_begin >= row_end) break;
        pool.emplace_back(matmul_worker, a, bt.data(), c, n, k, row_begin, row_end, block_size);
    }
    for (auto& th : pool) th.join();
    return 0;
}

}
