#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32) || defined(__CYGWIN__)
#define FM_EXPORT __declspec(dllexport)
#else
#define FM_EXPORT __attribute__((visibility("default")))
#endif

struct MatrixData {
    std::int64_t rows = 0;
    std::int64_t cols = 0;
    std::vector<double> values;

    MatrixData() = default;
    MatrixData(std::int64_t r, std::int64_t c) : rows(r), cols(c), values(static_cast<std::size_t>(r * c), 0.0) {}
};

namespace {
thread_local std::string g_last_error;

inline void set_error(const char* msg) {
    g_last_error = msg ? msg : "unknown error";
}

inline int normalize_threads(int requested) {
    if (requested > 0) return requested;
    unsigned int hc = std::thread::hardware_concurrency();
    return hc == 0 ? 4 : static_cast<int>(hc);
}

inline MatrixData* as_matrix(void* handle) {
    return reinterpret_cast<MatrixData*>(handle);
}

void add_worker(const double* a, const double* b, double* out, std::size_t start, std::size_t end) {
    for (std::size_t i = start; i < end; ++i) out[i] = a[i] + b[i];
}

void sub_worker(const double* a, const double* b, double* out, std::size_t start, std::size_t end) {
    for (std::size_t i = start; i < end; ++i) out[i] = a[i] - b[i];
}

void scalar_worker(const double* a, double scalar, double* out, std::size_t start, std::size_t end) {
    for (std::size_t i = start; i < end; ++i) out[i] = a[i] * scalar;
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

void parallel_binary(const double* a, const double* b, double* out, std::size_t total, int threads,
                     void (*worker)(const double*, const double*, double*, std::size_t, std::size_t)) {
    if (total == 0) return;
    int real_threads = std::max(1, std::min<int>(threads, static_cast<int>(total)));
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(real_threads));
    std::size_t chunk = (total + static_cast<std::size_t>(real_threads) - 1) / static_cast<std::size_t>(real_threads);
    for (int t = 0; t < real_threads; ++t) {
        std::size_t start = static_cast<std::size_t>(t) * chunk;
        std::size_t end = std::min(total, start + chunk);
        if (start >= end) break;
        pool.emplace_back(worker, a, b, out, start, end);
    }
    for (auto& th : pool) th.join();
}

void parallel_scalar(const double* a, double scalar, double* out, std::size_t total, int threads) {
    if (total == 0) return;
    int real_threads = std::max(1, std::min<int>(threads, static_cast<int>(total)));
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(real_threads));
    std::size_t chunk = (total + static_cast<std::size_t>(real_threads) - 1) / static_cast<std::size_t>(real_threads);
    for (int t = 0; t < real_threads; ++t) {
        std::size_t start = static_cast<std::size_t>(t) * chunk;
        std::size_t end = std::min(total, start + chunk);
        if (start >= end) break;
        pool.emplace_back(scalar_worker, a, scalar, out, start, end);
    }
    for (auto& th : pool) th.join();
}

MatrixData* clone_matrix(const MatrixData* src) {
    if (!src) {
        set_error("null handle");
        return nullptr;
    }
    try {
        return new MatrixData(*src);
    } catch (...) {
        set_error("memory allocation failed");
        return nullptr;
    }
}

MatrixData* add_matrix(const MatrixData* a, const MatrixData* b, int requested_threads) {
    if (!a || !b) {
        set_error("null handle in add");
        return nullptr;
    }
    if (a->rows != b->rows || a->cols != b->cols) {
        set_error("add shape mismatch");
        return nullptr;
    }
    try {
        MatrixData* out = new MatrixData(a->rows, a->cols);
        parallel_binary(a->values.data(), b->values.data(), out->values.data(), out->values.size(), normalize_threads(requested_threads), add_worker);
        return out;
    } catch (...) {
        set_error("memory allocation failed");
        return nullptr;
    }
}

MatrixData* sub_matrix(const MatrixData* a, const MatrixData* b, int requested_threads) {
    if (!a || !b) {
        set_error("null handle in sub");
        return nullptr;
    }
    if (a->rows != b->rows || a->cols != b->cols) {
        set_error("sub shape mismatch");
        return nullptr;
    }
    try {
        MatrixData* out = new MatrixData(a->rows, a->cols);
        parallel_binary(a->values.data(), b->values.data(), out->values.data(), out->values.size(), normalize_threads(requested_threads), sub_worker);
        return out;
    } catch (...) {
        set_error("memory allocation failed");
        return nullptr;
    }
}

MatrixData* scalar_mul_matrix(const MatrixData* a, double scalar, int requested_threads) {
    if (!a) {
        set_error("null handle in scalar_mul");
        return nullptr;
    }
    try {
        MatrixData* out = new MatrixData(a->rows, a->cols);
        parallel_scalar(a->values.data(), scalar, out->values.data(), out->values.size(), normalize_threads(requested_threads));
        return out;
    } catch (...) {
        set_error("memory allocation failed");
        return nullptr;
    }
}

MatrixData* transpose_matrix(const MatrixData* src, int requested_threads) {
    if (!src) {
        set_error("null handle in transpose");
        return nullptr;
    }
    try {
        MatrixData* out = new MatrixData(src->cols, src->rows);
        if (src->rows == 0 || src->cols == 0) return out;
        int real_threads = std::max(1, std::min<int>(normalize_threads(requested_threads), static_cast<int>(src->rows)));
        std::vector<std::thread> pool;
        pool.reserve(static_cast<std::size_t>(real_threads));
        std::int64_t chunk = (src->rows + real_threads - 1) / real_threads;
        for (int t = 0; t < real_threads; ++t) {
            std::int64_t r0 = static_cast<std::int64_t>(t) * chunk;
            std::int64_t r1 = std::min<std::int64_t>(src->rows, r0 + chunk);
            if (r0 >= r1) break;
            pool.emplace_back(transpose_worker, src->values.data(), out->values.data(), src->rows, src->cols, r0, r1);
        }
        for (auto& th : pool) th.join();
        return out;
    } catch (...) {
        set_error("memory allocation failed");
        return nullptr;
    }
}

MatrixData* matmul_matrix(const MatrixData* a, const MatrixData* b, int requested_threads, int block_size) {
    if (!a || !b) {
        set_error("null handle in matmul");
        return nullptr;
    }
    if (a->cols != b->rows) {
        set_error("matmul dimension mismatch");
        return nullptr;
    }
    if (block_size <= 0) {
        set_error("block_size must be positive");
        return nullptr;
    }
    try {
        MatrixData* out = new MatrixData(a->rows, b->cols);
        if (a->rows == 0 || a->cols == 0 || b->cols == 0) return out;
        std::vector<double> bt(static_cast<std::size_t>(b->cols * b->rows), 0.0);
        for (std::int64_t i = 0; i < b->rows; ++i) {
            const double* b_row = b->values.data() + i * b->cols;
            for (std::int64_t j = 0; j < b->cols; ++j) {
                bt[static_cast<std::size_t>(j * b->rows + i)] = b_row[j];
            }
        }
        int real_threads = std::max(1, std::min<int>(normalize_threads(requested_threads), static_cast<int>(a->rows == 0 ? 1 : a->rows)));
        std::vector<std::thread> pool;
        pool.reserve(static_cast<std::size_t>(real_threads));
        std::int64_t chunk = (a->rows + real_threads - 1) / real_threads;
        for (int t = 0; t < real_threads; ++t) {
            std::int64_t row_begin = static_cast<std::int64_t>(t) * chunk;
            std::int64_t row_end = std::min<std::int64_t>(a->rows, row_begin + chunk);
            if (row_begin >= row_end) break;
            pool.emplace_back(matmul_worker, a->values.data(), bt.data(), out->values.data(), a->cols, b->cols, row_begin, row_end, block_size);
        }
        for (auto& th : pool) th.join();
        return out;
    } catch (...) {
        set_error("memory allocation failed");
        return nullptr;
    }
}

}  // namespace

extern "C" {

FM_EXPORT const char* fm_last_error() {
    return g_last_error.c_str();
}

FM_EXPORT int fm_hardware_threads() {
    return normalize_threads(0);
}

FM_EXPORT void* fm_create_from_flat(const double* values, std::int64_t rows, std::int64_t cols) {
    if (rows < 0 || cols < 0) {
        set_error("negative matrix dimensions");
        return nullptr;
    }
    if ((rows * cols) > 0 && values == nullptr) {
        set_error("null values pointer");
        return nullptr;
    }
    try {
        MatrixData* out = new MatrixData(rows, cols);
        if (rows * cols > 0) {
            std::memcpy(out->values.data(), values, static_cast<std::size_t>(rows * cols) * sizeof(double));
        }
        return out;
    } catch (...) {
        set_error("memory allocation failed");
        return nullptr;
    }
}

FM_EXPORT void fm_free(void* handle) {
    delete as_matrix(handle);
}

FM_EXPORT std::int64_t fm_rows(void* handle) {
    MatrixData* m = as_matrix(handle);
    if (!m) {
        set_error("null handle in fm_rows");
        return -1;
    }
    return m->rows;
}

FM_EXPORT std::int64_t fm_cols(void* handle) {
    MatrixData* m = as_matrix(handle);
    if (!m) {
        set_error("null handle in fm_cols");
        return -1;
    }
    return m->cols;
}

FM_EXPORT int fm_copy_to_flat(void* handle, double* out, std::int64_t size) {
    MatrixData* m = as_matrix(handle);
    if (!m || !out) {
        set_error("null pointer in fm_copy_to_flat");
        return -1;
    }
    std::int64_t expected = m->rows * m->cols;
    if (size != expected) {
        set_error("copy buffer size mismatch");
        return -2;
    }
    if (expected > 0) {
        std::memcpy(out, m->values.data(), static_cast<std::size_t>(expected) * sizeof(double));
    }
    return 0;
}

FM_EXPORT void* fm_clone(void* handle) {
    return clone_matrix(as_matrix(handle));
}

FM_EXPORT void* fm_add(void* a, void* b, int threads) {
    return add_matrix(as_matrix(a), as_matrix(b), threads);
}

FM_EXPORT void* fm_sub(void* a, void* b, int threads) {
    return sub_matrix(as_matrix(a), as_matrix(b), threads);
}

FM_EXPORT void* fm_scalar_mul(void* a, double scalar, int threads) {
    return scalar_mul_matrix(as_matrix(a), scalar, threads);
}

FM_EXPORT void* fm_transpose(void* a, int threads) {
    return transpose_matrix(as_matrix(a), threads);
}

FM_EXPORT void* fm_matmul(void* a, void* b, int threads, int block_size) {
    return matmul_matrix(as_matrix(a), as_matrix(b), threads, block_size);
}

}
