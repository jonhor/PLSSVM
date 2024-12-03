#ifndef PLSSVM_BACKENDS_SYCL_LINALG_QR_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_QR_HPP_

#include "plssvm/backends/SYCL/detail/matrix_view.hpp"
#include "plssvm/detail/assert.hpp"

namespace plssvm::sycl::linalg {

class householder {
  public:
    householder(const matrix_view<matrix_type::general> &R) :
        R_(R),
        N_(R.n_rows) {
        v_ = std::vector<real_type>(R.n_rows);
    }

    std::vector<real_type> &operator()(std::size_t k) {
        // x = R[k:, k]
        auto n = N_ - k;

        // alpha = -sgn(a[0]) * np.linalg.norm(a)
        auto sign = R_(k, k) >= 0 ? real_type{ 1 } : real_type{ -1 };
        real_type norm{ 0 };
        for (std::size_t i = k; i < N_; ++i) {
            norm += R_(i, k) * R_(i, k);
        }
        norm = std::sqrt(norm);
        auto alpha = -sign * norm;

        v_.resize(n);
        v_[0] = alpha - R_(k, k);
        for (std::size_t i = k + 1; i < N_; ++i) {
            auto j = i - k;
            v_[j] = -R_(i, k);
        }

        norm = real_type{ 0 };
        for (auto e : v_) {
            norm += e * e;
        }
        norm = std::sqrt(norm);

        for (auto &e : v_) {
            e = e / norm;
        }

        return v_;
    }

  private:
    std::vector<real_type> v_;
    const matrix_view<matrix_type::general> &R_;
    const std::size_t N_;
};

/**
 *
 *
 * Computes a QR decomposition of A, such that A = Q * R.
 * This is implemented by using Householder Reflections, see Book
 * TODO is this a thin decomposition?
 * @param A TODO
 * @param Q TODO
 * @param R TODO
 *
 * this transforms A into R inplace, maybe write a wrapper that copies A into Q beforehand?
 */
using qr_return_type = std::pair<
    managed_matrix_view<matrix_type::general>,
    managed_matrix_view<matrix_type::general>>;

/*
 * Reduced QR factorization A = Q * R
 */
inline managed_matrix_view<matrix_type::general> qr(::sycl::queue &queue, const matrix_view<matrix_type::general> &A) {
    PLSSVM_ASSERT(A.n_rows >= A.n_cols, "QR decomposition is only implemented for matrices, where n_rows >= n_cols");

    const auto N = A.n_rows;
    const auto M = A.n_cols;

    // auto R = detail::utility::zeros<matrix_type::general>(queue, N, N, PADDING_SIZE);
    // auto Q = detail::utility::zeros<matrix_type::general>(queue, N, M, PADDING_SIZE);
    auto Q = detail::utility::identity(queue, N, M, PADDING_SIZE);
    auto R = detail::utility::copy(queue, A);

    std::vector<real_type> w(M);
    auto compute_householder = householder{ R };
    for (std::size_t k = 0; k < M; ++k) {
        auto v = compute_householder(k);

        if (k <= 50) {
            fmt::print("householder: ");
            for (auto e : v) {
                fmt::print("{:.4f} ", e);
            }
            fmt::println("");
        }

        auto n = N - k;
        auto m = M - k;

        // w = v @ R[k:, k:]
        for (std::size_t i = 0; i < m; ++i) {
            real_type sum{ 0 };
            for (std::size_t j = 0; j < n; ++j) {
                sum += v[j] * R(j + k, i + k);
            }
            w[i] = sum;
        }

        // update R = R - 2 * outer(v, w)
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
                auto ri = i + k;
                auto rj = j + k;
                R(ri, rj) -= real_type{ 2 } * (v[i] * w[j]);

                // clip small values
                if (std::abs(R(ri, rj)) < eps) {
                    R(ri, rj) = real_type{ 0 };
                }
            }
        }

        // Q shape = n x m (5 x 3)

        // w = Q[k:,k:] @ v
        for (std::size_t i = 0; i < m; ++i) {
            real_type sum{ 0 };
            for (std::size_t j = 0; j < n; ++j) {
                sum += Q(i + k, j + k) * v[j];
            }
            w[i] = sum;
        }

        // update Q = Q - 2 * outer(w, v)
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
                auto qi = i + k;
                auto qj = j + k;
                Q(qi, qj) -= real_type{ 2 } * (w[j] * v[i]);

                // clip small values
                if (std::abs(Q(qi, qj)) < eps) {
                    Q(qi, qj) = real_type{ 0 };
                }
            }
        }
    }

    // TODO extract upper triangular part of R to get the reduced form NxN
    // return std::make_pair(std::move(Q), std::move(R));
    return Q;
}

}  // namespace plssvm::sycl::linalg

#endif  // PLSSVM_BACKENDS_SYCL_LINALG_QR_HPP_
