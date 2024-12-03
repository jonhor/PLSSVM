
#ifndef PLSSVM_BACKENDS_SYCL_LINALG_SVD2_HPP_
#define PLSSVM_BACKENDS_SYCL_LINALG_SVD2_HPP_

#include "plssvm/backends/SYCL/linalg/constants.hpp"
#include "plssvm/backends/SYCL/linalg/decompositions/qr.hpp"
#include "plssvm/backends/SYCL/linalg/norms.hpp"
#include "plssvm/backends/SYCL/linalg/transforms.hpp"
#include "plssvm/detail/assert.hpp"

/**
 * Implements a SVD computed on CPU.
 *
 * This algorithm is not GPU friendly without some modifications.
 * TODO write more here
 */

namespace plssvm::sycl::linalg {

inline void clip_small_value(real_type &v) {
    if (std::abs(v) < eps) {
        v = 0;
    }
}

struct givens_rotation {
    real_type c, s;

    inline void apply_left(matrix_view<matrix_type::general> &A, std::size_t p, std::size_t q) {
        const auto n = A.n_cols;

        // Copy rows p and q to save their original values.
        std::vector<real_type> x(n);
        std::vector<real_type> y(n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = A(p, i);
            y[i] = A(q, i);
        }

        // Update rows p and q by applying the rotation.
        for (std::size_t i = 0; i < n; ++i) {
            A(p, i) = c * x[i] + s * y[i];
            clip_small_value(A(p, i));
            A(q, i) = -s * x[i] + c * y[i];
            clip_small_value(A(q, i));
        }
    }

    inline void apply_right(matrix_view<matrix_type::general> &A, std::size_t p, std::size_t q) {
        const auto n = A.n_rows;

        // Copy columns p and q to save their original values.
        std::vector<real_type> x(n);
        std::vector<real_type> y(n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = A(i, p);
            y[i] = A(i, q);
        }

        // Update columns p and q by applying the rotation.
        for (std::size_t i = 0; i < n; ++i) {
            A(i, p) = c * x[i] - s * y[i];
            clip_small_value(A(i, p));
            A(i, q) = s * x[i] + c * y[i];
            clip_small_value(A(i, q));
        }
    }

    inline void transpose() {
        s = -s;
    }
};

class svd_2x2 {
  public:
    svd_2x2(const matrix_view<matrix_type::general> &A) :
        A_(A),
        M_vec_(4),
        M_(M_vec_.data(), 2, 2) { }

    inline std::pair<givens_rotation, givens_rotation> operator()(std::size_t p, std::size_t q) {
        M_(0, 0) = A_(p, p);
        M_(0, 1) = A_(p, q);
        M_(1, 0) = A_(q, p);
        M_(1, 1) = A_(q, q);

        auto first_rotation = symmetric_givens_rotation();
        first_rotation.apply_left(M_, 0, 1);

        const auto jacobi_rotation_right = jacobi_rotation();
        const givens_rotation jacobi_rotation_left{
            first_rotation.c * jacobi_rotation_right.c + first_rotation.s * jacobi_rotation_right.s,
            first_rotation.s * jacobi_rotation_right.c - first_rotation.c * jacobi_rotation_right.s
        };

        return std::pair{ jacobi_rotation_left, jacobi_rotation_right };
    }

  private:
    inline givens_rotation symmetric_givens_rotation() {
        const real_type t = M_(0, 0) + M_(1, 1);
        const real_type d = M_(1, 0) - M_(0, 1);

        real_type s, c;
        if (std::abs(d) < eps) {
            s = real_type{ 0 };
            c = real_type{ 1 };
        } else {
            const real_type u = t / d;
            const real_type tmp = std::sqrt(real_type{ 1 } + u * u);

            s = real_type{ 1 } / tmp;
            c = u / tmp;
        }

        return { c, s };
    }

    inline givens_rotation jacobi_rotation() {
        const real_type x = M_(0, 0), y = M_(0, 1), z = M_(1, 1);

        const auto deno = real_type{ 2 } * std::abs(y);

        real_type _c, s;
        if (deno < eps) {
            _c = real_type{ 1 };
            s = real_type{ 0 };
        } else {
            const real_type tau = (x - z) / deno;
            const real_type w = std::sqrt(tau * tau + real_type{ 1 });

            real_type t;
            if (tau > 0) {
                t = real_type{ 1 } / (tau + w);
            } else {
                t = real_type{ 1 } / (tau - w);
            }

            const real_type n = real_type{ 1 } / std::sqrt(t * t + real_type{ 1 });

            real_type t_sign = t > real_type{ 0 } ? real_type{ 1 } : real_type{ -1 };

            s = -t_sign * (y / std::abs(y)) * std::abs(t) * n;
            _c = n;
        }

        return { _c, s };
    }

    const matrix_view<matrix_type::general> &A_;
    std::vector<real_type> M_vec_;
    matrix_view<matrix_type::general> M_;
};

/**
 *  Constructs a singular value decomposition of A, such that A = U * S * VT, where
 *  - S is a diagonal matrix that contains the singular values of A
 *  - U is
 *  - VT is
 *  The idea is to apply a sequence of linear transformations to make S diagonal and capture
 *  the impact of those transformations by applying them to U and VT where appropriate.
 *
 *  This algorithm is based on the Jacobi SVD in Handbook of Linear Algebra chapter xyz..
 *
 *  The SVD algorithm iterates over pairs of indices, computing a 2x2 SVD of the sub-matrix obtained
 *  from the permutations of those indices.
 *  This procedure will iteratively reduce the impact of any off-diagonal element until we meet the
 *  convergence criteria where the norm of those elements is small enough to consider S a diagonal matrix.
 *
 *  The main work is therefore done in computing the SVD of those 2x2 sub-matrices.
 *  This is done in the following steps:
 *  1. Apply a Givens rotation such that the sub-matrix becomes symmetric.
 *  2. Apply a two-sided Jacobi rotation to eliminate both off-diagonal elements.
 *
 *  It additionally uses a QR decomposition as a preconditioner for A,
 *  enabling us to only care about the upper square part of the matrix when computing the SVD.
 *  This is helpful because we will be working with tall, skinny matrices (n >> m) obtained from
 *  Nyström approximations in the context of the Randomly Pivoted Cholesky algorithm.
 *
 * TODO because this is called more than once U, V and B should be allocated once and then reused
 * it is not called more than once, the svd has to be only calculated once for the preconditioner!!
 */
using svd_return_type = std::tuple<
    managed_matrix_view<matrix_type::general>,
    managed_matrix_view<matrix_type::general>,  // TODO introduce a diagonal matrix type (also useful for jacobi preconditioner)
    managed_matrix_view<matrix_type::general>>;

inline svd_return_type svd(::sycl::queue &queue, const matrix_view<matrix_type::general> &A, unsigned int max_iterations = 1000) {
    PLSSVM_ASSERT(A.n_rows >= A.n_cols, "SVD is currently optimized for tall matrices");

    const auto M = A.n_rows;
    const auto N = A.n_cols;

    auto U = detail::utility::identity(queue, M, N);
    auto V = detail::utility::identity(queue, N, N);

    // TODO implement QR preconditioning later!

    auto B = detail::utility::copy(queue, A);

    svd_2x2 compute_2x2_svd{ B };

    const auto norm_squared = frobenius(B, false);
    const auto epsilon_squared = eps * eps;

    bool converged = false;
    unsigned int n_iterations = 0;
    while (!converged && n_iterations++ < max_iterations) {
        real_type s = real_type{ 0 };

        // Iterate over all index pairs in the upper triangular, excluding the diagonal.
        for (std::size_t i = 0; i < N - 1; ++i) {
            for (std::size_t j = i + 1; j < N; ++j) {
                // Update off diagonal norm
                real_type s_i = B(i, j) * B(i, j) + B(j, i) * B(j, i);
                s += s_i;

                // Skip if 2x2 is already considered diagonal.
                // TODO check if this is right
                if (s_i < epsilon_squared) {
                    continue;
                }

                // Apply a two-sided Jacobi rotation to reduce off diagonal elements.
                auto [jacobi_rotation_left, jacobi_rotation_right] = compute_2x2_svd(i, j);
                jacobi_rotation_left.apply_left(B, i, j);
                jacobi_rotation_right.apply_right(B, i, j);

                // Update U and V.
                jacobi_rotation_left.transpose();
                jacobi_rotation_left.apply_right(U, i, j);
                jacobi_rotation_right.apply_right(V, i, j);
            }
        }

        converged = s <= epsilon_squared * norm_squared;
    }
    fmt::println("computed SVD after {} iterations", n_iterations);

    return std::make_tuple(std::move(U), std::move(B), std::move(V));
}
};  // namespace plssvm::sycl::linalg

#endif  // PLSSVM_BACKENDS_SYCL_LINALG_SVD2_HPP_
