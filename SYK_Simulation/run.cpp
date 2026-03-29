#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <complex>
#include <stdexcept>
#include <cmath>

#include "utils.hpp"
#include "hamiltonian.hpp"

bool approxEqual(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

bool approxEqualComplex(std::complex<double> a, std::complex<double> b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

bool matricesApproxEqual(const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B, double tol = 1e-10) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        return false;
    }

    for (int r = 0; r < A.rows(); r++) {
        for (int c = 0; c < A.cols(); c++) {
            if (!approxEqualComplex(A(r, c), B(r, c), tol)) {
                return false;
            }
        }
    }

    return true;
}

bool vectorsApproxEqual(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2, double tol = 1e-10) {
    if (v1.size() != v2.size()) {
        return false;
    }

    for (int j = 0; j < v1.size(); j++) {
        if (!approxEqual(v1(j), v2(j), tol)) {
            return false;
        }
    }

    return true;
}

bool isHermitian(const Eigen::MatrixXcd& H, double tol = 1e-10) {
    if (H.rows() != H.cols()) {
        return false;
    }

    for (int r = 0; r < H.rows(); r++) {
        for (int c = 0; c < H.cols(); c++) {
            if (!approxEqualComplex(H(r, c), std::conj(H(c, r)), tol)) {
                return false;
            }
        }
    }

    return true;
}

void printTestResult(const std::string& testName, bool passed) {
    if (passed) {
        std::cout << "[PASS] " << testName << "\n";
    } else {
        std::cout << "[FAIL] " << testName << "\n";
    }
}

int main() {
    try {
        int K = 4;
        double J = 1.0;
        int seed = 42;

        std::cout << "Creating Hamiltonian...\n";
        Hamiltonian H(K, J, seed);

        // -----------------------------
        // Test 1: getters
        // -----------------------------
        printTestResult("getK()", H.getK() == K);
        printTestResult("getJ()", approxEqual(H.getJ(), J));
        printTestResult("getSeed()", H.getSeed() == seed);

        // -----------------------------
        // Test 2: matrix dimensions
        // Hilbert space dimension should be 2^K
        // -----------------------------
        const Eigen::MatrixXcd& M = H.getMatrix();
        int expectedDim = 1 << K;

        printTestResult("Matrix rows correct", M.rows() == expectedDim);
        printTestResult("Matrix cols correct", M.cols() == expectedDim);

        // -----------------------------
        // Test 3: Hermitian check
        // -----------------------------
        printTestResult("Hamiltonian is Hermitian", isHermitian(M));

        // -----------------------------
        // Test 4: eigenvalue count
        // -----------------------------
        const Eigen::VectorXd& evals = H.getEigenValues();
        printTestResult("Eigenvalue count correct", evals.size() == expectedDim);

        // -----------------------------
        // Test 5: eigenvalues sorted nondecreasing
        // SelfAdjointEigenSolver should do this
        // -----------------------------
        bool sorted = true;
        for (int j = 0; j < evals.size() - 1; j++) {
            if (evals(j) > evals(j + 1)) {
                sorted = false;
                break;
            }
        }
        printTestResult("Eigenvalues sorted", sorted);

        // -----------------------------
        // Test 6: level spacings count
        // -----------------------------
        std::vector<double> spacings = H.findLevelSpacings();
        printTestResult("Level spacings count correct", static_cast<int>(spacings.size()) == expectedDim - 1);

        // -----------------------------
        // Test 7: level spacings nonnegative
        // -----------------------------
        bool nonnegativeSpacings = true;
        for (double s : spacings) {
            if (s < -1e-12) {
                nonnegativeSpacings = false;
                break;
            }
        }
        printTestResult("Level spacings nonnegative", nonnegativeSpacings);

        // -----------------------------
        // Test 8: eigenvectors dimension
        // -----------------------------
        Eigen::MatrixXcd evecs = H.findEigenVectors();
        printTestResult("Eigenvector matrix rows correct", evecs.rows() == expectedDim);
        printTestResult("Eigenvector matrix cols correct", evecs.cols() == expectedDim);

        // -----------------------------
        // Test 9: save and reload
        // -----------------------------
        H.saveHamObj();

        std::string filename =
            "../hamiltonianObjects/Ham_K" +
            std::to_string(K) +
            "_J" + std::to_string(J) +
            "_seed" + std::to_string(seed) +
            ".txt";

        Hamiltonian H_loaded(filename);

        printTestResult("Loaded K matches", H_loaded.getK() == H.getK());
        printTestResult("Loaded J matches", approxEqual(H_loaded.getJ(), H.getJ()));
        printTestResult("Loaded seed matches", H_loaded.getSeed() == H.getSeed());
        printTestResult("Loaded matrix matches", matricesApproxEqual(H_loaded.getMatrix(), H.getMatrix()));
        printTestResult("Loaded eigenvalues match", vectorsApproxEqual(H_loaded.getEigenValues(), H.getEigenValues()));

        // -----------------------------
        // Optional: print some data
        // -----------------------------
        std::cout << "\nFirst few eigenvalues:\n";
        for (int j = 0; j < std::min<int>(5, evals.size()); j++) {
            std::cout << evals(j) << "\n";
        }

        std::cout << "\nFirst few level spacings:\n";
        for (int j = 0; j < std::min<int>(5, static_cast<int>(spacings.size())); j++) {
            std::cout << spacings[j] << "\n";
        }

        std::cout << "\nAll main tests finished.\n";
    }
    catch (const std::exception& e) {
        std::cout << "[EXCEPTION] " << e.what() << "\n";
    }

    return 0;
}
