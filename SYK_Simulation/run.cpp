#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include "utils.hpp"

int main() {
    int K = 5;

    SYKRecursion(K);

    std::string filename = "../majorana_matricies/ops_" + std::to_string(K) + "Modes.txt";
    std::vector<Eigen::MatrixXcd> ops = loadMats(filename);

    int dim = 1 << K;
    Eigen::MatrixXcd id = identity(dim);
    Eigen::MatrixXcd zero = Eigen::MatrixXcd::Zero(dim, dim);

    double tol = 1e-6;
    bool passed = true;

    for (size_t i = 0; i < ops.size(); i++) {
        for (size_t j = 0; j < ops.size(); j++) {
            Eigen::MatrixXcd anticom = ops[i] * ops[j] + ops[j] * ops[i];
            Eigen::MatrixXcd target = (i == j) ? id : zero;

            if (!anticom.isApprox(target, tol)) {
                std::cout << "Error for i = " << i << ", j = " << j << "\n";
                passed = false;
            }
        }
    }

    if (passed) {
        std::cout << "All Clifford algebra checks passed.\n";
    }

    return 0;
}
