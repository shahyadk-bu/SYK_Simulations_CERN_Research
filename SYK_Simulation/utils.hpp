#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <string>
#include <complex>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>

// We create the imaginary unit i
std::complex<double> i(0.0, 1.0);

/*
This function will ouput the wanted pauli matrix

Inputs: 
    String type: "x", "y, or "z";
Output:
    Eigen::Matrix2cd mat; which is a 2d complex matrix
*/
Eigen::Matrix2cd pauliMat(const std::string& type) {
    Eigen::Matrix2cd mat;

    if (type == "x"){
        mat << 0, 1,
               1, 0;
    }

    else if (type == "y"){
        mat << 0, -i,
               i, 0;
    }

    else if (type == "z"){
        mat << 1, 0,
               0, -1;
    }

    else {
        throw std::invalid_argument("Invalid Pauli matrix type");
    }

    return mat;
}

/*
This function will ouput the tensor product of two matricies

Inputs: 
    Eigen::MatrixXcd A; The left matrix (m x n) in the product 
    Eigen::MatrixXcd B; The right matrix (p x q) in the product
Output:
    Eigen::MatrixXcd mat; The tensor product of the two matricies (mp x nq)
*/
Eigen::MatrixXcd tensor(const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B){

    Eigen::MatrixXcd mat(A.rows() * B.rows(), A.cols() * B.cols());

    for (int j = 0; j < A.rows(); j++){
        for (int k = 0; k < A.cols(); k++){
            mat.block(j * B.rows(), k * B.cols(), B.rows(), B.cols()) = A(j,k) * B;
        }
    }

    return mat;
}

/*
This function just gives you the NxN identity 
Inputs: 
    int N; dimension you want for the identity
Output:
    Eigen::MatrixXcd I; The NxN identity 
*/
Eigen::MatrixXcd identity(int N){
    Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(N, N);
    return I;
}

/*
This function saves the vector of matricies in a .txt file with the following format

N - number of matricies
n1 m1 - number of rows and columns in first matrix
a_11_r a_11_i - real component then imaginary component of the first matrix element
a_12_r a_12_i
...
a_1m1_r a1m1_i
...
a_n1m1_r an1m1_i
n2 m2 - number of rows, cols of second matrix
...

Inputs: 
    std::vector<Eigen::MatrixXcd> ops; the vector of matricies
    std::string filename; the filename you want to save it to
Output:
    void
*/
void saveMats(const std::vector<Eigen::MatrixXcd>& ops, const std::string& filename){
    std::ofstream out(filename);

    if (!out) {
            throw std::invalid_argument("Could not open file.\n");
        }

    out << ops.size() << "\n";

    for (const auto& M : ops){
        out << M.rows() << " " << M.cols() << "\n";

        for (int r = 0; r < M.rows(); r++){
            for (int c = 0; c < M.cols(); c++){
                out << M(r,c).real() << " " << M(r,c).imag() << "\n";
            }
        }
        
    }
}

/*
This function loads the vector of matricies in a .txt file with the following format

N - number of matricies
n1 m1 - number of rows and columns in first matrix
a_11_r a_11_i - real component then imaginary component of the first matrix element
a_12_r a_12_i
...
a_1m1_r a1m1_i
...
a_n1m1_r an1m1_i
n2 m2 - number of rows, cols of second matrix
...

Inputs: 
    std::string filename; the filename of the data you want to load in 
Output:
    std::vector<Eigen::MatrixXcd> matsVector; the vector of matricies you loaded in
*/
std::vector<Eigen::MatrixXcd> loadMats(const std::string& filename){
        std::ifstream in(filename);

        if (!in) {
            throw std::invalid_argument("Could not open file.\n");
        }

        std::vector<Eigen::MatrixXcd> matsVector;

        int numMats;
        in >> numMats;

        int rows, cols;
        double re, img;

        for (int j = 0; j < numMats; j++){
            in >> rows >> cols;
            Eigen::MatrixXcd mat(rows, cols);

            for (int r = 0; r < rows; r++){
                for (int c = 0; c < cols; c++){
                    in >> re >> img;
                    mat(r,c) = std::complex<double>(re, img);
                }
            }

            matsVector.push_back(mat);
        }

    return matsVector;
}

/*
This function defines the recursion to find the Majorana Operator Representations
and saves them to a .txt file
Inputs: 
    int K; The number of fermionic Modes
Output:
    void
*/
void SYKRecursion(int K){
    std::vector<Eigen::MatrixXcd> ops;
    std::string filename = "../majorana_matricies/ops_" + std::to_string(K) + "Modes.txt";

    if (K == 1 && !std::filesystem::exists(filename)) {
        Eigen::Matrix2cd psi_1_1;
        Eigen::Matrix2cd psi_2_1;

        // Note we have mat_lowerIndex_upperIndex
        psi_1_1 = pauliMat("y");
        psi_1_1 = (1 / std::sqrt(2.0)) * psi_1_1;
        ops.push_back(psi_1_1);

        psi_2_1 = pauliMat("x");
        psi_2_1 = (1 / std::sqrt(2.0)) * psi_2_1;
        ops.push_back(psi_2_1);

        saveMats(ops, filename);
    }
    else if (!std::filesystem::exists(filename)) {

        // Load in the previous Majorana operators for K-1 Modes
        std::string previous_ops_filename = "../majorana_matricies/ops_" + std::to_string(K-1) + "Modes.txt";
        if (!std::filesystem::exists(previous_ops_filename)){
            SYKRecursion(K-1);
            std::cout << "The " + std::to_string(K-1) + " modes file did not exist so it was made.\n";
        }

        std::vector<Eigen::MatrixXcd> prev_ops = loadMats(previous_ops_filename);

        // This is the matrix used to take psi_j_K-1 to psi_j_K
        Eigen::Matrix2cd upgrade;

        upgrade << -1, 0,
                    0, 1;


        // Actually does the recursion to find the Majorana operators for K modes
        for (const auto& M : prev_ops){
            Eigen::MatrixXcd newM = tensor(M, upgrade);
            ops.push_back(newM);
        }

        // This is called a binary shift where 1 << n is 2^n since 0001 = 1, 0010 = 2, 0100 = 4, 1000 = 8, etc.
        int dim = 1 << (K-1);

        Eigen::MatrixXcd lastMats1 = tensor(identity(dim), pauliMat("x"));
        lastMats1 = (1 / std::sqrt(2.0)) * lastMats1;
        ops.push_back(lastMats1);

        Eigen::MatrixXcd lastMats2 = tensor(identity(dim), pauliMat("y"));
        lastMats2 = (1 / std::sqrt(2.0)) * lastMats2;
        ops.push_back(lastMats2);

        saveMats(ops, filename);
    } 
    else {
        std::cout << "The operators for " + std::to_string(K) + " fermionic modes are already saved. \n";
    }
}

#endif