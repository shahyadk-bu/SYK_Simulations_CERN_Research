#include "hamiltonian.hpp"
#include "utils.hpp"

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <string>
#include <complex>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <random>


// We create the imaginary unit i
const std::complex<double> i(0.0, 1.0);

// Constructor used to create a new Hamiltonian object
Hamiltonian::Hamiltonian(int K, double J, int seed)
    : K_(K), J_(J), gen_(seed), seed_(seed)
{
    createMatrix();
    findEigenValues();
}

// Constructor used to load in a saved Hamiltonian object
Hamiltonian::Hamiltonian(const std::string& filename)
    : K_(0), J_(0.0), gen_(0), seed_(0)
{
    loadHamObj(filename);
    gen_ = std::mt19937(seed_);
}

/*
This function prints the systems hamiltonian.

Inputs: 
    void
Output:
    void
*/
void Hamiltonian::printHam() const {
    std::cout << hamMat << "\n";
}

/*
This function fetches K

Inputs: 
    void
Output:
    int K_; this is the number of fermionic modes
*/
int Hamiltonian::getK() const {
    return K_;
}

/*
This function fetches J

Inputs: 
    void
Output:
    double J_; this is the connections constant
*/
double Hamiltonian::getJ() const {
    return J_;
}

/*
This function fetches the seed.

Inputs: 
    void
Output:
    int seed; this is the seed used for generating the random numbers
*/
int Hamiltonian::getSeed() const {
    return seed_;
}

/*
This function fetches the eigen values vector.

Inputs: 
    void
Output:
    Eigen::VectorXd e_vals; these are the hamiltonians eigen values.
*/
const Eigen::VectorXd& Hamiltonian::getEigenValues() const {
   return e_vals; 
}

/*
This function fetches the hamiltonian matrix

Inputs: 
    void
Output:
    Eigen::MatrixXcd hamMat; The matrix form of the hamiltonian
*/
const Eigen::MatrixXcd& Hamiltonian::getMatrix() const {
    return hamMat;
}

/*
This function samples one connection J_ijkl from the following distribution: Normal(0, 6J^2 / N^3)

Inputs: 
    void
Output:
    int K_; this is the number of fermionic modes
*/
double Hamiltonian::sampleCon() {
    double var = (6.0 * J_ * J_) / std::pow((2.0*K_), 3.0); // Remember to always use doubles for Pow() function

    std::normal_distribution<double> dist(0, std::sqrt(var)); // The package uses dist(mean, std)

    return dist(gen_);
}

/*
This function creates the hamiltonian matrix for our SYK Model.
Specifically it assigns a matrix to the "hamMat" variable we initalize.

Inputs: 
    void
Output:
    void
*/
void Hamiltonian::createMatrix(){
    std::string filename = "../majorana_matricies/ops_" + std::to_string(K_) + "Modes.txt";

    // Make the Majorana Operators if they dont exist yet
    if (!std::filesystem::exists(filename)){
        SYKRecursion(K_);
    }

    std::vector<Eigen::MatrixXcd> Ops = loadMats(filename); // Our Majorana Operators

    if (Ops.empty()) {
        throw std::runtime_error("No operators loaded from file.");
    }
    else if (Ops.size() != static_cast<size_t>(2 * K_)) {
        throw std::runtime_error("Incorrect number of operators loaded from this file.");  
    }

    // Initalize our Hamiltonian
    int dim = Ops[0].rows();
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(dim,dim);

    // Create the Hamiltonian

    for (int j = 0; j < 2*K_; j++) {
        for (int k = j+1; k < 2*K_; k++) {
            for (int l = k+1; l < 2*K_; l++) {
                for (int w = l+1; w < 2*K_; w++) {
                    H += sampleCon() * Ops[j] * Ops[k] * Ops[l] * Ops[w];
                }
            }
        }
    }

    hamMat = H;
} 

/*
This function find the hamiltonians eigen values and fills e_vals.

Inputs: 
    void
Output:
    void
*/
void Hamiltonian::findEigenValues() {
    if (e_vals.size() > 0) {
        std::cout << "The eigen values vector is non-empty so no computations were done";
    }
    else {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(hamMat);

        if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigenvalue computation failed.");
        }

        e_vals = solver.eigenvalues();
    }
}

/*
This function find the hamiltonians eigen vectors.

Inputs: 
    void
Output:
    Eigen::VectorXd solver.eigenvalues(); The eigenvectors of our hamiltonian matrix with each column being an egienvector.
*/
Eigen::MatrixXcd Hamiltonian::findEigenVectors() const {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(hamMat);

    if (solver.info() != Eigen::Success) {
    throw std::runtime_error("Eigenvector computation failed.");
    }

    return solver.eigenvectors();
}

/*
This function finds and returns the eigenvalue level spacings.

Inputs: 
    void
Output:
    std::vector<double> spacings; A vector of the level spacings.
*/
std::vector<double> Hamiltonian::findLevelSpacings() const {
    if (e_vals.size() < 2) {
        throw std::logic_error("There are not enough eigen values to find spacings");
    }

    std::vector<double> spacings;

    for (int j = 0; j < e_vals.size()-1; j++) {
        spacings.push_back(e_vals(j+1) - e_vals(j));
    }

    return spacings;
}

/*
This function saves the hamiltonian matrix to a .txt file with the following format

K - number of fermionic modes
J - Coupling constant
S - seed
n1 m1 - number of rows and columns
a_11_r a_11_i - real component then imaginary component of the first matrix element
a_12_r a_12_i
...
a_1m1_r a1m1_i
...
a_n1m1_r an1m1_i
e_1 - first eigenvalue
e_2 - second eigenvalue
...
e_n - last eigenvalue

Inputs: 
    void
Output:
    void
*/
void Hamiltonian::saveHamObj() const {
    std::string filename = "../hamiltonianObjects/Ham_K" + std::to_string(K_) 
    + "_J" + std::to_string(J_) + "_seed" + std::to_string(seed_) + ".txt";

    std::ofstream out(filename);

    if (!out) {
            throw std::invalid_argument("Could not open file during saveHamObj() function.\n");
        }

    out << std::scientific << std::setprecision(17);

    out << K_ << "\n";
    out << J_ << "\n";
    out << seed_ << "\n";
    out << hamMat.rows() << " " << hamMat.cols() << "\n";

    for (int r = 0; r < hamMat.rows(); r++){
        for (int c = 0; c < hamMat.cols(); c++){
            out << hamMat(r,c).real() << " " << hamMat(r,c).imag() << "\n";
        }
    }

    for (int j = 0; j < e_vals.size(); j++){
        out << e_vals(j) << "\n";
    }
}

/*
This function loads in the Hamiltonian Object from the file provided

K - number of fermionic modes
J - Coupling constant
S - seed
n1 m1 - number of rows and columns
a_11_r a_11_i - real component then imaginary component of the first matrix element
a_12_r a_12_i
...
a_1m1_r a1m1_i
...
a_n1m1_r an1m1_i
e_1 - first eigenvalue
e_2 - second eigenvalue
...
e_n - last eigenvalue

Inputs: 
    void
Output:
    void
*/
void Hamiltonian::loadHamObj(const std::string& filename) {
    std::ifstream in(filename);

    if (!in) {
        throw std::invalid_argument("Could not open file during loadHamObj() function.\n");
    }

    int rows, cols;
    double re, img;

    in >> K_;
    in >> J_;
    in >> seed_;
    in >> rows >> cols;

    Eigen::MatrixXcd mat(rows, cols);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++){
            in >> re >> img;
            mat(r,c) = std::complex<double>(re,img);
        }
    }

    double eval;
    Eigen::VectorXd evals(rows);

    for (int j = 0; j < rows; j++) {
        in >> eval;
        evals(j) = eval;
    }

    hamMat = mat;
    e_vals = evals;

}

/*
This function saves the eigen values and the eigen repulsion data into CSV files

Inputs: 
    void
Output:
    void
*/
void Hamiltonian::saveSpectralData() const {
    std::vector<double> spacings = findLevelSpacings();

    std::string evalsFilename = "../spectralData/eigenValueData/evals_K" + std::to_string(K_) 
    + "_J" + std::to_string(J_) + "_seed" + std::to_string(seed_) + ".csv";

    std::string eRepulsionFilename = "../spectralData/eigenRepulsionData/erep_K" + std::to_string(K_) 
    + "_J" + std::to_string(J_) + "_seed" + std::to_string(seed_) + ".csv";

    dataToCSV(e_vals, evalsFilename, "saveSpectralData() evals part");
    dataToCSV(spacings, eRepulsionFilename, "saveSpectralData() spacings part");
}