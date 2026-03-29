#ifndef HAMILTONIAN_HPP
#define HAMILTONIAN_HPP

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <complex>
#include <random>
#include <string>

class Hamiltonian {

    private:
        int K_; // Fermionic Modes, recall the number of majorana matrices N = 2K
        double J_; // Coupling constant used to modify the variance of the connections J_ijkl are pulled from
        Eigen::MatrixXcd hamMat; // The hamiltonian matrix 
        std::mt19937 gen_; // Random number engine
        int seed_; // The seed used to generate random numbers
        Eigen::VectorXd e_vals; // The eigenvalues in an Eigen::vector sorted in increasing order

        double sampleCon();
        void createMatrix();
        void loadHamObj(const std::string& filename);

        void findEigenValues();

    public:
        Hamiltonian(int K, double J, int seed);
        Hamiltonian(const std::string& filename);

        void printHam() const;

        int getK() const;
        double getJ() const;
        int getSeed() const;
        const Eigen::MatrixXcd& getMatrix() const;
        const Eigen::VectorXd& getEigenValues() const;

        void saveHamObj() const;

        Eigen::MatrixXcd findEigenVectors() const;
        std::vector<double> findLevelSpacings() const;
        

};

#endif
