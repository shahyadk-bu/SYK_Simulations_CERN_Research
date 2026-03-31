#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <complex>
#include <stdexcept>
#include <cmath>

#include "utils.hpp"
#include "hamiltonian.hpp"

int main() {
    Hamiltonian ham = Hamiltonian(6, 3, 10);
    ham.saveHamObj();
    ham.saveSpectralData();
    return 0;
}
