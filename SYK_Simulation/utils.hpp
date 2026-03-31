#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <Eigen/Dense>
#include <string>
#include <complex>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>

Eigen::Matrix2cd pauliMat(const std::string& type);
Eigen::MatrixXcd tensor(const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B);
Eigen::MatrixXcd identity(int N);
void saveMats(const std::vector<Eigen::MatrixXcd>& ops, const std::string& filename);
std::vector<Eigen::MatrixXcd> loadMats(const std::string& filename);
void SYKRecursion(int K);
void dataToCSV(const Eigen::VectorXd& data, const std::string& filename, const std::string& funcName = "");
void dataToCSV(const std::vector<double>& data, const std::string& filename, const std::string& funcName = "");

#endif