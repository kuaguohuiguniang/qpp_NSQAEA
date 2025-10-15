#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include "qpp/qpp.hpp"

using namespace qpp;
using namespace Eigen;
using namespace std;

// Define constants and parameters
const double gamma_1 = 1.4;   
const double mu = 0.001;    
const double lambda = 0.0001; 
const double R = 287.0;      
const double kappa = 0.01;   

double L; 
double T; 
int m;    
int w = 3;    
int n = static_cast<int> (pow(2, w));   
int k = 2; 
int N = static_cast<int> (pow(n, k-1)); 
VectorXd t(n); 

// Function to compute the spatial derivative of the flux vector F
Eigen::Vector3d dFlux(const Eigen::MatrixXd& U, int j, double dx) {
    Eigen::Vector3d F;
    F(0) = (U(j + 1, 1) - U(j - 1, 1))/(2 * dx); 

    F(1) = (3 - gamma_1) * (U(j + 1, 1) * U(j + 1, 1)/(2 * U(j + 1, 0)) - U(j - 1, 1) * U(j - 1, 1)/(2 * U(j - 1, 0)))/(2 * dx)
    + (gamma_1 - 1) * (U(j + 1, 2) - U(j - 1, 2))/(2 * dx)
    - (2 * mu + lambda) * (U(j + 1, 1)/U(j + 1, 0) + U(j - 1, 1)/U(j - 1, 0) - 2 * U(j, 1)/U(j, 0))/(dx * dx); 

    F(2) = gamma_1 * (U(j + 1, 1) * U(j + 1, 2)/U(j + 1, 0) - U(j - 1, 1) * U(j - 1, 2)/U(j - 1, 0))/(2 * dx)
    + (2 - gamma_1) * (U(j + 1, 1) * U(j + 1, 1) * U(j + 1, 1)/(2 * U(j + 1, 0) * U(j + 1, 0)) - U(j - 1, 1) * U(j - 1, 1) * U(j - 1, 1)/(2 * U(j - 1, 0) * U(j - 1, 0)))/(2 * dx)
    - (2 * mu + lambda) * ( U(j + 1, 1) * U(j + 1, 1)/(U(j + 1, 0) * U(j + 1, 0)) + U(j - 1, 1) * U(j - 1, 1)/(U(j - 1, 0) * U(j - 1, 0)) - 2 * U(j, 1) * U(j, 1)/(U(j, 0) * U(j, 0)))/(2 * dx * dx) 
    + kappa * (gamma_1 - 1) *((2 * U(j + 1, 0) * U(j + 1, 2) - U(j + 1, 1) * U(j + 1, 1))/(U(j + 1, 0) * U(j + 1, 0)) + (2 * U(j - 1, 0) * U(j - 1, 2) - U(j - 1, 1) * U(j - 1, 1))/(U(j - 1, 0) * U(j - 1, 0)) - 2 * (2 * U(j, 0) * U(j, 2) - U(j, 1) * U(j, 1))/(U(j, 0) * U(j, 0)))/(2 * R * dx * dx); 
    return F;
}

// Function to perform QAEA
double QAEA(const Eigen::VectorXd& f) {

//renormalization
    Eigen::VectorXd g(N);
    double f_min = f.minCoeff();
    double f_max = f.maxCoeff();
    for (int i = 0; i < f.size(); ++i) {
        g(i) = (f(i) - f_min) / (f_max - f_min); 
    }
    
idx n1 = w * (k - 1);
idx ntot = 2 * n1 + 1;
 
//initial state: all qubits are in zero state
ket psi = mket(std::vector<idx>(ntot, 0)); 

//numbering the qubits
vector<idx> register1(n1);
std::iota(register1.begin(), register1.end(), 0);
vector<idx> register2(n1);
std::iota(register2.begin(), register2.end(), n1);
vector<idx> register2t(n1 + 1);
std::iota(register2t.begin(), register2t.end(), n1);

//act A on initial state
psi = applyQFT(psi, register2);     
cmat U = MatrixXd::Zero(2 * N, 2 * N);
for (idx i = 0; i < N; ++i){
U(2 * i, 2 * i) = sqrt(g(i));
U(2 * i + 1, 2 * i) = sqrt(1 - g(i));
}

psi = apply(psi, U, register2t);

//define Q
ket psi2(2 * N);
bra psibar2(2 * N);
for (idx i = 0; i< 2 * N; ++i){
psi2(i) = psi(i);
}
psibar2 = psi2.conjugate();

//apply QFT on register1
psi = applyQFT(psi, register1);

//apply Lambda(Q) on psi
cmat rhopsi = MatrixXd::Zero(2 * N, 2 * N);
for (idx i = 0; i < 2 * N; ++i){
    for (idx j = 0; j < 2 * N; ++j){
rhopsi(i, j) = psi2(i) * psibar2(j);
    }
}
cmat Identity = MatrixXd::Zero(2 * N, 2 * N);
for (idx i = 0; i < 2 * N; ++i){
    Identity(i, i) = 1;
}
cmat Sign = MatrixXd::Zero(2 * N, 2 * N);
for (idx i = 0; i< 2 * N; ++i){
    Sign(i, i) = pow(- 1, i);
}
cmat Q = MatrixXd::Zero(2 * N, 2 * N);
cmat Qi = MatrixXd::Zero(2 * N, 2 * N);
Q = (Identity - 2 * rhopsi) * Sign;
Qi = Sign * (Identity - 2 * rhopsi);

cmat P(N, N);
ket Psi = VectorXd::Zero(2 * N * N);

for (idx i = 0; i < N; ++i){
P = MatrixXd::Zero(N, N);
P(i, i) = 1;
Qi = Qi * Q;
ket Ppsi = apply(psi, P, register1);
ket QPpsi = apply(Ppsi, Qi, register2t);
Psi += QPpsi;
}
psi = Psi;

//apply TFQ on register1
psi = applyTFQ(psi, register1);

//measure register1
auto measured = measure_seq(psi, {register1});
auto res = std::get<RES>(measured);

idx y = multiidx2n(res, std::vector<idx>(n1, 2));
double average_g = pow(sin(M_PI * y / N), 2); 

//get estimation of average
return average_g * (f_max - f_min) + f_min; 
}

int main() {
    // User inputs
    std::cout << "Enter the length L: ";
    std::cin >> L;
    std::cout << "Enter the time interval T: ";
    std::cin >> T;
    std::cout << "Enter the number of segments m: ";
    std::cin >> m;

double dt = T / n;
double dx = L / m;

    Eigen::MatrixXd U(m + 1, 3); 
    VectorXd rho(m+1); 
    VectorXd v(m+1); 
    VectorXd T(m+1); 

    // Set initial conditions
    std::cout << "Enter initial conditions (rho, v, T) for each j from 0 to " << m << ":\n";
    for (int j = 0; j <= m; ++j) {
        double rho_init, v_init, T_init;
        cout << "j = " << j << ": ";
        cin >> rho_init >> v_init >> T_init;
        U(j, 0) = rho_init;                            
        U(j, 1) = rho_init * v_init;                 
        U(j, 2) = rho_init * (R * T_init/(gamma_1 - 1) + v_init * v_init/2);       
    }

 // Main loop for time evolution

 // Create matrics to hold the polynomial approximations
            MatrixXd u0(m + 1, N+1);
            MatrixXd u1(m + 1, N+1);
            MatrixXd u2(m + 1, N+1);
            MatrixXd u(m + 1, 3);
            MatrixXd f0(m + 1, N);
            MatrixXd f1(m + 1, N);
            MatrixXd f2(m + 1, N);

    for (int i = 0; i < n; ++i) {
        t(i) = (i + 1) * dt;
        // Set initial condition for the polynomial approximation
        for (int j = 0; j <= m; ++j) {
                u0(j, 0) = U(j, 0);
                u1(j, 0) = U(j, 1); 
                u2(j, 0) = U(j, 2); 
        }
            
            // Fix the boundary, the boundary conditions can be chosen differently, the results are sensitive to the boundary conditions
        for (int l = 0; l < N; ++l) {
            u(0, 0) = u0(0, 0);
            u(m, 0) = u0(m, 0);
            u(0, 1) = u1(0, 0);
            u(m, 1) = u1(m, 0);
            u(0, 2) = u2(0, 0);
            u(m, 2) = u2(m, 0);
            u0(0, l+1) = u0(0, l);
            u0(m, l+1) = u0(m, l);
            u1(0, l+1) = u1(0, l);
            u1(m, l+1) = u1(m, l);
            u2(0, l+1) = u2(0, l);
            u2(m, l+1) = u2(m, l);

            // Update u based on polynomial approximations (up to order 1)
            for (int j = 1; j < m; ++j) {
                    u(j, 0) = u0(j, l);
                    u(j, 1) = u1(j, l);
                    u(j, 2) = u2(j, l);
            }
            for (int j = 1; j < m; ++j) {
                    Vector3d f_current = - dFlux(u, j, dx);
                    u0(j, l+1) = u0(j, l) + (dt / N) * f_current(0); 
                    u1(j, l+1) = u1(j, l) + (dt / N) * f_current(1); 
                    u2(j, l+1) = u2(j, l) + (dt / N) * f_current(2); 
                    f0(j, l) = f_current(0);
                    f1(j, l) = f_current(1);
                    f2(j, l) = f_current(2);
            }
        }

                // Update U based on QAEA 
                // Output the density, velocity and temperature distribution at t(i)

            VectorXd F0(N);
            VectorXd F1(N);
            VectorXd F2(N);

        std::cout << "the solutions at t=" << t(i) << ":\n";
        rho(0) = U(0, 0);
        v(0) = U(0, 1)/U(0, 0);
        T(0) = (gamma_1 - 1) * (2 * U(0, 0) * U(0, 2) - U(0, 1) * U(0, 1))/(2 * R *U(0, 0) * U(0, 0));
        std::cout << "rho(0)=" << rho(0) << " ";
        std::cout << "v(0)=" << v(0) << " ";
        std::cout << "T(0)=" << T(0) << "\n";
        for (int j = 1; j < m; ++j) {
            for (int l = 0; l < N; ++l) { 
                F0(l) = f0(j, l);
                F1(l) = f1(j, l);
                F2(l) = f2(j, l);
            }
                U(j, 0) += dt * QAEA(F0);
                U(j, 1) += dt * QAEA(F0);
                U(j, 2) += dt * QAEA(F0);
                rho(j) = U(j, 0);
                v(j) = U(j, 1)/U(j, 0);
                T(j) = (gamma_1 - 1) * (2 * U(j, 0) * U(j, 2) - U(j, 1) * U(j, 1))/(2 * R *U(j, 0) * U(j, 0));
        std::cout << "rho(" <<j << ")=" << rho(j) << " ";
        std::cout << "v(" <<j << ")=" << v(j) << " ";
        std::cout << "T(" <<j << ")=" << T(j) << "\n";
        }
        rho(m) = U(m, 0);
        v(m) = U(m, 1)/U(m, 0);
        T(m) = (gamma_1 - 1) * (2 * U(m, 0) * U(m, 2) - U(m, 1) * U(m, 1))/(2 * R *U(m, 0) * U(m, 0));
        std::cout << "rho(" << m << ")=" << rho(m) << " ";
        std::cout << "v(" << m << ")=" << v(m) << " ";
        std::cout << "T(" << m << ")=" << T(m) << "\n";
    }
    //Further possible function: draw pictures

    return 0;
}