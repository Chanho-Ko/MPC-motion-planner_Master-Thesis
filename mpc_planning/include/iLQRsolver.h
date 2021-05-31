#include <cppad/cppad.hpp>
#include <cppad/example/cppad_eigen.hpp>
#include <cppad/speed/det_by_minor.hpp>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <chrono>

using namespace std;

using CppAD::AD;
using CppAD::NearEqual;
using Eigen::Matrix;
using Eigen::Dynamic;

#define STATE_NUM 3

typedef Matrix< double    , Dynamic, Dynamic > MatrixXd;
typedef Matrix< AD<double>, Dynamic, Dynamic > MatrixAD;

typedef Matrix< double    , Dynamic, 1 >       VectorXd;
typedef Matrix< AD<double>, Dynamic, 1 >       VectorAD;

#define __DYN_EULER__
// #define __DYN_RK4__

// #define __VERBOSE__

#define __DIFF_CPPAD__
// #define __DIFF_FINITE__

class iLQRsolver
{
private:
    // solver settings
    size_t n_itrs;
    size_t n_line;
    double J_min;
    double J_reltol;
    double diff_eps;
    bool converged = false;

    // cost related variables
    double Q_pos, Q_yaw, Q_v;
    double R_delta, R_T;
    double S_delta, S_T;
    double yaw_ref, x_ref, y_ref, v_ref;
    int inst_index = 1;
    // model variables
    double mu;
    double m, g, Iz, Lw, lf, lr, L, h, ax, ay; // vehicle body
    double Cf, Cr, Iw, reff; // tire
    double A, rho, Cd; // aero dynamics
   
    // iteration variables
    vector<VectorXd> x_array;
    vector<VectorXd> u_array; // u_array[N] should be always 0
    vector<VectorXd> x_array_new;
    vector<VectorXd> u_array_new;
    vector<VectorXd> k_array;
    vector<MatrixXd> K_array;
    VectorXd J_opt;
    VectorXd J_new;
    CppAD::ADFun<double> f;
    CppAD::ADFun<double> l;
    VectorXd xu;
    MatrixXd f_xu_t, f_xu, fx, fu;
    MatrixXd l_xu, lx, lu;
    MatrixXd l_xuxu, lxx, lux, luu;
    MatrixXd Vx, Vxx;
    MatrixXd Qx, Qu, Qxx, Quu, Qux, invQuu;

public:
    int itr_;
    double pi = 3.141592;
    size_t state_dim;
    size_t input_dim;
    double dt;
    double t_cur = 0;
    size_t N;
    double v_init;
    double v = 30 / 3.6;
    

    iLQRsolver();
    void initialize_mtx();
    void set_mpc_params();
    void set_vehicle_params();
    void debuging();

    template <typename var>
    Matrix< var, Dynamic, 1 > dynamics_continuous(
        const var &t,
        const Matrix< var, Dynamic, 1 > &x,
        const Matrix< var, Dynamic, 1 > &u)
    {
        // state, input
        var psi = x[2];
        var delta = u[0];

        // State-space
        var Xdot = v * cos(psi+delta);
        var Ydot = v * sin(psi+delta);
        var Pdot = v/L*sin(delta);

        Matrix< var, Dynamic, 1 > dx(state_dim);
        dx << Xdot, Ydot, Pdot;
        return dx;
    }
    
    template <typename var>
    Matrix< var, 1, 1 > cost_instantaneous(
        const size_t &i,
        const Matrix< var, Dynamic, 1 > &x,
        const Matrix< var, Dynamic, 1 > &u,
        const Matrix< var, Dynamic, 1 > &u_p);

    
    template <typename var>
    Matrix< var, Dynamic, 1 > dynamics_discrete(
        const size_t &i,
        const Matrix< var, Dynamic, 1 > &x,
        const Matrix< var, Dynamic, 1 > &u)
    {
#ifdef __DYN_EULER__
        Matrix< var, Dynamic, 1 > dx(state_dim);
        Matrix< var, Dynamic, 1 > x_new(state_dim);
        var t = i * dt;
        dx = dynamics_continuous(t, x, u);
        x_new = x + dx * dt;
        return x_new;
#endif
    }

    void forward_propagation(
        const VectorXd &x0,
        const vector<VectorXd> &u_array,
        vector<VectorXd> &x_array)
    {
        x_array[0] = x0;
        for (size_t i = 0; i < N; i++)
            x_array[i+1] = dynamics_discrete(i, x_array[i], u_array[i]);
    }

    VectorXd trajectory_cost(
        const vector<VectorXd> &x_array,
        const vector<VectorXd> &u_array)
    {
        VectorXd J(1);
        for (size_t i = 0; i < N; i++)
            J += cost_instantaneous(i, x_array[i], u_array[i], u_array[i+1]);
        return J;
    }

    void apply_control(
        const vector<VectorXd> &x_array,
        const vector<VectorXd> &u_array,
        const vector<VectorXd> &k_array,
        const vector<MatrixXd> &K_array,
        const double alpha,
        vector<VectorXd> &x_array_new,
        vector<VectorXd> &u_array_new)
    {
        x_array_new[0] = x_array[0];
        for (size_t i = 0; i < N; i++){
            u_array_new[i] = u_array[i] + alpha*k_array[i] + K_array[i]*(x_array_new[i]-x_array[i]);
            x_array_new[i+1] = dynamics_discrete(i, x_array_new[i], u_array_new[i]);
        }
    }

    void build_ADFun(
        const size_t &i,
        CppAD::ADFun<double> &f,
        CppAD::ADFun<double> &l)
    {
        VectorAD ad_xu(state_dim+input_dim);
        VectorAD ad_x(state_dim);
        VectorAD ad_u(input_dim);
        VectorAD ad_u_p(input_dim-1);
        VectorAD ad_dx(state_dim);
        VectorAD ad_cost(1);
        ad_xu.setRandom(state_dim+input_dim, 1);

        CppAD::Independent(ad_xu);
        ad_x = ad_xu.head(state_dim);
        ad_u = ad_xu.tail(input_dim);
        ad_dx = dynamics_discrete(i, ad_x, ad_u);
        CppAD::ADFun<double> f_(ad_xu, ad_dx);
        f = f_;

        CppAD::Independent(ad_xu);
        ad_x = ad_xu.head(state_dim);
        ad_u = ad_xu.tail(input_dim);
        ad_u_p = ad_xu.tail(input_dim-1);

        ad_cost = cost_instantaneous(i, ad_x, ad_u, ad_u);
        CppAD::ADFun<double> l_(ad_xu, ad_cost);
        l = l_;
    }

    void ilqr_iterate(
        const VectorXd &x0,
        const vector<VectorXd> &u_init,
        vector<VectorXd> &x_array_opt,
        vector<VectorXd> &u_array_opt,
        vector<VectorXd> &k_array_opt,
        vector<MatrixXd> &K_array_opt);
};

template <typename var>
    Matrix< var, 1, 1 > iLQRsolver::cost_instantaneous(
        const size_t &i,
        const Matrix< var, Dynamic, 1 > &x,
        const Matrix< var, Dynamic, 1 > &u,
        const Matrix< var, Dynamic, 1 > &u_p)
    {
        // state, input
        var x_glo = x[0], y_glo = x[1], psi = x[2];
        var delta = u[0];
        var delta_d = u_p[0]-u[0];

        Matrix< var, 1, 1 > cost;
        //cost[0] += Q_pos * pow(x_glo-x_ref, 2);
        cost[0] += Q_pos * pow(y_glo-y_ref, 2);
        cost[0] += R_delta * pow(delta, 2);
        cost[0] += S_delta * pow(delta_d, 2);
        //cost[0] += S_T * pow(T_d, 2);

    
        /************************** Obstacle Potential Field **************************/
        double af = 3*sqrt(2)/2, b = 2.5*sqrt(2)/2;
        int p_obs = 30;
        var K_obs;
    
        K_obs = pow(pow((x_glo - p_obs)/af,2) + pow((y_glo - 1.9)/b,2), 0.5) - 1;
        cost[0] += 5*exp(-1*K_obs);

        /******************************************************************************/
    
        /*********************** Road & Lane Potential Field **************************/
        //int B = 200;
        //cost[0] += 1/2*B*pow( pow(1/(y_glo+2),2)+pow(1/(y_glo-6),2), 2); // Road boundary
        //cost[0] += 10*exp(-pow((y_glo+2)/0.5,2)); // Lane
        /******************************************************************************/

        return cost;
    }

void iLQRsolver::ilqr_iterate(
        const VectorXd &x0,
        const vector<VectorXd> &u_init,
        vector<VectorXd> &x_array_opt,
        vector<VectorXd> &u_array_opt,
        vector<VectorXd> &k_array_opt,
        vector<MatrixXd> &K_array_opt)
{
    double alpha = 1.0;
    converged = false;

    u_array = u_init;
    forward_propagation(x0, u_array, x_array);
    J_opt = trajectory_cost(x_array, u_array);
    J_opt[0] = 1e+6;
    for (int itr = 0; itr < n_itrs; itr++){
        itr_ = itr+1;
        cout << "Iteration : " << itr_ << endl;
        // Initialization of Vx, Vxx
        build_ADFun(N, f, l);
        xu << x_array[N], u_array[N];

        l_xu = l.Jacobian(xu);
        lx = l_xu.topRows(state_dim);

        l_xuxu = l.Hessian(xu, 0);
        l_xuxu.resize(state_dim+input_dim, state_dim+input_dim);
        lxx = l_xuxu.topLeftCorner(state_dim, state_dim);

        Vx = lx;
        Vxx = lxx;

        // Back propagation
        for (int i = N-1; i >= 0; i--){
            build_ADFun(i, f, l);
            xu << x_array[i], u_array[i];

            f_xu_t = f.Jacobian(xu);
            f_xu_t.resize(state_dim+input_dim, state_dim);
            f_xu = f_xu_t.transpose();
            fx = f_xu.leftCols(state_dim);
            fu = f_xu.rightCols(input_dim);

            l_xu = l.Jacobian(xu);
            lx = l_xu.topRows(state_dim);
            lu = l_xu.bottomRows(input_dim);

            l_xuxu = l.Hessian(xu, 0);
            l_xuxu.resize(state_dim+input_dim, state_dim+input_dim);
            lxx = l_xuxu.topLeftCorner(state_dim, state_dim);
            lux = l_xuxu.bottomLeftCorner(input_dim, state_dim);
            luu = l_xuxu.bottomRightCorner(input_dim, input_dim);

            Qx = lx + fx.transpose() * Vx;
            Qu = lu + fu.transpose() * Vx;
            Qxx = lxx + fx.transpose() * Vxx * fx;
            Quu = luu + fu.transpose() * Vxx * fu;
            Qux = lux + fu.transpose() * Vxx * fx;

            invQuu = Quu.inverse();
            k_array[i] = -invQuu*Qu;
            K_array[i] = -invQuu*Qux;
            // k_array[i] = -Quu.ldlt().solve(Qu);
            // K_array[i] = -Quu.ldlt().solve(Qux);

            Vx = Qx - K_array[i].transpose() * Quu * k_array[i];
            Vxx = Qxx - K_array[i].transpose() * Quu * K_array[i];
        }

        // Line search
        for (int j = 0; j < n_line; j++) {
            alpha = pow(1.1, -pow(j, 2));
            // alpha = pow(0.8, j);
            
            apply_control(x_array, u_array, k_array, K_array, alpha, x_array_new, u_array_new);
            J_new = trajectory_cost(x_array_new, u_array_new);
            //std::cout << "u_array_new :  \n" << u_array_new[0] << "\n"  << u_array_new[1] << "\n"  << u_array_new[5] << "\n"  << u_array_new[10]<< "\n"   << u_array_new[19] << std::endl;
            //std::cout << "x_array_new :  \n" << x_array_new[1] << std::endl;
            cout << "line search Num : " << j << endl;
            cout << "J_opt : " << J_opt[0] << endl;
            cout << "J_new : " << J_new[0] << endl;
            if (itr == 0 && j == 0){
                J_opt = J_new;
                x_array = x_array_new;
                u_array = u_array_new;
            }else {
                if (abs((J_opt[0]-J_new[0])/J_opt[0]) < J_reltol || J_opt[0] <= J_min || J_new[0] <= J_min || (J_new[0] > J_opt[0] && j==0 )) {
                    J_opt = J_new;
                    x_array = x_array_new;
                    u_array = u_array_new;
                    converged = true;
                    break;
                }
                else if (J_new[0] < J_opt[0]) {
                    J_opt = J_new;
                    x_array = x_array_new;
                    u_array = u_array_new;
                    break;
                }
            }
        }
        if (converged)
            break;
    }
    //std::cout << itr_ << std::endl;
    x_array_opt = x_array;
    u_array_opt = u_array;
    k_array_opt = k_array;
    K_array_opt = K_array;

}

iLQRsolver::iLQRsolver()
{
    set_mpc_params();
    set_vehicle_params();
    initialize_mtx();   
}

void iLQRsolver::set_mpc_params()
{
    // solver settings
    state_dim = 3;
    input_dim = 1;
    dt = 0.05;
    N = 6;
    n_itrs = 10;
    n_line = 3;
    J_min = 1e-3;
    J_reltol = 1e-3;
    diff_eps = 1e-7;
    
    // cost related variables
    double yaw_error_max = 1.8*pi/180;
    double pos_error_max = 0.5;
    double v_error_max = 0.03; //0.12
    double SF = 1;
    Q_yaw = SF*1/pow(yaw_error_max,2);
    Q_pos = SF*1/pow(pos_error_max,2);
    Q_v = SF*1/pow(v_error_max,2);
    R_delta = 5;
    R_T = 0.0002;
    S_delta = 3000;
    S_T = 0.0005;
    y_ref = 2.;
}

void iLQRsolver::initialize_mtx()
{
    x_array = vector<VectorXd>(N+1, VectorXd(state_dim));
    u_array = vector<VectorXd>(N+1, VectorXd(input_dim)); // u_array[N] should be always 0
    x_array_new = vector<VectorXd>(N+1, VectorXd(state_dim));
    u_array_new = vector<VectorXd>(N+1, VectorXd(input_dim));
    k_array = vector<VectorXd>(N, VectorXd(input_dim));
    K_array = vector<MatrixXd>(N, MatrixXd(input_dim, state_dim));
    J_opt = VectorXd(1);
    J_new = VectorXd(1);
    xu = VectorXd(state_dim+input_dim);
}

void iLQRsolver::set_vehicle_params()
{   
    /* Simulation parameter */
    mu = 0.9;
    v_init = 60; // kph

    /* Vehicle body */
    m = 1650 + 180; // Sprung mass and unsprung mass
    g = 9.81;
    Iz = 3234;
    Lw = 1.6; // Track width
    lf = 1.4;
    lr = 1.65;
    L = lf + lr;
    h = 0.53;
    ax = 0; ay = 0;

    /* Tire */
    reff = 0.353;
    Iw = 0.9; // wheel inertia
    // norminal cornering stiffness [N/rad]
    Cf = (1305.3)*180/pi; // Fzf = 4856*2 N
    Cr = (1122.7)*180/pi; // Fzr = 4140*2 N


    /* Aero Dynamics */
    A = 2.8;
    rho = 1.206;
    Cd = 0.3;
}
