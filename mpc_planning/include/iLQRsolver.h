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
    

    iLQRsolver();
    void initialize_mtx();
    void set_mpc_params();
    void set_vehicle_params();
    void debuging();
    template <typename var>
    Matrix< var, 1, 1 > switch_inst_index(
        const Matrix< var, Dynamic, 1 > &x,
        const Matrix< var, Dynamic, 1 > &u);

    template <typename var>
    Matrix< var, Dynamic, 1 > dynamics_continuous(
        const var &t,
        const Matrix< var, Dynamic, 1 > &x,
        const Matrix< var, Dynamic, 1 > &u)
    {
        // state, input
        var beta = x[0], r = x[1], psi = x[2], x_glo = x[3], y_glo = x[4], v = x[5];
        var delta = u[0], T = u[1];

        // Vertical Forces
        var ay = v*r;
        var Fzfl = (g*lr/2-ax*h/2-ay*lr*h/Lw+ax*ay*h*h/g/Lw)*m/(lr+lf);
        var Fzfr = (g*lr/2-ax*h/2+ay*lr*h/Lw-ax*ay*h*h/g/Lw)*m/(lr+lf);
        var Fzrl = (g*lf/2+ax*h/2-1.2*ay*lf*h/Lw-ax*ay*h*h/g/Lw)*m/(lr+lf);
        var Fzrr = (g*lf/2+ax*h/2+1.2*ay*lf*h/Lw+ax*ay*h*h/g/Lw)*m/(lr+lf);

        // Lateral Forces (Dugoff tire model, assuming longitudinal slip is zero)
        var alpha_f = delta - beta - r*lf/v;
        var alpha_r = -beta + r*lr/v;
        Matrix< var, 1, 1 > Fy1, Fy2, Fy3, Fy4;
        Fy1 = dugoff_model(Cf, Fzfl, alpha_f);
        Fy2 = dugoff_model(Cf, Fzfr, alpha_f);
        Fy3 = dugoff_model(Cr, Fzrl, alpha_r);
        Fy4 = dugoff_model(Cr, Fzrr, alpha_r);
        var Fyfl = Fy1[0], Fyfr = Fy2[0], Fyrl = Fy3[0], Fyrr = Fy4[0];

        var betadot = (Fyfl+Fyfr+Fyrl+Fyrr)/v/m - r;
        var rdot = ((Fyfl+Fyfr)*lf - (Fyrl+Fyrr)*lr)/Iz; 
        var psidot = r;
        var Xdot = v * cos(psi+beta);
        var Ydot = v * sin(psi+beta);
        var vdot = (T/reff - 0.0225*m*g - v*v*Cd*rho*A/2)/(m+4*Iw/reff/reff);

        Matrix< var, Dynamic, 1 > dx(state_dim);
        dx << betadot, rdot, psidot, Xdot, Ydot, vdot;
        return dx;
    }
    
    template <typename var>
    Matrix< var, 1, 1 > dugoff_model(
        const double &Ca,
        const var &Fz,
        const var &alpha)
    {
        var F = 1;
        Matrix< var, 1, 1 > Fy;
        Fy[0] = 1e-1;
        if (abs(alpha) > 0.01){
            
            var lammda = mu*Fz/pow(2*pow(Ca*tan(alpha),2)+1,1/2);
            
            if (lammda < 1 && lammda > 0) {
                F = (2-lammda)*lammda;
                
            }
            
            Fy[0] = Ca*tan(alpha)*F;

        }else{
            Fy[0] = Ca*tan(alpha);
        }
        return Fy;
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

#ifdef __DYN_RK4__
        Matrix< var, Dynamic, 1 > dx(state_dim);
        Matrix< var, Dynamic, 1 > x_new(state_dim);
        Matrix< var, Dynamic, 1 > k1(state_dim);
        Matrix< var, Dynamic, 1 > k2(state_dim);
        Matrix< var, Dynamic, 1 > k3(state_dim);
        Matrix< var, Dynamic, 1 > k4(state_dim);
        var t = i * dt;
        var t_;
        Matrix< var, Dynamic, 1 > x_(state_dim);

        k1 = dynamics_continuous(t, x, u);
        x_ = x + 0.5*k1*dt;
        k2 = dynamics_continuous(t_, x_, u);

        x_ = x + 0.5*k2*dt;
        k3 = dynamics_continuous(t_, x_, u);

        t_ = t + dt;
        x_ = x + k3*dt;
        k4 = dynamics_continuous(t_, x_, u);
        x_new = x + (k1 + 2*k2 + 2*k3 + k4) * (dt/6.0);
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

    for (int itr = 0; itr < n_itrs; itr++){
        itr_ = itr+1;
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
            if (itr == 0 && j == 0){
                J_opt = J_new;
                x_array = x_array_new;
                u_array = u_array_new;
            }else {
                if (abs((J_opt[0]-J_new[0])/J_opt[0]) < J_reltol) {
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

template <typename var>
    Matrix< var, 1, 1 > iLQRsolver::switch_inst_index(
        const Matrix< var, Dynamic, 1 > &x,
        const Matrix< var, Dynamic, 1 > &u)
    {
        // state, input
        var beta = x[0], r = x[1], psi = x[2], x_glo = x[3], y_glo = x[4], v = x[5];
        var delta = u[0], T = u[1];

        // Vertical Forces
        var ay = v*r;
        var Fzfl = (g*lr/2-ax*h/2-ay*lr*h/Lw+ax*ay*h*h/g/Lw)*m/(lr+lf);
        var Fzfr = (g*lr/2-ax*h/2+ay*lr*h/Lw-ax*ay*h*h/g/Lw)*m/(lr+lf);
        var Fzrl = (g*lf/2+ax*h/2-1.2*ay*lf*h/Lw-ax*ay*h*h/g/Lw)*m/(lr+lf);
        var Fzrr = (g*lf/2+ax*h/2+1.2*ay*lf*h/Lw+ax*ay*h*h/g/Lw)*m/(lr+lf);

        // Lateral Forces (Dugoff tire model, assuming longitudinal slip is zero)
        var alpha_f = delta - beta - r*lf/v;
        var alpha_r = -beta + r*lr/v;
        Matrix< var, 1, 1 > Fy1, Fy2, Fy3, Fy4;
        Fy1 = dugoff_model(Cf, Fzfl, alpha_f);
        Fy2 = dugoff_model(Cf, Fzfr, alpha_f);
        Fy3 = dugoff_model(Cr, Fzrl, alpha_r);
        Fy4 = dugoff_model(Cr, Fzrr, alpha_r);
        var Fyfl = Fy1[0], Fyfr = Fy2[0], Fyrl = Fy3[0], Fyrr = Fy4[0];

        var arg1 = Fyfl/Fzfl, arg2 = Fyfr/Fzfr, arg3 = Fyrl/Fzrl, arg4 = Fyrr/Fzrr, arg5 = ay/g;
        var val = arg1, val_temp = arg3;
        int index = 1, index_temp = 3;
        if (arg2 > val){
            val = arg2;
            index = 2;
        }
        if (arg4 > val_temp){
            val_temp = arg4;
            index_temp = 4;
        }
        if (arg5 > val_temp){
            val_temp = arg5;
            index_temp = 5;
        }
        if (val_temp > val){
            index = index_temp;
        }

        inst_index = index;
        Matrix< var, 1, 1 > return_val;
        return_val[0] = inst_index;
        return return_val;
    }

iLQRsolver::iLQRsolver()
{
    set_mpc_params();
    set_vehicle_params();
    initialize_mtx();   
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

