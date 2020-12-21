#include <mpc_planning_node.h>

using namespace chrono;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mpc_planning_node");
	ros::NodeHandle nh;

    Planner* planner = new Planner(&nh);

    bool is_RT = false;

    /*********** Real-time with ROS version **********/
    ros::Rate r(20); // 20 Hz
    int i = 0;
    while (ros::ok() && is_RT) {
        planner->tf_broadcaster();
        if (planner->start_plan == true){
            planner->solve_OP(i);
        }
        ros::spinOnce();
        r.sleep();
        i++;

        // Repeat planner
        if (i == planner->N_final){
            planner->initialize();
            i = 0;
        }
    }
    /*************************************************/

    /******* Non-real-time simulation version ********/
    if (~is_RT){
        for (size_t i = 0; i < planner->N_final; i++) {
            if (planner->start_plan == true){
                planner->solve_OP(i);
            }
        }
    }
    /*************************************************/

    planner->writeData(); // write data in .txt in ~/save_data

    return 0;
}

void Planner::solve_OP(const int i)
{
    /* 
     * 
     * Solve Optimization Problem using iLQR solver
     * 
     * */
    ilqr.t_cur = i*ilqr.dt;
    u_init = u_array_opt;
    system_clock::time_point start = system_clock::now();
    start_prev = start;

    /********************************* Iterate iLQR ********************************/
    ilqr.ilqr_iterate(x, u_init, x_array_opt, u_array_opt, k_array_opt, K_array_opt);
    /*******************************************************************************/

    system_clock::time_point end = system_clock::now();
    cal_time_vec[i] = duration_cast<microseconds>(end - start);

    /* Control input and States are updated */
    u = u_array_opt[0];
    x = x_array_opt[0];
    x = ilqr.dynamics_discrete(i, x, u);
    u_hist[i] = u;
    x_hist[i] = x;

    /* Switch instabiliy minimize target */
    VectorXd inst_target(1);
    inst_target = ilqr.switch_inst_index(x, u);
    instTarget_hist[i] = inst_target[0];
}

void iLQRsolver::set_mpc_params()
{
    // solver settings
    state_dim = 6;
    input_dim = 2;
    dt = 0.05;
    N = 20;
    n_itrs = 100;
    n_line = 10;
    J_reltol = 1e-6;
    diff_eps = 1e-5;
    
    // cost related variables
    double yaw_error_max = 1.8*pi/180;
    double pos_error_max = 0.5;
    double v_error_max = 0.03; //0.12
    double SF = 0.05;
    Q_yaw = SF*1/pow(yaw_error_max,2);
    Q_pos = SF*1/pow(pos_error_max,2);
    Q_v = SF*1/pow(v_error_max,2);
    R_delta = 50;
    R_T = 0.0002;
    S_delta = 1500;
    S_T = 0.0005;
    v_ref = 60. * 1000./3600.;
    y_ref = 0.;
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

template <typename var>
    Matrix< var, 1, 1 > iLQRsolver::cost_instantaneous(
        const size_t &i,
        const Matrix< var, Dynamic, 1 > &x,
        const Matrix< var, Dynamic, 1 > &u,
        const Matrix< var, Dynamic, 1 > &u_p)
    {
        // state, input
        var beta = x[0], r = x[1], psi = x[2], x_glo = x[3], y_glo = x[4], v = x[5];
        var delta = u[0], T = u[1];
        var delta_d = u_p[0]-u[0], T_d = u_p[1] - u[1];

        Matrix< var, 1, 1 > cost;
        //cost[0] += Q_pos * pow(x_glo-x_ref, 2);
        cost[0] += Q_pos * pow(y_glo-y_ref, 2);
        cost[0] += Q_v * pow(v-v_ref, 2);
        cost[0] += R_delta * pow(delta, 2);
        cost[0] += R_T * pow(T, 2);
        cost[0] += S_delta * pow(delta_d, 2);
        //cost[0] += S_T * pow(T_d, 2);

    
        /************************** Obstacle Potential Field **************************/
        double af = 6*sqrt(2)/2, b = 2.5*sqrt(2)/2;
        int p_obs = 50;
        var K_obs;
        if (t_cur > 1.7 )
        {
            if (x_glo < p_obs){
                double Dis = 5;
                K_obs = pow(pow((x_glo - p_obs)/Dis,2) + pow((y_glo + 0.2)/b,2), 0.5) - 1;  
            }else
            {
                K_obs = pow(pow((x_glo - p_obs)/af,2) + pow((y_glo + 0.2)/b,2), 0.5) - 1;
            }
            if (K_obs < 1e-2){K_obs = 1e-2;}
            cost[0] += 2*exp(-0.1*K_obs)/K_obs;
            var K_obs2 = pow(pow((x_glo - p_obs + 5)/5,2) + pow((y_glo + 2)/b,2), 0.5) - 1;  
            cost[0] += 2*exp(-0.1*K_obs2)/K_obs2;
        }
        /******************************************************************************/
    
        /*********************** Road & Lane Potential Field **************************/
        int B = 200;
        //cost[0] += 1/2*B*pow( pow(1/(y_glo+2),2)+pow(1/(y_glo-6),2), 2); // Road boundary
        //cost[0] += 10*exp(-pow((y_glo+2)/0.5,2)); // Lane
        /******************************************************************************/
    

        /******************* Instabiliy Risk Potential Field (IRPF) *******************/
        var ay = v*r;
        var Fz, alpha;
        double Ca;
        bool is_tire = true;
        switch (inst_index)
        {
            case 1: // Front Left Tire
                Fz = (g*lr/2-ax*h/2-ay*lr*h/Lw+ax*ay*h*h/g/Lw)*m/(lr+lf);
                alpha = delta - beta - r*lf/v;
                Ca = Cf;
                break;
            case 2: // Front Right Tire
                Fz = (g*lr/2-ax*h/2+ay*lr*h/Lw-ax*ay*h*h/g/Lw)*m/(lr+lf);
                alpha = delta - beta - r*lf/v;
                Ca = Cf;
                break;
            case 3: // Rear Left Tire
                Fz = (g*lf/2+ax*h/2-1.2*ay*lf*h/Lw-ax*ay*h*h/g/Lw)*m/(lr+lf);
                alpha = -beta + r*lr/v;
                Ca = Cr;
                break;
            case 4: // Rear Right Tire
                Fz = (g*lf/2+ax*h/2+1.2*ay*lf*h/Lw+ax*ay*h*h/g/Lw)*m/(lr+lf);
                alpha = -beta + r*lr/v;
                Ca = Cr;
                break;
            case 5: // case for body acceleration limit
                is_tire = false; 
        }
        var R_inst = 1;
        if (is_tire)
        {
            //Matrix< var, 1, 1 > Fy_;
            //var Fy = 0.85*ay*m;
            //R_inst = 1 - sqrt(pow(T/(mu*m*g*reff),2) + pow(1/m/g/(mu*0.9),2));
        }else{
            R_inst = 1 - sqrt(pow((T-0.0225*m*g*reff)/(mu*(m+4*Iw/reff/reff)*g*reff),2) + pow(0.85*ay/(mu*0.9*g),2));
        }
        R_inst = 1 - sqrt(pow((T-0.0225*m*g*reff)/(mu*(m+4*Iw/reff/reff)*g*reff),2) + pow(0.85*ay/(mu*0.9*g),2));
        if(R_inst > 1){R_inst = 1;}
        if(R_inst < 1e-2){R_inst = 1e-2;}
        cost[0] += 0.16*exp(-0.1*R_inst)/R_inst;

        /******************************************************************************/
        
        return cost;
    }

void iLQRsolver::debuging()
{

}