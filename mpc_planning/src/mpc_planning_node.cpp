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
            std::cout << "Hi there!" << std::endl;
            planner->solve_OP(i);
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
    cout << "Cal Time : " << cal_time_vec[i].count()/1000 << endl;

    /* Control input and States are updated */
    u = u_array_opt[0];
    x = x_array_opt[0];
    x = ilqr.dynamics_discrete(i, x, u);
    u_hist[i] = u;
    x_hist[i] = x;

    /* Switch instabiliy minimize target */
    VectorXd inst_target(1);
    //inst_target = ilqr.switch_inst_index(x, u);
    instTarget_hist[i] = inst_target[0];

    itr_num[i] = ilqr.itr_;
}


void iLQRsolver::debuging()
{

}