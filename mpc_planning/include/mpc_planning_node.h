#include <iLQRsolver.h>
#include <fstream>
#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <std_msgs/Float32MultiArray.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

using namespace chrono;

class Planner
{
private:
    ros::NodeHandle nh_; 
    ros::Subscriber sub_joy;
    ros::Subscriber sub_states;
    ros::Publisher pub_inputs;

    int sim_time = 5;
    int N_state, N_input, Hp;
    iLQRsolver ilqr;
    VectorXd x;
    VectorXd u;
    vector<VectorXd> u_init;
    vector<VectorXd> x_array_opt;
    vector<VectorXd> u_array_opt;
    vector<VectorXd> k_array_opt;
    vector<MatrixXd> K_array_opt;
    microseconds cal_time, step_time;
    system_clock::time_point start_prev;
    vector<VectorXd> x_hist;
    vector<VectorXd> u_hist;
    VectorXd instTarget_hist;    
    VectorXd itr_num;  
    vector<microseconds> cal_time_vec;

public:
    int N_final;
    Planner(ros::NodeHandle* nodehandle);
    void joyCallback(const sensor_msgs::Joy::ConstPtr& msg);
    void statesCallback(const std_msgs::Float32MultiArray::ConstPtr& msg);
    void solve_OP(const int i);
    void initialize();
    void writeData();
    bool start_plan;
    void tf_broadcaster();
};

Planner::Planner(ros::NodeHandle* nodehandle) : nh_(*nodehandle)
{
    /* Topic sub & Pub */
    sub_joy = nh_.subscribe("joy", 1, &Planner::joyCallback, this);
    sub_states = nh_.subscribe("states", 1, &Planner::statesCallback, this);
    pub_inputs = nh_.advertise<std_msgs::Float32MultiArray>("control_inputs",100);

    this->initialize();
}

void Planner::joyCallback(const sensor_msgs::Joy::ConstPtr& msg)
{
    /* Joy listener for high-level command */

    // Start planner when X button pressed
    if (msg->buttons[0] == 1){
        ROS_INFO("Start Motion Planning!! ");
        start_plan = true;
    } 

    // Abort planner when B button pressed
    if (msg->buttons[1] == 1){
        ROS_WARN("Planner is RESET!! ");
        initialize();
    }
}

void Planner::statesCallback(const std_msgs::Float32MultiArray::ConstPtr& msg)
{
    /* State update from estimator */
    // double vx = msg->data[2], vy = msg->data[3];
    // x(0) = msg->data[0];
    // x(1) = msg->data[1];
    // x(2) = msg->data[6];
    // x(3) = sqrt(vx*vx+vy*vy);
}
void Planner::initialize()
{
    ilqr.initialize_mtx();
    /* initialization */
    start_plan = true;
    Hp = ilqr.N;
    N_state = ilqr.state_dim;
    N_input = ilqr.input_dim;
    N_final = sim_time/ilqr.dt;
    x = VectorXd::Zero(ilqr.state_dim);
    u_init = vector<VectorXd>(Hp+1,VectorXd::Zero(N_input));
    x_array_opt = vector<VectorXd>(Hp+1,VectorXd::Zero(N_state));
    u_array_opt = vector<VectorXd>(Hp+1,VectorXd::Zero(N_input));
    k_array_opt = vector<VectorXd>(Hp,VectorXd::Zero(N_input));
    K_array_opt = vector<MatrixXd>(Hp,MatrixXd::Zero(N_input,N_state));
    x_hist = vector<VectorXd>(N_final,VectorXd::Zero(N_state));
    u_hist = vector<VectorXd>(N_final,VectorXd::Zero(N_input));
    instTarget_hist = VectorXd::Zero(N_final);
    itr_num = VectorXd::Zero(N_final);
    cal_time_vec = vector<microseconds>(N_final);

}

void Planner::writeData()
{
    ofstream writeFile;
    writeFile.open("/home/nvidia/save_data/mpc_test.csv");
    if (writeFile.is_open()) {
        writeFile << "cal_time,x,y,yaw,steer_cmd,itr_num" << endl;
        for (int i = 0; i < N_final; i++)
            writeFile
                << cal_time_vec[i].count()/1000.0 << ","
                << x_hist[i][0] << ","
                << x_hist[i][1] << ","
                << x_hist[i][2] << ","
                << u_hist[i].transpose() << ","
                << itr_num[i] << endl;
    }
    writeFile.close();
}

void Planner::tf_broadcaster()
{
    /* Transformation broadcaster for Rviz visualization */

    // static tf2_ros::TransformBroadcaster br;
    // geometry_msgs::TransformStamped transformStamped;

    // transformStamped.header.stamp = ros::Time::now();
    // transformStamped.header.frame_id = "base_link";
    // transformStamped.child_frame_id = "robot";
    // transformStamped.transform.translation.x = x[3];
    // transformStamped.transform.translation.y = x[4];
    // transformStamped.transform.translation.z = 0.0;
    // tf2::Quaternion q;
    // q.setRPY(0, 0, x[2]);
    // transformStamped.transform.rotation.x = q.x();
    // transformStamped.transform.rotation.y = q.y();
    // transformStamped.transform.rotation.z = q.z();
    // transformStamped.transform.rotation.w = q.w();

    // br.sendTransform(transformStamped);
}