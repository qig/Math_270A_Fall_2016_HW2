#ifndef _SIMULATION_DRIVER_H_
#define _SIMULATION_DRIVER_H_

#include <Eigen/Dense>
#include <Eigen/Core>
#include <fstream>
#include "LagrangianForce.h"

namespace FILE_IO{
  inline void Write_Binary(std::string directory, std::string name,Eigen::VectorXd& v){
    std::string filename(directory+std::string("/")+name+std::string(".binary"));
    std::ofstream outdata(filename.c_str(),std::ios::out|std::ios::binary);
    for(int i=0;i<v.size();i++)
      outdata.write(reinterpret_cast<char*>(&v(i)),sizeof(double));
    outdata.close();
  }

  inline void Read_Binary(std::string directory, std::string name,Eigen::VectorXd& v){
    std::string filename(directory+std::string("/")+name+std::string(".binary"));
    std::ifstream indata(filename.c_str(),std::ios::in|std::ios::binary);
    for(int i=0;i<v.size();i++)
      indata.read(reinterpret_cast<char*>(&v(i)),sizeof(double));
    indata.close();
  }

  inline void Write_Binary(std::string directory, std::string name,Eigen::VectorXf& v){
    std::string filename(directory+std::string("/")+name+std::string(".binary"));
    std::ofstream outdata(filename.c_str(),std::ios::out|std::ios::binary);
    for(int i=0;i<v.size();i++)
      outdata.write(reinterpret_cast<char*>(&v(i)),sizeof(float));
    outdata.close();
  }

  inline void Read_Binary(std::string directory, std::string name,Eigen::VectorXf& v){
    std::string filename(directory+std::string("/")+name+std::string(".binary"));
    std::ifstream indata(filename.c_str(),std::ios::in|std::ios::binary);
    for(int i=0;i<v.size();i++)
      indata.read(reinterpret_cast<char*>(&v(i)),sizeof(float));
    indata.close();
  }

  inline void Write_DAT_File(std::string file,const Eigen::VectorXf& array){
    FILE* fpointer;
    fpointer=fopen(file.c_str(),"w");
    for(int i=0;i<array.size();i++)
        fprintf(fpointer,"%f\n",array(i));
    fclose(fpointer);
}

inline void Write_DAT_File(std::string file,const Eigen::VectorXd& array){
    FILE* fpointer;
    fpointer=fopen(file.c_str(),"w");
    for(int i=0;i<array.size();i++)
        fprintf(fpointer,"%g\n",array(i));
    fclose(fpointer);
}
}

namespace JIXIE{

template <class T>
class SimulationParameters{
public:
  T final_time;
  int frames_per_second;
  T dt;
  std::string output_dir;
  SimulationParameters(){}
};

template <class T>
class SimulationDriver{
public:
T time,dt,dt_target,dt_frame,final_time,dt_min;
int frames_per_second,current_frame;
std::string output_directory;

  SimulationDriver(const T final_time_input,const int frames_per_second_input,const T dt_input,std::string& output_dir):
  final_time(final_time_input),frames_per_second(frames_per_second_input),dt(dt_input),dt_target(dt),
  dt_min((T)1e-6),output_directory(output_dir){
    dt_frame=(T)1/(T)frames_per_second;
    if(dt_target > dt_frame)
      dt_target=dt_frame;
    dt=dt_target;
  }

  SimulationDriver(SimulationParameters<T>& parameters):
  SimulationDriver(parameters.final_time,parameters.frames_per_second,parameters.dt,parameters.output_dir){}

  virtual void Set_Dt(bool& write_frame){
    dt=dt_target;
    if((T)(current_frame + 1) * dt_frame - time < dt_min + dt){
      dt=(T)(current_frame + 1) * dt_frame - time;
      current_frame++;
      write_frame=true;
    }
    else if(time + dt > (T)(current_frame + 1) * dt_frame){
      dt=(T)(current_frame + 1) * dt_frame - time;
      current_frame++;
      write_frame=true;
    }
    else write_frame=false;
  }

  virtual void Initialize(){
    time=(T)0;
    WriteState(current_frame);
  }

  void WriteState(const int number){
    std::ofstream outdata;
    std::string simulation_data_filename(output_directory+std::string("/simulation_info.dat"));
    outdata.open(simulation_data_filename.c_str());
    outdata << number << std::endl;
    outdata.close();
    Write_State(number,simulation_data_filename);
  }

  virtual void Write_State(const int number,std::string& simulation_data_filename){}
  virtual bool Read_State(const int number,std::string& simulation_data_filename){return false;}

  virtual void Advance_One_Time_Step(const bool verbose){
    time+=dt;
  }

  void RunSimulation(const bool verbose=false){

    Initialize();
    while(time<final_time){
      if(verbose)
        std::cout << "Time = " << time << ", frame = " << current_frame << ", dt = " << dt << std::endl;
      bool write_to_file=false;
      Set_Dt(write_to_file);
      Advance_One_Time_Step(verbose);
      if(write_to_file) WriteState(current_frame);
    }
  }
};

template <class T>
class ElasticityParameters:public SimulationParameters<T>{
public:
  int N;
  int Nm1;
  T a;
  T dX;
  T rho;
  T k;
  T Newton_tol;
  int max_newton_it;

  ElasticityParameters(){}
};

template <class T>
class ElasticityDriver: public SimulationDriver<T>{
  using SimulationDriver<T>::output_directory;
  using SimulationDriver<T>::time;
  using SimulationDriver<T>::dt;
  typedef Eigen::Matrix<T,Eigen::Dynamic, 1> TVect;
  int N;
  T a,dX;
  T rho,k;
  TVect x_n,x_np1,v_n,x_hat,residual,mass,delta,delta_sub,residual_sub;
  T Newton_tol;
  int max_newton_it;
  ConstitutiveModel<T>* cons_model;
  LagrangianForces<T>* lf;
  SymmetricTridiagonal<T> be_matrix,be_matrix_sub;
public:

  ElasticityDriver(ElasticityParameters<T>& parameters):
  SimulationDriver<T>(parameters),N(parameters.N),a(parameters.a),dX(parameters.dX),
  rho(parameters.rho),k(parameters.k),x_n(parameters.N),x_np1(parameters.N),v_n(parameters.N),x_hat(parameters.N),residual(parameters.N),mass(parameters.N),delta(parameters.N),
  Newton_tol(parameters.Newton_tol),max_newton_it(parameters.max_newton_it),be_matrix(parameters.N),residual_sub(parameters.Nm1),delta_sub(parameters.Nm1),be_matrix_sub(parameters.Nm1){
    //cons_model=new LinearElasticity<T>(k);
    cons_model=new NeoHookean<T>(k);
    lf=new FEMHyperelasticity<T>(a,dX,N,*cons_model);
  }

  ~ElasticityDriver(){
    delete cons_model;
    delete lf;
  }

  void Initialize(){
    //set intiial positions and velocity
    for(int i=0;i<N;i++){
      T x=(a+(T)i*dX);
      x_n(i)=(T).7*x;
      v_n(i)=(T)0;
    }
    //intialize mass lumped mass matrix from density
    for(int e=0;e<N-1;e++){
      mass(e)+=(T).5*rho*dX;
      mass(e+1)+=(T).5*rho*dX;}
    // To implement the Dirichlet boundary condition
    mass(0) = 0;
    mass(1) = 5./6. * rho * dX;

    SimulationDriver<T>::Initialize();
  }

  virtual void Advance_One_Time_Step(const bool verbose){
    time+=dt;
    x_hat=x_n+dt*v_n;
    x_np1=x_n;//initial guess

    for(int it=1;it<max_newton_it;it++){
      residual=mass.asDiagonal()*(x_hat-x_np1);
      lf->AddForce(residual,x_np1,dt*dt);
      T norm=(T)0;
      for(int i=1;i<N;i++){norm+=residual(i)*residual(i)/mass(i);} // set i = 1 since the first one would be inf
      norm=sqrt(norm);
      if(verbose)
        std::cout << "Newton residual at iteration " << it << " = " << norm << std::endl;
      if(norm<Newton_tol){
        Exit_BE();
        return;}
      //std::cout << "POINT 1" << std::endl; 
      be_matrix.SetToZero();
      for(int i=0;i<N;i++) be_matrix(i,i)=mass(i);
      lf->AddForceDerivative(be_matrix,x_np1,-dt*dt);
      
      for(int i=0;i<N-1;i++) {
	      be_matrix_sub(i,i) = be_matrix(i+1,i+1);
	      if(i<N-2) be_matrix_sub(i,i+1) = be_matrix(i+1,i+2);
	      delta_sub(i) = delta(i+1);
	      residual_sub(i) = residual(i+1);
      }
      
      be_matrix_sub.QRSolve(delta_sub,residual_sub);
      for(int i=0;i<N-1;i++) {
	      be_matrix(i+1,i+1) = be_matrix_sub(i,i);
	      if(i<N-2) be_matrix(i+1,i+2) = be_matrix_sub(i,i+1);
	      delta(i+1) = delta_sub(i);
	      residual(i+1) = residual_sub(i);
      }
      //be_matrix.QRSolve(delta,residual);
      x_np1+=delta;
    }
    Exit_BE();
  }

  void Exit_BE(){
    v_n=((T)1/dt)*(x_np1-x_n);
    x_n=x_np1;
  }

  void Write_State(const int number,std::string& simulation_data_filename){
    std::ofstream basic_outdata;
    basic_outdata.open(simulation_data_filename.c_str(), std::ofstream::out | std::ofstream::app);
    basic_outdata << N << std::endl;
    basic_outdata.close();

    char str[12];
    sprintf(str, "%d", number);
    std::string frame_name(str);

    std::string positions_filename(std::string("particle_x_")+frame_name);
    FILE_IO::Write_Binary(output_directory,positions_filename,x_n);
    std::string velocities_filename(std::string("particle_v_")+frame_name);
    FILE_IO::Write_Binary(output_directory,velocities_filename,v_n);
  }

  void Read_State(const int number){
    Read_State(x_n,v_n,N,output_directory);
    x_np1.resize(N);residual.resize(N);x_hat.resize(N);
  }

  static bool Read_State(TVect& x,TVect& v,int& N,std::string output_directory,const int number){
    std::ifstream basic_indata;
    int last_frame;
    std::string simulation_data_filename(output_directory+std::string("/simulation_info.dat"));
    basic_indata.open(simulation_data_filename.c_str());
    basic_indata >> last_frame;
    basic_indata >> N;
    basic_indata.close();

    if(number>last_frame || number<0) return false;

    char str[12];
    sprintf(str, "%d", number);
    std::string frame_name(str);
    x.resize(N);v.resize(N);
    std::string positions_filename(std::string("particle_x_")+frame_name);
    FILE_IO::Read_Binary(output_directory,positions_filename,x);
    std::string velocities_filename(std::string("particle_v_")+frame_name);
    FILE_IO::Read_Binary(output_directory,velocities_filename,v);

    return true;
  }

};
}
#endif
