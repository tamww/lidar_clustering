// Copyright (C) 2018  Zhi Yan and Li Sun

// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.

// You should have received a copy of the GNU General Public License along
// with this program.  If not, see <http://www.gnu.org/licenses/>.
#include <stdlib.h>
#include <stdio.h> 
//#include "depth2PCL.cpp"
// ROS
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include "sensor_msgs/Image.h"
#include <geometry_msgs/PoseArray.h>
#include <visualization_msgs/MarkerArray.h>
#include "adaptive_clustering/ClusterArray.h"

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>

//#define LOG
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <math.h>


ros::Publisher cluster_array_pub_;
ros::Publisher cloud_filtered_pub_;
ros::Publisher pose_array_pub_;
ros::Publisher marker_array_pub_;
ros::Publisher dist_pub_car;
ros::Publisher dist_pub_line;
ros::Publisher depth_photo;

bool print_fps_;
float z_axis_min_;
float z_axis_max_;
int cluster_size_min_;
int cluster_size_max_;

const int region_max_ = 50; // Change this value to match how far you want to detect.
int regions_[100];

int frames; clock_t start_time; bool reset = true;//fps

//function to calculate dot product of two vectors
// int dot_product(int vector_a[], int vector_b[]) {
//    int product = 0;
//    for (int i = 0; i < size; i++)
//    product = product + vector_a[i] * vector_b[i];
//    return product;
// }
// //function to calculate cross product of two vectors
// void cross_product(int vector_a[], int vector_b[], int &temp[]) {
//    temp[0] = vector_a[1] * vector_b[2] - vector_a[2] * vector_b[1];
//    temp[1] = vector_a[0] * vector_b[2] - vector_a[2] * vector_b[0];
//    temp[2] = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0];
// }

double* dist_to_point(Eigen::Vector2d P, Eigen::Vector2d A, Eigen::Vector2d B){
    static double r[3];
    Eigen::Vector2d AP = P-A;
    Eigen::Vector2d BP = P-B;
    Eigen::Vector2d AB = B-A;
    Eigen::Vector2d minusAB = A-B;
    Eigen::Vector2d ans(0,0);
    auto dist = 0;
    if (AB.dot(AP)>0 && minusAB.dot(BP) >0){
      // auto diagonal = AP.norm()
      // Eigen::Vector2d temp = AB.cross2(AP);
      // auto val = temp.norm();
      // dist = abs(val) / AB.norm();
      ans = (AP.dot(AB) / AB.dot(AB) * AB + A);
      double x1 = ans[0];
      double y1 = ans[1];

      double x2 = A[0];
      double y2 = A[1];

      double x3 = P[0];
      double y3 = P[1];

      dist = sqrt((x3-x2)*(x3-x2) - (y1-y2)*(y1-y2));
      r[0] = dist;
      r[1] = ans[0];
      r[2] = ans[1];
      return r;
    }

    double d_PA = AP.norm();
    double d_PB = BP.norm();
    if (d_PA < d_PB){
      dist = d_PA;
      ans = A;
      r[0] = dist;
      r[1] = ans[0];
      r[2] = ans[1];
      return r;
    }else{
      dist = d_PB;
      ans = B;
      r[0] = dist;
      r[1] = ans[0];
      r[2] = ans[1];
      return r;
    }
}
    // Vector2d a(5.0, 6.0);
    // Vector2d a(5.0, 6.0);
    // Vector2d a(5.0, 6.0);
    // Vector2d a(5.0, 6.0);

void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& ros_pc2_in) {
  if(print_fps_)if(reset){frames=0;start_time=clock();reset=false;}//fps
  
  /*** Convert ROS message to PCL ***/
  pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_in(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*ros_pc2_in, *pcl_pc_in);
  
  /*** Remove ground and ceiling ***/
  pcl::IndicesPtr pc_indices(new std::vector<int>);
  pcl::PassThrough<pcl::PointXYZI> pt;
  pt.setInputCloud(pcl_pc_in);
  pt.setFilterFieldName("z");
  pt.setFilterLimits(z_axis_min_, z_axis_max_);
  // pt.setFilterFieldName("y");
  // pt.setFilterLimits(5, -5);
  // pt.setFilterLimitsNegative(true);
  pt.filter(*pc_indices);


  // pt.setFilterLimitsNegative(false);
  // pt.filter(*pc_indices);

  // pt.setFilterFieldName("y");
  // pt.setFilterLimits(-20, -300);




  
  /*** Divide the point cloud into nested circular regions ***/
  boost::array<std::vector<int>, region_max_> indices_array;
  for(int i = 0; i < pc_indices->size(); i++) {
    float range = 0.0;
    for(int j = 0; j < region_max_; j++) {
      float d2 = pcl_pc_in->points[(*pc_indices)[i]].x * pcl_pc_in->points[(*pc_indices)[i]].x +
	pcl_pc_in->points[(*pc_indices)[i]].y * pcl_pc_in->points[(*pc_indices)[i]].y +
	pcl_pc_in->points[(*pc_indices)[i]].z * pcl_pc_in->points[(*pc_indices)[i]].z;
      if(d2 > range * range && d2 <= (range+regions_[j]) * (range+regions_[j])) {
      	indices_array[j].push_back((*pc_indices)[i]);
      	break;
      }
      range += regions_[j];
    }
  }
  
  /*** Euclidean clustering ***/
  float tolerance = 0.0;
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr, Eigen::aligned_allocator<pcl::PointCloud<pcl::PointXYZI>::Ptr > > clusters;
  
  for(int i = 0; i < region_max_; i++) {
    tolerance += 0.1;
    if(indices_array[i].size() > cluster_size_min_) {
      boost::shared_ptr<std::vector<int> > indices_array_ptr(new std::vector<int>(indices_array[i]));
      pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
      tree->setInputCloud(pcl_pc_in, indices_array_ptr);
      
      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
      ec.setClusterTolerance(tolerance);
      ec.setMinClusterSize(cluster_size_min_);
      ec.setMaxClusterSize(cluster_size_max_);
      ec.setSearchMethod(tree);
      ec.setInputCloud(pcl_pc_in);
      ec.setIndices(indices_array_ptr);
      ec.extract(cluster_indices);
      
      for(std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); it++) {
      	pcl::PointCloud<pcl::PointXYZI>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZI>);
      	for(std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit) {
      	  cluster->points.push_back(pcl_pc_in->points[*pit]);
  	}
      	cluster->width = cluster->size();
      	cluster->height = 1;
      	cluster->is_dense = true;
	clusters.push_back(cluster);
      }
    }
  }
  
  /*** Output ***/
  if(cloud_filtered_pub_.getNumSubscribers() > 0) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_pc_out(new pcl::PointCloud<pcl::PointXYZI>);
    sensor_msgs::PointCloud2 ros_pc2_out;
    pcl::copyPointCloud(*pcl_pc_in, *pc_indices, *pcl_pc_out);
    pcl::toROSMsg(*pcl_pc_out, ros_pc2_out);
    cloud_filtered_pub_.publish(ros_pc2_out);
  }
  
/************************************/
  int idd = clusters.size();
    // basic coordinate of the car
    // draw outline
    // To- do

    geometry_msgs::Point car[11];
    // lower
    car[0].x =0; car[0].y =0.415; car[0].z =0;
    car[1].x =0; car[1].y =-0.415; car[1].z =0;
    car[2].x =0.8; car[2].y =0.415; car[2].z =0; 
    car[3].x =0.8; car[3].y =-0.415; car[3].z =0; 
    // upper
    car[4].x =0; car[4].y =0.415; car[4].z =-1.3;
    car[5].x =0; car[5].y =-0.415; car[5].z =-1.3;
    car[6].x =0.8; car[6].y =0.415; car[6].z =-1.3; 
    car[7].x =0.8; car[7].y =-0.415; car[7].z =-1.3; 

/************************************/
    // view of camera
    visualization_msgs::MarkerArray dist_array_car;
    visualization_msgs::Marker markerk;
    car[8].x = 0.8; car[8].y=0; car[8].z = -0.53;
    car[9].x = 3.8; car[9].y=1.5; car[9].z = -0.53;
    car[10].x = 3.8; car[10].y=-1.5; car[10].z = -0.53;
    markerk.scale.x = 0.05;
    markerk.color.a = 1.0;
    markerk.color.r = 0.0;
    markerk.color.g = 1.0;
    markerk.color.b = 0.0;
    markerk.pose.orientation.w = 1.0;
    markerk.header = ros_pc2_in->header;
    markerk.ns = "adaptive_clustering";
    markerk.id = idd+1;
    idd++;
    markerk.type = visualization_msgs::Marker::LINE_LIST;

    markerk.points.push_back(car[8]);
    markerk.points.push_back(car[9]);
    markerk.points.push_back(car[8]);
    markerk.points.push_back(car[10]);
    markerk.lifetime = ros::Duration(0.1);
    dist_array_car.markers.push_back(markerk);
    if(dist_array_car.markers.size()) {
      dist_pub_car.publish(dist_array_car);
    }
/**********************************/
/************************************/
    // dist line
    visualization_msgs::MarkerArray dist_array_dist;
    
    // Eigen::Vector2d a(5.0, 6.0);
    // Eigen::Vector2d a(5.0, 6.0);
    // Eigen::Vector2d a(5.0, 6.0);
    // Eigen::Vector2d a(5.0, 6.0);
/************************************/
  adaptive_clustering::ClusterArray cluster_array;
  geometry_msgs::PoseArray pose_array;
  visualization_msgs::MarkerArray marker_array;
  int flag = 0;
  for(int i = 0; i < clusters.size(); i++) {
    if(cluster_array_pub_.getNumSubscribers() > 0) {
      sensor_msgs::PointCloud2 ros_pc2_out;
      pcl::toROSMsg(*clusters[i], ros_pc2_out);
      cluster_array.clusters.push_back(ros_pc2_out);
    }
    
    if(pose_array_pub_.getNumSubscribers() > 0) {
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*clusters[i], centroid);
      
      geometry_msgs::Pose pose;
      pose.position.x = centroid[0];
      pose.position.y = centroid[1];
      pose.position.z = centroid[2];
      pose.orientation.w = 1;
      pose_array.poses.push_back(pose);
      
// #ifdef LOG
      Eigen::Vector4f min, max;
      pcl::getMinMax3D(*clusters[i], min, max);
      std::cerr << ros_pc2_in->header.seq << " "
		<< ros_pc2_in->header.stamp << " "
		<< min[0] << " "
		<< min[1] << " "
		<< min[2] << " "
		<< max[0] << " "
		<< max[1] << " "
		<< max[2] << " "
		<< std::endl;
// #endif
    }
    
    if(marker_array_pub_.getNumSubscribers() > 0) {
      Eigen::Vector4f min, max;
      pcl::getMinMax3D(*clusters[i], min, max);
      
      visualization_msgs::Marker marker;
      marker.header = ros_pc2_in->header;
      marker.ns = "adaptive_clustering";
      marker.id = i;
      marker.type = visualization_msgs::Marker::LINE_LIST;
      
      geometry_msgs::Point p[28];
      p[0].x = max[0];  p[0].y = max[1];  p[0].z = max[2];
      p[1].x = min[0];  p[1].y = max[1];  p[1].z = max[2];

      p[2].x = max[0];  p[2].y = max[1];  p[2].z = max[2];
      p[3].x = max[0];  p[3].y = min[1];  p[3].z = max[2];

      p[4].x = max[0];  p[4].y = max[1];  p[4].z = max[2];
      p[5].x = max[0];  p[5].y = max[1];  p[5].z = min[2];

      p[6].x = min[0];  p[6].y = min[1];  p[6].z = min[2];
      p[7].x = max[0];  p[7].y = min[1];  p[7].z = min[2];

      p[8].x = min[0];  p[8].y = min[1];  p[8].z = min[2];
      p[9].x = min[0];  p[9].y = max[1];  p[9].z = min[2];

      p[10].x = min[0]; p[10].y = min[1]; p[10].z = min[2];
      p[11].x = min[0]; p[11].y = min[1]; p[11].z = max[2];

      p[12].x = min[0]; p[12].y = max[1]; p[12].z = max[2];
      p[13].x = min[0]; p[13].y = max[1]; p[13].z = min[2];

      p[14].x = min[0]; p[14].y = max[1]; p[14].z = max[2];
      p[15].x = min[0]; p[15].y = min[1]; p[15].z = max[2];

      p[16].x = max[0]; p[16].y = min[1]; p[16].z = max[2];
      p[17].x = max[0]; p[17].y = min[1]; p[17].z = min[2];

      p[18].x = max[0]; p[18].y = min[1]; p[18].z = max[2];
      p[19].x = min[0]; p[19].y = min[1]; p[19].z = max[2];

      p[20].x = max[0]; p[20].y = max[1]; p[20].z = min[2];
      p[21].x = min[0]; p[21].y = max[1]; p[21].z = min[2];

      p[22].x = max[0]; p[22].y = max[1]; p[22].z = min[2];
      p[23].x = max[0]; p[23].y = min[1]; p[23].z = min[2];
      /*extra*/
      p[24].x = max[0]; p[24].y = max[1];  p[24].z = min[2];
      p[25].x = max[0]; p[25].y = min[1];  p[25].z = max[2];

      p[26].x = max[0];p[26].y = max[1];  p[26].z = min[2];
      p[27].x = max[0]; p[27].y = min[1];  p[27].z = max[2];

      // int flag = 0;
      for(int i = 0; i < 28; i++) {
        // if(p[i].x <-2){
        //   flag = 1;
        //   break;
        // }
  	    marker.points.push_back(p[i]);
      }
      // if(!flag){
        marker.scale.x = 0.02;
        marker.color.a = 1.0;
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.5;
        marker.lifetime = ros::Duration(0.1);
        marker_array.markers.push_back(marker);
      // }
/*****************/
/**shortest path between box and egocar**/
      geometry_msgs::Point pt[4];
      pt[0] = p[0];
      pt[1] = p[1];
      pt[2] = p[3];
      // pt[3] = p[5];
      // pt[4] = p[6];
      // pt[5] = p[7];
      // pt[6] = p[9];
      pt[3] = p[11];
    

      double mindist = 999;
      Eigen::Vector3d minQ(0, 0,1.3);
      Eigen::Vector3d minP(0, 0,1.3);
      double temp = 0;
      for (int i = 4; i <8;i++){
        for (int j = 0;j<4;j++){
          Eigen::Vector2d P(car[i].x, car[i].y);
          Eigen::Vector2d A(pt[j].x, pt[j].y);
          Eigen::Vector2d B(pt[0].x, pt[0].y);
          if (j!=3){
            // Eigen::Vector2d B(pt[j+1].x,pt[j+1].y);
            B[0] = pt[j+1].x;
            B[1] = pt[j+1].y;
          }
          // Eigen::Vector2d ans(pt[j].x,pt[j].y);
          double* distance = dist_to_point(P, A,B);
          if (mindist > distance[0] ){
            mindist = distance[0];
            minQ[0]=distance[1];
            minQ[1]=distance[2];
            minQ[2]=pt[j].z;

            minP[0]=car[i].x;
            minP[1]=car[i].y;
            minP[2]=car[i].z;
          }
        }
      }
      for (int j = 0;j<4;j++){
        for (int i = 4; i <8;i++){
          Eigen::Vector2d P(pt[j].x, pt[j].y);
          Eigen::Vector2d A(car[i].x, car[i].y);
          Eigen::Vector2d B(car[4].x,car[4].y);
          if(i!=7){
            B[0] = car[i+1].x;
            B[1] = car[i+1].y;
          }
          // Eigen::Vector2d ans(car[i].x,car[i].y);
          double* distance = dist_to_point(P, A,B);
          if (mindist > distance[0] ){
            mindist = distance[0];
            minQ[0]=distance[1];
            minQ[1]=distance[2];
            minQ[2]=car[i].z;

            minP[0]=pt[j].x;
            minP[1]=pt[j].y;
            minP[2]=pt[j].z;
          }
        }
      }
      geometry_msgs::Point finalss[2];
      if(minQ[0] >0 && minP[0]> -1 && !(minP[1] > car[1].y && minP[1] <car[0].y )){
        finalss[0].x = minQ[0];
        finalss[0].y = minQ[1];
        finalss[0].z = minQ[2];

        finalss[1].x = minP[0];
        finalss[1].y = minP[1];
        finalss[1].z = minP[2];
        std::cout<<finalss<<std::endl;

        visualization_msgs::Marker markerDist;

        markerDist.header = ros_pc2_in->header;
        markerDist.ns = "adaptive_clustering";
        markerDist.id = idd+1;
        idd++;
        markerDist.type = visualization_msgs::Marker::LINE_LIST;
        markerDist.scale.x = 0.02;
        markerDist.color.a = 1.0;
        markerDist.color.r = 1.0;
        markerDist.color.g = ((int)mindist) %255/100;
        markerDist.color.b = 0.0;
        markerDist.lifetime = ros::Duration(0.1);
        markerDist.points.push_back(finalss[0]);
        markerDist.points.push_back(finalss[1]);
        dist_array_dist.markers.push_back(markerDist);

        visualization_msgs::Marker textMarker;
        textMarker.header.frame_id = "/dist_text";
        textMarker.header.stamp = ros::Time();
        textMarker.id = i + 100;
        textMarker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        textMarker.action = visualization_msgs::Marker::ADD;
        

        double x3 = (finalss[0].x + finalss[1].x)/2.0;
        double y3 = (finalss[0].y + finalss[1].y)/2.0;
        double z3 = (finalss[0].z + finalss[1].z)/2.0;
        textMarker.lifetime = ros::Duration(0.1);
        textMarker.pose.position.x = x3;
        textMarker.pose.position.y = y3;
        textMarker.pose.position.z = z3;
        // auto s = ;
        textMarker.text = std::to_string(mindist);

        // textMarker.scale.x = 2;
        // textMarker.scale.y = 2;
        textMarker.scale.z = 0.5;

        textMarker.color.r = 1.0f;
        textMarker.color.g = 1.0f;
        textMarker.color.b = 1.0f;
        textMarker.color.a = 1.0;
        std::cerr<<x3<<" "<<y3<<" "<<z3<<" dist: "<<mindist<<std::endl;

        dist_array_dist.markers.push_back(textMarker);
        if(mindist <= 1.5 && (y3 <0.5 && y3>-0.5)){
          flag++;
        }
      }
    }


  }
  //  <node pkg="rviz" type="rviz" name="rviz" args="-d $(env PWD)/adaptive_clustering.rviz"/>
  if(flag != 0){
    std::cerr<<"****************stop"<<std::endl;
    system("bash end.sh");
  }else{
    std::cerr<<"!!!!!!!!!!!!!!!!!continue"<<std::endl;
    system("bash start.sh");
  }
  visualization_msgs::Marker marker;
  marker.header.frame_id = "/my_frame";
  marker.header.stamp = ros::Time::now();
  marker.ns = "basic_shapes";
  marker.id = 999;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::Marker::ADD;

  marker.pose.position.x = 0.0 + 1;
  marker.pose.position.y = 1.0;
  marker.pose.position.z = 1.0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;

  marker.text = "blablabla";

  marker.scale.x = 0.3;
  marker.scale.y = 0.3;
  marker.scale.z = 0.1;

  marker.color.r = 0.0f;
  marker.color.g = 1.0f;
  marker.color.b = 0.0f;
  marker.color.a = 1.0;
  dist_array_dist.markers.push_back(marker);
// overall publish

  if(cluster_array.clusters.size()) {
    cluster_array.header = ros_pc2_in->header;
    cluster_array_pub_.publish(cluster_array);
  }

  if(pose_array.poses.size()) {
    pose_array.header = ros_pc2_in->header;
    pose_array_pub_.publish(pose_array);
  }
  
  if(marker_array.markers.size()) {
    marker_array_pub_.publish(marker_array);
  }
  
  if(dist_array_dist.markers.size()) {
    dist_pub_line.publish(dist_array_dist);
  }

  if(print_fps_)if(++frames>10){std::cerr<<"[adaptive_clustering] fps = "<<float(frames)/(float(clock()-start_time)/CLOCKS_PER_SEC)<<", timestamp = "<<clock()/CLOCKS_PER_SEC<<std::endl;reset = true;}//fps
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "adaptive_clustering");
  
  /*** Subscribers ***/
  ros::NodeHandle nh;
  ros::Subscriber point_cloud_sub = nh.subscribe<sensor_msgs::PointCloud2>("points_raw", 1, pointCloudCallback);
  //ros::Subscriber depth_input = nh.subscribe<sensor_msgs::Image>("/simulator/camera_node/depth", 100, );
  ROS_INFO_STREAM("Hello ROS");
  /*** Publishers ***/
  ros::NodeHandle private_nh("~");
  cluster_array_pub_ = private_nh.advertise<adaptive_clustering::ClusterArray>("clusters", 100);
  cloud_filtered_pub_ = private_nh.advertise<sensor_msgs::PointCloud2>("cloud_filtered", 100);
  pose_array_pub_ = private_nh.advertise<geometry_msgs::PoseArray>("poses", 100);
  marker_array_pub_ = private_nh.advertise<visualization_msgs::MarkerArray>("markers", 100);
  dist_pub_car = private_nh.advertise<visualization_msgs::MarkerArray>("marker_car", 10);
  dist_pub_line = private_nh.advertise<visualization_msgs::MarkerArray>("marker_dist", 10);
  //depth_photo = private_nh.advertise<sensor_msgs::PointCloud2>("cloud_filtered", 100);
  /*** Parameters ***/
  std::string sensor_model;
  
  private_nh.param<std::string>("sensor_model", sensor_model, "VLP-16"); // VLP-16, HDL-32E, HDL-64E
  private_nh.param<bool>("print_fps", print_fps_, false);
  private_nh.param<float>("z_axis_min", z_axis_min_, -1);
  private_nh.param<float>("z_axis_max", z_axis_max_, 3);
  private_nh.param<int>("cluster_size_min", cluster_size_min_, 70);
  private_nh.param<int>("cluster_size_max", cluster_size_max_, 2200000);
  
  // Divide the point cloud into nested circular regions centred at the sensor.
  // For more details, see our IROS-17 paper "Online learning for human classification in 3D LiDAR-based tracking"
  if(sensor_model.compare("VLP-16") == 0) {
    regions_[0] = 2; regions_[1] = 3; regions_[2] = 3; regions_[3] = 3; regions_[4] = 3;
    regions_[5] = 3; regions_[6] = 3; regions_[7] = 2; regions_[8] = 3; regions_[9] = 3;
    regions_[10]= 3; regions_[11]= 3; regions_[12]= 3; regions_[13]= 3;
  } else if (sensor_model.compare("HDL-32E") == 0) {
    regions_[0] = 4; regions_[1] = 5; regions_[2] = 4; regions_[3] = 5; regions_[4] = 4;
    regions_[5] = 5; regions_[6] = 5; regions_[7] = 4; regions_[8] = 5; regions_[9] = 4;
    regions_[10]= 5; regions_[11]= 5; regions_[12]= 4; regions_[13]= 5;
  } else if (sensor_model.compare("HDL-64E") == 0) {
    regions_[0] = 14; regions_[1] = 14; regions_[2] = 14; regions_[3] = 15; regions_[4] = 14;
  } else {
    ROS_FATAL("Unknown sensor model!");
  }
  
  ros::spin();

  return 0;
}
