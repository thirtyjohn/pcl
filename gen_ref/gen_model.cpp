#include <iostream>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <unistd.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/features/shot_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <boost/thread/thread.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/transforms.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::Normal NormalType;
typedef pcl::SHOT352 DescriptorType;

#define  PC_XYZ_PTR pcl::PointCloud<pcl::PointXYZ>::Ptr 
#define  PC_XYZ pcl::PointCloud<pcl::PointXYZ> 
#define  PASS_XYZ pcl::PassThrough<pcl::PointXYZ> 
#define  PC_READER  pcl::PCDReader
#define  PC_VIEWER  pcl::visualization::PCLVisualizer
#define SACSEG_XYZ  pcl::SACSegmentation<pcl::PointXYZ> 
#define INDICE_PTR pcl::PointIndices::Ptr
#define INDICE pcl::PointIndices
#define MODEL_COEF_PTR pcl::ModelCoefficients::Ptr 
#define MODEL_COEF  pcl::ModelCoefficients
#define PCD_WRITER pcl::PCDWriter
#define EXTRACT_INDICE_XYZ pcl::ExtractIndices<pcl::PointXYZ> 
#define VEC_POINT_IDICES std::vector<pcl::PointIndices> 
#define VEC_INT std::vector<int> 
#define STD_STR_STREAM std::stringstream
#define KD_TREE pcl::search::KdTree<pcl::PointXYZ>
#define KD_TREE_PTR pcl::search::KdTree<pcl::PointXYZ>::Ptr
#define EUCLIDEAN_EXTRACT_XYZ pcl::EuclideanClusterExtraction<pcl::PointXYZ> 


float z_min (-0.500f);
float z_max (1.200f);
float y_min (-1.00f);
float y_max (1.00f);
float x_min (-0.1f);
float x_max (0.1f);
float sample_size_ (0.01f);


float descr_rad_ (0.02f);
// PCL viewer //
boost::mutex cloud_mutex;
pcl::SHOTEstimationOMP<PointT, NormalType, DescriptorType> descr_est;
pcl::NormalEstimationOMP<PointT, NormalType> norm_est;

void
parseCommandLine (int argc, char *argv[])
{
  //General parameters
  pcl::console::parse_argument (argc, argv, "--z_max", z_max);
  pcl::console::parse_argument (argc, argv, "--z_min", z_min);
  pcl::console::parse_argument (argc, argv, "--y_max", y_max);
  pcl::console::parse_argument (argc, argv, "--y_min", y_min);
  pcl::console::parse_argument (argc, argv, "--x_max", x_max);
  pcl::console::parse_argument (argc, argv, "--x_min", x_min); 
  pcl::console::parse_argument (argc, argv, "--sample_size", sample_size_);

  std::cout << "x_min:" << x_min << " x_max:" << x_max << " y_min:" << y_min << " y_max:"<< y_max <<std::endl;


}
void DrtFilter(PC_XYZ_PTR cloud_in, PC_XYZ_PTR cloud_out, float x_min, float x_max,
               float y_min, float y_max, float z_min, float z_max)
{
 	PASS_XYZ pass;

       // pass.setFilterLimitsNegative (true);
	pass.setInputCloud (cloud_in);
	pass.setFilterFieldName ("z");
	pass.setFilterLimits (z_min, z_max);
	pass.filter (*cloud_out);

        pass.setInputCloud (cloud_out);
	pass.setFilterFieldName ("y");
	pass.setFilterLimits (y_min, y_max);
	pass.filter (*cloud_out);
	
        pass.setInputCloud (cloud_out);
	pass.setFilterFieldName ("x");
	pass.setFilterLimits (x_min, x_max);
	pass.filter (*cloud_out);

   return;
}

void ReadPc(char* fileName, PC_XYZ_PTR cloud)
{
    PC_READER reader;
    reader.read(fileName, *cloud);

    std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl;
    return;
}

void DisplayPc(PC_XYZ_PTR cloud)
{
    PC_VIEWER viewer("PCL Viewer");

    viewer.addPointCloud<PointT> (cloud, "input_cloud");
    while (!viewer.wasStopped ())
    {
   	viewer.spinOnce ();
    }      	
   
    viewer.removeAllPointClouds();

    return;
}
void Extract(PC_XYZ_PTR cloud_in)
{
  // Create the segmentation object for the planar model and set all the parameters
  SACSEG_XYZ seg;
  INDICE_PTR inliers(new INDICE);
  MODEL_COEF_PTR coefficients (new MODEL_COEF);
  PC_XYZ_PTR cloud_plane (new PC_XYZ);
  PCD_WRITER writer;
  EXTRACT_INDICE_XYZ extract;
  PC_XYZ_PTR cloud_f(new PC_XYZ);
  int i, nr_points;

  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.02);

  i=0;
  nr_points = (int) cloud_in->points.size ();

  while (cloud_in->points.size () > 0.3 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud(cloud_in);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the planar inliers from the input cloud
    extract.setInputCloud (cloud_in);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);
    std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_f);
    *cloud_in = *cloud_f;
  }
  
  return;
}
void Ec(PC_XYZ_PTR cloud_in)
{
  PCD_WRITER writer;
  KD_TREE_PTR tree(new KD_TREE);
  VEC_POINT_IDICES cluster_indices;
  EUCLIDEAN_EXTRACT_XYZ ec;
  PC_XYZ_PTR cloud_cluster(new PC_XYZ);
  STD_STR_STREAM ss;
  int j;
  VEC_POINT_IDICES::const_iterator it;
  VEC_INT::const_iterator pit;

  tree->setInputCloud (cloud_in);
  ec.setClusterTolerance (0.02); // 2cm
  ec.setMinClusterSize (100);
  ec.setMaxClusterSize (25000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_in);
  ec.extract (cluster_indices);

  j = 0;
  for (it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    for (pit = it->indices.begin (); pit != it->indices.end (); pit++)
      cloud_cluster->points.push_back (cloud_in->points[*pit]);

    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    cout << "ss:" << j << std::endl;  
    
    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
    ss << "cloud_cluster_" << j << ".pcd";
    cout << "ss:" << j << std::endl;  
    writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false); //*
    DisplayPc(cloud_cluster);
    j++;
  }   

    return;
}

int
main (int argc, char** argv)
{
  PC_XYZ_PTR cloud_in(new PC_XYZ), cloud_filtered(new PC_XYZ), cloud(new PC_XYZ);

  //Parse command line
  parseCommandLine(argc, argv);

  // Read in the cloud data
  ReadPc("../bottle.pcd", cloud_in);

  DrtFilter(cloud_in, cloud_filtered, x_min, x_max, y_min,  y_max,  z_min, z_max);

  //DisplayPc(cloud_filtered);

  Extract(cloud_filtered);

  //DisplayPc(cloud_filtered);

  Ec(cloud_filtered);

  return 0;
}
