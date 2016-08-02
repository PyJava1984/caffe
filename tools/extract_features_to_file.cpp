#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <sys/stat.h>
#include "boost/algorithm/string.hpp"

#include "caffe/blob.hpp"
#include "caffe/net.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using std::string;

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  const int num_required_args = 6;
  if (argc < num_required_args) {
    LOG(ERROR)<<
              "This program takes in a trained network and an input data layer, and then"
                  " extract features of the input data produced by the net.\n"
                  "Usage: extract_features  pretrained_net_param"
                  "  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
                  "  save_feature_dataset_sub_folder_name1[,name2,...]  num_mini_batches"
                  "  [CPU/GPU] [DEVICE_ID=0]\n"
                  "Note: you can extract multiple features in one pass by specifying"
                  " multiple feature blob names and dataset names separated by ','."
                  " The names cannot contain white space characters and the number of blobs"
                  " and datasets must be equal.";
    return 1;
  }
  int arg_pos = num_required_args;

  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    int device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  arg_pos = 0;  // the name of the executable
  std::string base_folder_name(argv[++arg_pos]);
  mkdir(base_folder_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

  std::string pretrained_binary_proto(argv[++arg_pos]);

  std::string feature_extraction_proto(argv[++arg_pos]);
  boost::shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto, caffe::TEST));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  std::string extract_feature_blob_names(argv[++arg_pos]);
  std::vector<std::string> blob_names;
  boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

  std::string save_feature_dataset_names(argv[++arg_pos]);
  std::vector<std::string> dataset_names;
  boost::split(dataset_names, save_feature_dataset_names,
               boost::is_any_of(","));
  CHECK_EQ(blob_names.size(), dataset_names.size()) <<
                                                    " the number of blob names and dataset names must be equal";
  size_t num_blobs = blob_names.size();

  for (size_t i = 0; i < num_blobs; i++) {
    CHECK(feature_extraction_net->has_blob(blob_names[i]))
    << "Unknown feature blob name " << blob_names[i]
    << " in the network " << feature_extraction_proto;
  }

  int num_mini_batches = atoi(argv[++arg_pos]);

  std::vector<std::string> feature_folder_names;
  for (size_t i = 0; i < num_blobs; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];
    std::string feature_folder_name(base_folder_name + "/" + dataset_names[i]);
    mkdir(feature_folder_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    feature_folder_names.push_back(feature_folder_name);
  }

  LOG(ERROR)<< "Extracting Features";

  std::vector<int> image_indices(num_blobs, 0);

  // The total number of images to be processed is the product of batch_size in prototxt and num_mini_batches here.
  // The app here has no awareness of the number of image files.
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    feature_extraction_net->Forward();
    for (int i = 0; i < num_blobs; ++i) {
      const boost::shared_ptr<Blob<Dtype> > feature_blob =
          feature_extraction_net->blob_by_name(blob_names[i]);
      int batch_size = feature_blob->num();
      int dim_features = feature_blob->count() / batch_size;
      const Dtype* feature_blob_data;
      for (int n = 0; n < batch_size; ++n) {
        string key_str = caffe::format_int(image_indices[i], 10);
        std::string feature_file_path = feature_folder_names[i] + "/" + key_str + ".txt";
        std::fstream feature_stream(feature_file_path.c_str(), std::ios_base::out | std::ios_base::trunc);

        feature_stream << feature_blob->height() << " "
                       << feature_blob->width() << " "
                       << feature_blob->channels() << " "
                       << dim_features << " ";

        feature_blob_data = feature_blob->cpu_data() +
                            feature_blob->offset(n);

        if (dim_features > 0) {
          feature_stream << feature_blob_data[0];

          for (int d = 1; d < dim_features; ++d) {
            feature_stream << " " << feature_blob_data[d];
          }
        }

        ++image_indices[i];

        if (image_indices[i] % 1000 == 0) {
          LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
                    " query images for feature blob " << blob_names[i];
        }
      }  // for (int n = 0; n < batch_size; ++n)
    }  // for (int i = 0; i < num_features; ++i)
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  // write the last batch
  for (int i = 0; i < num_blobs; ++i) {
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
              " query images for feature blob " << blob_names[i];
  }

  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}
