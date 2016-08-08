//
// Created by Zengpan Fan on 8/3/16.
//

#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/text_format.h>
#include <caffe/util/db_pipe.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/mr_feature_extraction.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
namespace db = caffe::db;

int MRFeatureExtraction::run_feature_extraction_pipeline(
  const char* pretrained_binary_proto,
  const char* feature_extraction_proto
) {
  stop_signal_ = false;

  return feature_extraction_pipeline<float>(
    pretrained_binary_proto,
    feature_extraction_proto,
    "pool5/7x7_s1",
    get_output_pipe_path(),
    50,
    Caffe::GPU,
    0
  );
}

template<typename Dtype>
int MRFeatureExtraction::feature_extraction_pipeline(
  std::string pretrained_binary_proto,
  std::string feature_extraction_proto,
  std::string extract_feature_blob_names,
  std::string target_pipe_path,
  int mini_batch_size,
  Caffe::Brew caffe_mode,
  int device_id
) {
  ::google::InitGoogleLogging("feature_extraction_pipeline");

  if (caffe_mode == Caffe::GPU) {
    LOG(ERROR)<< "Using GPU";

    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  boost::shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto, caffe::TEST));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  std::vector<std::string> blob_names;
  boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));

  size_t num_blobs = blob_names.size();

  for (size_t i = 0; i < num_blobs; i++) {
    CHECK(feature_extraction_net->has_blob(blob_names[i]))
        << "Unknown feature blob name " << blob_names[i]
        << " in the network " << feature_extraction_proto;
  }

  LOG(INFO)<< "Opening pipe " << target_pipe_path;
  boost::shared_ptr<db::Pipe> feature_db(new db::Pipe());
  feature_db->Open(target_pipe_path, db::WRITE);

  size_t num_features = blob_names.size();

  std::vector<boost::shared_ptr<db::PipeTransaction> > txns;
  for (size_t i = 0; i < num_features; ++i) {
    boost::shared_ptr<db::PipeTransaction> txn(feature_db->NewTransaction());
    txns.push_back(txn);
  }

  LOG(ERROR)<< "Extracting Features";

  Datum datum;
  std::vector<int> image_indices(num_blobs, 0);

  // The total number of images to be processed is the product of batch_size in prototxt and num_mini_batches here.
  // The app here has no awareness of the number of image files.
  while (!stop_signal_) {
    feature_extraction_net->Forward();
    for (int i = 0; i < num_blobs; ++i) {
      const boost::shared_ptr<Blob<Dtype> > feature_blob =
          feature_extraction_net->blob_by_name(blob_names[i]);
      int batch_size = feature_blob->num();
      int dim_features = feature_blob->count() / batch_size;
      const Dtype* feature_blob_data;
      for (int n = 0; n < batch_size; ++n) {
        datum.set_height(feature_blob->height());
        datum.set_width(feature_blob->width());
        datum.set_channels(feature_blob->channels());
        datum.clear_data();
        datum.clear_float_data();
        feature_blob_data = feature_blob->cpu_data() +
                            feature_blob->offset(n);
        for (int d = 0; d < dim_features; ++d) {
          datum.add_float_data(feature_blob_data[d]);
        }

        LOG(ERROR) << "origin message size" << datum.ByteSize();
        txns.at(i)->Put(datum);
        ++image_indices[i];
        if (image_indices[i] % mini_batch_size == 0) {
          txns.at(i)->Commit();
          txns.at(i).reset(feature_db->NewTransaction());
          LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
                    " query images for feature blob " << blob_names[i];
        }
      }  // for (int n = 0; n < batch_size; ++n)
    }  // for (int i = 0; i < num_features; ++i)
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  // write the last batch

  for (int i = 0; i < num_features; ++i) {
    if (image_indices[i] % mini_batch_size != 0) {
      txns.at(i)->Commit();
    }
    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
              " query images for feature blob " << blob_names[i];
  }

  feature_db->Close();

  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}
