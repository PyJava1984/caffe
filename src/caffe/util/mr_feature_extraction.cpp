//
// Created by Zengpan Fan on 8/3/16.
//

#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/text_format.h>
#include <caffe/util/db_pipe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/mr_feature_extraction.hpp"

#include <aws/core/client/ClientConfiguration.h>
#include <aws/core/client/CoreErrors.h>
#include <aws/core/auth/AWSCredentialsProviderChain.h>
#include <aws/core/platform/Platform.h>
#include <aws/core/utils/Outcome.h>
#include <aws/s3/S3Client.h>
#include <aws/core/utils/ratelimiter/DefaultRateLimiter.h>
#include <aws/s3/model/DeleteBucketRequest.h>
#include <aws/s3/model/CreateBucketRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/PutObjectRequest.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>
#include <aws/core/utils/HashingUtils.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/DeleteObjectRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/CreateMultipartUploadRequest.h>
#include <aws/s3/model/UploadPartRequest.h>
#include <aws/s3/model/CompleteMultipartUploadRequest.h>
#include <aws/s3/model/ListObjectsRequest.h>
#include <aws/s3/model/GetBucketLocationRequest.h>
#include <aws/core/utils/DateTime.h>
#include <aws/core/http/HttpClientFactory.h>
#include <aws/core/http/HttpClient.h>

#include <fstream>

#include <aws/core/http/standard/StandardHttpRequest.h>

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
namespace db = caffe::db;

using namespace Aws::Auth;
using namespace Aws::Http;
using namespace Aws::Client;
using namespace Aws::S3;
using namespace Aws::S3::Model;
using namespace Aws::Utils;

int MRFeatureExtraction::run_feature_extraction_pipeline(
  const char* pretrained_binary_proto,
  const char* feature_extraction_proto
) {
  stop_signal_ = false;

  ::google::InitGoogleLogging("feature_extraction_pipeline");

  return feature_extraction_pipeline(
    pretrained_binary_proto,
    feature_extraction_proto,
    "pool5/7x7_s1",
    get_output_pipe_path(),
    50,
    Caffe::CPU,
    0
  );
}

void MRFeatureExtraction::start_feature_extraction_pipeline(
  std::string pretrained_binary_proto,
  std::string feature_extraction_proto
) {
  ::google::InitGoogleLogging("feature_extraction_pipeline");

  stop_signal_ = false;

  LOG(ERROR) << "Using GPU";

  LOG(ERROR) << "Using Device_id=" << 0;
  // Caffe::SetDevice(0);
  Caffe::set_mode(Caffe::CPU);

  feature_extraction_net_ = new Net<float>(feature_extraction_proto, caffe::TEST);
  feature_extraction_net_->CopyTrainedLayersFrom(pretrained_binary_proto);

  LOG(ERROR) << "Finished loading model";
}

const boost::shared_ptr<Blob<float> > MRFeatureExtraction::process_batch(
  std::vector<caffe::Datum> &batch
) {
  boost::shared_ptr<caffe::MemoryDataLayer<float>> inmem_layer =
    boost::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(feature_extraction_net_->layers()[0]);

  if (!inmem_layer) {
    LOG(ERROR) << "Null inmem layer";

    return boost::shared_ptr<Blob<float>>();
  }

  inmem_layer->AddDatumVector(batch);
  feature_extraction_net_->Forward();

  return feature_extraction_net_->blob_by_name("pool5/7x7_s1");
}

int MRFeatureExtraction::feature_extraction_pipeline(
  std::string pretrained_binary_proto,
  std::string feature_extraction_proto,
  std::string extract_feature_blob_names,
  std::string target_pipe_path,
  int mini_batch_size,
  Caffe::Brew caffe_mode,
  int device_id
) {
  if (caffe_mode == Caffe::GPU) {
    LOG(ERROR) << "Using GPU";

    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  boost::shared_ptr<Net<float> > feature_extraction_net(
      new Net<float>(feature_extraction_proto, caffe::TEST));
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
      const boost::shared_ptr<Blob<float> > feature_blob =
          feature_extraction_net->blob_by_name(blob_names[i]);
      int batch_size = feature_blob->num();
      int dim_features = feature_blob->count() / batch_size;
      const float* feature_blob_data;
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

static std::shared_ptr<S3Client> _s3_client;
static std::shared_ptr<Aws::Utils::RateLimits::RateLimiterInterface> _s3_limiter;

#define ALLOCATION_TAG "FoursquarePhotoCNN"

void MRFeatureExtraction::destroy_s3() {
  _s3_client = nullptr;
  _s3_limiter = nullptr;
}

void MRFeatureExtraction::initialize_s3(
  std::string access_key,
  std::string secret_key,
  std::string s3_bucket
) {
  _s3_limiter = Aws::MakeShared<Aws::Utils::RateLimits::DefaultRateLimiter<>>(ALLOCATION_TAG, 50000000);

  ClientConfiguration config;
  // config.region = Aws::Region::US_EAST_1;
  // config.scheme = Scheme::HTTPS;
  config.connectTimeoutMs = 10000;
  config.requestTimeoutMs = 10000;
  config.readRateLimiter = _s3_limiter;
  config.writeRateLimiter = _s3_limiter;
  config.maxConnections = 1;
  // config.maxErrorRetry = 5;
  config.proxyHost = "proxyout-aux-vip.prod.foursquare.com";
  config.proxyPort = 80;

  const char* cstr_access_key = access_key.c_str();
  LOG(ERROR) << "Access key " << cstr_access_key;
  const char* cstr_secret_key = secret_key.c_str();
  LOG(ERROR) << "Secret key " << cstr_secret_key;

  _s3_client = Aws::MakeShared<S3Client>(
    ALLOCATION_TAG,
    AWSCredentials(cstr_access_key, cstr_secret_key),
    config,
    false
  );
}