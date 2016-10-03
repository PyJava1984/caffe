#include <fcntl.h>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/random_generator.hpp>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "caffe/util/db.hpp"
#include "caffe/proto/caffe.pb.h"

#include "mr_feature_extraction.hpp"
#include "com_foursquare_caffe_jMRFeatureExtraction.h"

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

using namespace Aws::Auth;
using namespace Aws::Http;
using namespace Aws::Client;
using namespace Aws::S3;
using namespace Aws::S3::Model;
using namespace Aws::Utils;

MRFeatureExtraction instance; // Singleton

JNIEXPORT jint JNICALL Java_com_foursquare_caffe_jMRFeatureExtraction_startFeatureExtraction(
  JNIEnv *env,
  jobject obj,
  jstring pretrained_binary_proto,
  jstring feature_extraction_proto
) {
  const char* c_pretrained_binary_proto = env->GetStringUTFChars(pretrained_binary_proto, NULL);
  const char* c_feature_extraction_proto = env->GetStringUTFChars(feature_extraction_proto, NULL);

  instance.start_feature_extraction_pipeline(c_pretrained_binary_proto, c_feature_extraction_proto);

  env->ReleaseStringUTFChars(pretrained_binary_proto, c_pretrained_binary_proto);
  env->ReleaseStringUTFChars(feature_extraction_proto, c_feature_extraction_proto);

  return reinterpret_cast<jint>(0);
}

JNIEXPORT jint JNICALL Java_com_foursquare_caffe_jMRFeatureExtraction_runFeatureExtraction(
  JNIEnv *env,
  jobject obj,
  jstring pretrained_binary_proto,
  jstring feature_extraction_proto
) {
  const char* c_pretrained_binary_proto = env->GetStringUTFChars(pretrained_binary_proto, NULL);
  const char* c_feature_extraction_proto = env->GetStringUTFChars(feature_extraction_proto, NULL);

  int ret = instance.run_feature_extraction_pipeline(c_pretrained_binary_proto, c_feature_extraction_proto);

  env->ReleaseStringUTFChars(pretrained_binary_proto, c_pretrained_binary_proto);
  env->ReleaseStringUTFChars(feature_extraction_proto, c_feature_extraction_proto);

  return reinterpret_cast<jint>(ret);
}

JNIEXPORT jstring JNICALL Java_com_foursquare_caffe_jMRFeatureExtraction_processBatch
  (JNIEnv *env, jobject obj, jstring batchFilePath)
{
  const char* batch_file_path = env->GetStringUTFChars(batchFilePath, NULL);
  int batch_file_fd = open(batch_file_path, O_RDONLY);
  google::protobuf::io::ZeroCopyInputStream *batch_file_stream =
    new google::protobuf::io::FileInputStream(batch_file_fd);
  caffe::Datum datum;
  std::vector<caffe::Datum> batch;

  while (caffe::db::read_delimited_from(batch_file_stream, &datum) == 0) {
    batch.push_back(datum);
  }

  const boost::shared_ptr<caffe::Blob<float> > feature_blob = instance.process_batch(batch);
  if (!feature_blob) {
    return env->NewStringUTF("");
  }

  int batch_size = feature_blob->num();

  std::stringstream ss;
  boost::uuids::basic_random_generator<boost::mt19937> gen;
  boost::uuids::uuid u = gen();
  const std::string u_str = boost::uuids::to_string(u);

  ss << MRFeatureExtraction::get_share_memory_fs_path()
     << '/'
     << MRFeatureExtraction::get_from_nn_batch_file_name_prefix()
     << u_str;
  std::string file_name = ss.str();
  std::ofstream output_stream(file_name.c_str(), std::ios::binary);
  google::protobuf::io::ZeroCopyOutputStream* raw_output_stream =
    new google::protobuf::io::OstreamOutputStream(&output_stream);

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

    caffe::db::write_delimited_to(datum, raw_output_stream);
  }

  delete raw_output_stream;
  output_stream.close();

  env->ReleaseStringUTFChars(batchFilePath, batch_file_path);

  return env->NewStringUTF(file_name.c_str());
}

JNIEXPORT void JNICALL Java_com_foursquare_caffe_jMRFeatureExtraction_stopFeatureExtraction(
  JNIEnv *env,
  jobject obj
) {
  instance.stop_feature_extraction_pipeline();
}

JNIEXPORT jstring JNICALL Java_com_foursquare_caffe_jMRFeatureExtraction_getInputPipePath(
  JNIEnv *env,
  jobject obj
) {
  const char* inputPipePath = instance.get_input_pipe_path();

  return env->NewStringUTF(inputPipePath);
}

JNIEXPORT jstring JNICALL Java_com_foursquare_caffe_jMRFeatureExtraction_getOutputPipePath(
  JNIEnv *env,
  jobject obj
) {
  const char* outputPipePath = instance.get_output_pipe_path();

  return env->NewStringUTF(outputPipePath);
}

static std::shared_ptr<S3Client> _s3_client;
static std::shared_ptr<Aws::Utils::RateLimits::RateLimiterInterface> _s3_limiter;

#define ALLOCATION_TAG "FoursquarePhotoCNN"

JNIEXPORT void JNICALL Java_com_foursquare_caffe_jMRFeatureExtraction_initializeS3(
  JNIEnv *env,
  jobject obj,
  jstring access_key,
  jstring secret_key,
  jstring s3_bucket
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

  const char* cstr_access_key = env->GetStringUTFChars(access_key, NULL);
  LOG(ERROR) << "Access key " << cstr_access_key;
  const char* cstr_secret_key = env->GetStringUTFChars(secret_key, NULL);
  LOG(ERROR) << "Secret key " << cstr_secret_key;

  // std::shared_ptr<DefaultAWSCredentialsProviderChain> credentials =
  //   Aws::MakeShared<DefaultAWSCredentialsProviderChain>(ALLOCATION_TAG, config, false);

  _s3_client = Aws::MakeShared<S3Client>(
    ALLOCATION_TAG,
    AWSCredentials("access_key_id", "secret_key"),
    config,
    false
  );

  env->ReleaseStringUTFChars(access_key, cstr_access_key);
  env->ReleaseStringUTFChars(secret_key, cstr_secret_key);
}

/* JNIEXPORT jstring JNICALL Java_com_foursquare_caffe_jMRFeatureExtraction__1processS3Files(
  JNIEnv *env,
  jobject obj,
  jobject photo_ids,
  jobject s3_files
) {
  jclass ArrayList_class = (*env)->FindClass(env, "java/util/ArrayList");
  const char* c_photo_id;
  const char* c_s3_file;

  // Caffe has a bug that if new width and new height are set,
  // it will use them to initial network graph, which sometimes leads to invalid graph.
  // So I hard code the height and width to 256 to match pre-trained model.

  cv::Mat cv_img =

  return env->NewStringUTF("");
}*/