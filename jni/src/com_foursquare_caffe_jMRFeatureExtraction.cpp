#include <fcntl.h>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/random_generator.hpp>

#include <opencv2/core/core.hpp>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "caffe/blob.hpp"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#include "mr_feature_extraction.hpp"
#include "com_foursquare_caffe_jMRFeatureExtraction.h"

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

jstring save_batch(
  JNIEnv *env,
  const std::vector<boost::shared_ptr<caffe::Blob<float>>> feature_blobs,
  const std::vector<std::string>& new_ids
) {
  if (feature_blobs.empty()) {
    return env->NewStringUTF("");
  }

  int batch_size = feature_blobs[0]->num();
  for (int i = 1; i < feature_blobs.size(); ++i) {
    if (feature_blobs[i]->num() != batch_size) {
      return env->NewStringUTF("");
    }
  }

  if (!new_ids.empty() && new_ids.size() != batch_size) {
    throw std::runtime_error("New Id size must be batch size");
  }

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

  caffe::Datum datum;
  const float* feature_blob_data;
  for (int n = 0; n < batch_size; ++n) {
    if (!new_ids.empty() && new_ids[n] == "") {
      break;
    }



    for (int m = 0; m < feature_blobs.size(); ++m) {
      boost::shared_ptr<caffe::Blob<float>> feature_blob = feature_blobs[m];

      datum.set_height(feature_blob->height());
      datum.set_width(feature_blob->width());
      datum.set_channels(feature_blob->channels());
      datum.clear_data();
      datum.clear_float_data();

      int dim_features = feature_blob->count() / batch_size;

      feature_blob_data = feature_blob->cpu_data() +
                          feature_blob->offset(n);

      datum.add_float_data(dim_features); // TODO(zen): add dedicated column in Datum to pass size.

      for (int d = 0; d < dim_features; ++d) {
        datum.add_float_data(feature_blob_data[d]);
      }
    }


    caffe::db::write_delimited_to(datum, raw_output_stream);
  }

  delete raw_output_stream;
  output_stream.close();

  return env->NewStringUTF(file_name.c_str());
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

  env->ReleaseStringUTFChars(batchFilePath, batch_file_path);

  static const char *arr[] = {
    "pool5/7x7_s1",
    "prob"
  };
  std::vector<std::string> layers(arr, std::end(arr));
  const std::vector<boost::shared_ptr<caffe::Blob<float>>> feature_blobs = instance.process_batch(
    batch,
    layers
  );

  if (feature_blobs.size() == layers.size()) {
    return save_batch(env, feature_blobs, std::vector<std::string>());
  } else {
    return env->NewStringUTF("");
  }
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
  const std::string inputPipePath = instance.get_input_pipe_path();

  return env->NewStringUTF(inputPipePath.c_str());
}

JNIEXPORT jstring JNICALL Java_com_foursquare_caffe_jMRFeatureExtraction_getOutputPipePath(
  JNIEnv *env,
  jobject obj
) {
  const std::string outputPipePath = instance.get_output_pipe_path();

  return env->NewStringUTF(outputPipePath.c_str());
}

JNIEXPORT jstring JNICALL Java_com_foursquare_caffe_jMRFeatureExtraction__1resizeRawImage(
  JNIEnv *env, 
  jobject obj, 
  jbyteArray rawImage
) {
  int len = env->GetArrayLength(rawImage);
  std::vector<unsigned char> buf(len);

  env->GetByteArrayRegion(rawImage, 0, len, reinterpret_cast<jbyte*>(&buf[0]));
  
  cv::Mat downSampledImage = caffe::ReadImageBufferToCVMat(buf, 256, 256, true);

  if (downSampledImage.data && downSampledImage.channels() == 3) {
    caffe::Datum datum;
    CVMatToDatum(downSampledImage, &datum);

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

    caffe::db::write_to(datum, raw_output_stream);

    delete raw_output_stream;
    output_stream.close();

    return env->NewStringUTF(file_name.c_str());
  } else {
    return env->NewStringUTF("");
  }
}
