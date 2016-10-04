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
#include "caffe/proto/caffe.pb.h"

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
  const boost::shared_ptr<caffe::Blob<float> > feature_blob,
  const std::vector<std::string>& new_ids
) {
  if (!feature_blob) {
    return env->NewStringUTF("");
  }

  int batch_size = feature_blob->num();

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
  int dim_features = feature_blob->count() / batch_size;
  const float* feature_blob_data;
  for (int n = 0; n < batch_size; ++n) {
    if (!new_ids.empty() && new_ids[n] == "") {
      break;
    }

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

  const boost::shared_ptr<caffe::Blob<float> > feature_blob = instance.process_batch(batch);

  return save_batch(env, feature_blob, std::vector<std::string>());
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

JNIEXPORT void JNICALL Java_com_foursquare_caffe_jMRFeatureExtraction_destroyS3(
    JNIEnv *env,
jobject obj
) {
  instance.destroy_s3();
}

JNIEXPORT void JNICALL Java_com_foursquare_caffe_jMRFeatureExtraction_initializeS3(
  JNIEnv *env,
  jobject obj,
  jstring access_key,
  jstring secret_key,
  jstring s3_bucket
) {
  const char* cstr_access_key = env->GetStringUTFChars(access_key, NULL);
  LOG(ERROR) << "Access key " << cstr_access_key;
  const char* cstr_secret_key = env->GetStringUTFChars(secret_key, NULL);
  LOG(ERROR) << "Secret key " << cstr_secret_key;

  instance.initialize_s3();

  env->ReleaseStringUTFChars(access_key, cstr_access_key);
  env->ReleaseStringUTFChars(secret_key, cstr_secret_key);
}

const int CaffeModelBatchSize = 50;

JNIEXPORT jstring JNICALL Java_com_foursquare_caffe_jMRFeatureExtraction__1processS3Files(
  JNIEnv *env,
  jobject obj,
  jobjectArray photo_ids,
  jobjectArray s3_files
) {
  int photo_id_count = env->GetArrayLength(photo_ids);
  int s3_file_count = env->GetArrayLength(s3_files);

  if (photo_id_count > CaffeModelBatchSize) {
    throw std::runtime_error("The count of photo id must not be greater than caffe model batch size");
  }

  if (photo_id_count != s3_file_count) {
    throw std::runtime_error("The count of photo id must be equal to the count of s3 file");
  }

  cv::Mat good_image;
  std::vector<caffe::Datum> batch;
  std::vector<std::string> new_ids;
  int i = 0;

  for (; i < photo_id_count; i++) {
    jstring str_photo_id = (jstring)(env->GetObjectArrayElement(photo_ids, i));
    jstring str_s3_file = (jstring)(env->GetObjectArrayElement(s3_files, i));
    const char* photo_id = env->GetStringUTFChars(str_photo_id, NULL);
    const char* s3_file = env->GetStringUTFChars(str_s3_file, NULL);

    std::ostringstream image_stream = instance.get_image_stream_s3(photo_id, s3_file);
    if (image_stream.eof()) {
      // Caffe has a bug that if new width and new height are set,
      // it will use them to initial network graph, which sometimes leads to invalid graph.
      // So I hard code the height and width to 256 to match pre-trained model.

      good_image = caffe::ReadImageBufferToCVMat(image_stream, 256, 256, true);

      if (good_image.data != NULL) {
        caffe::Datum datum;
        CVMatToDatum(good_image, &datum);

        batch.push_back(datum);
        new_ids.push_back(photo_id);
      }
    }

    env->ReleaseStringUTFChars(str_photo_id, photo_id);
    env->ReleaseStringUTFChars(str_s3_file, s3_file);
  }

  if (good_image.data == NULL) {
    return env->NewStringUTF("");
  } else {
    caffe::Datum datum;
    CVMatToDatum(good_image, &datum);

    for ( ; i < CaffeModelBatchSize; ++i) {
      batch.push_back(datum);
      new_ids.push_back("");
    }
  }

  const boost::shared_ptr<caffe::Blob<float> > feature_blob = instance.process_batch(batch);

  return save_batch(env, feature_blob, new_ids);
}
