
#include "mr_feature_extraction.hpp"
#include "com_foursquare_caffe_jMRFeatureExtraction.h"

MRFeatureExtraction<float> instance; // Singleton

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

JNIEXPORT void JNICALL Java_com_foursquare_caffe_jMRFeatureExtraction_stopFeatureExtraction(JNIEnv *env, jobject obj)
{
  instance.stop_feature_extraction_pipeline();
}

JNIEXPORT jstring JNICALL Java_com_foursquare_caffe_jMRFeatureExtraction_getInputPipePath(JNIEnv *env, jobject obj)
{
  const char* inputPipePath = instance.get_input_pipe_path();

  return env->NewStringUTF(inputPipePath);
}

JNIEXPORT jstring JNICALL Java_com_foursquare_caffe_jMRFeatureExtraction_getOutputPipePath(JNIEnv *env, jobject obj)
{
  const char* outputPipePath = instance.get_output_pipe_path();

  return env->NewStringUTF(outputPipePath);
}