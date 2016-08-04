package com.foursquare.caffe;

import java.io.File;
import java.lang.*;

public class jMRFeatureExtraction {
  private native int startFeatureExtraction(String pretrainedBinaryProto, String featureExtractionProto);

  private native void stopFeatureExtraction();

  private Thread featureExtractionThread = null;
  private volatile int featureExtractionReturnCode = -1;

  public native String getInputPipePath();
  public native String getOutputPipePath();

  public void start(String pretrainedBinaryProto, String featureExtractionProto) {
    featureExtractionThread = new Thread(new Runnable() {
      public void run() {
        featureExtractionReturnCode = startFeatureExtraction(pretrainedBinaryProto, featureExtractionProto);
      }
    });

    featureExtractionThread.start();
  }

  public int stop() {
    stopFeatureExtraction();

    try {
      featureExtractionThread.join(5000);
    } catch(Exception e) {
      return -1;
    }

    return featureExtractionReturnCode;
  }

  static {
    File jar = new File(jMRFeatureExtraction.class.getProtectionDomain().getCodeSource().getLocation().getPath());
    System.load(jar.getParentFile().toURI().resolve("libcaffe_jni.so").getPath());
  }
}
