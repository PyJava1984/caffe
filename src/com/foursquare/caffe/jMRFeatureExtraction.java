package com.foursquare.caffe;

import caffe.Caffe.Datum;
import java.io.File;
import java.lang.*;

public class jMRFeatureExtraction {
  private native int startFeatureExtraction(String pretrainedBinaryProto, String featureExtractionProto);

  private native void stopFeatureExtraction();

  private Thread featureExtractionThread = null;
  private volatile int featureExtractionReturnCode = -1;

  public native String getInputPipePath();
  public native String getOutputPipePath();

  public const String shareMemoryFsPath = "/dev/shm";
  public const int batchSize = 50;

  public PrintWriter toNNWriter = new PrintWriter(new FileOutputStream(getInputPipePath()));
  public Scanner fromNNReader = new Scanner(new FileInputStream(getOutputPipePath()));

  private int currentToNNBatchId = 0;
  private int currentToNNBatchIndex = -1;
  private String currentToNNBatchFileNamePrefix = "foursquare_pcv1_in_";
  private FileOutputStream currentToNNBatchFileStream =
    new FileOutputStream(currentToNNBatchFileNamePrefix + currentToNNBatchId);

  public void writeDatum(Datum datum) {
    if (currentToNNBatchIndex == batchSize - 1) {
      toNNWriter.println(currentToNNBatchFileNamePrefix + currentToNNBatchId);

      currentToNNBatchIndex = -1;
      ++currentToNNBatchId;

      currentToNNBatchFileStream.close();
      currentToNNBatchFileStream =
        new FileOutputStream(currentToNNBatchFileNamePrefix + currentToNNBatchId);
    }

    datum.writeTo(currentToNNBatchFileStream)
    ++currentToNNBatchIndex;
  }

  private int currentFromNNBatchIndex = -1;
  private String currentFromNNBatchFileNamePrefix = "foursquare_pcv1_out_";
  private FileInputStream currentFromNNBatchFileStream =
    new FileInputStream(currentFromNNBatchFileNamePrefix + currentFromNNBatchId);

  public Datum readDatum() {
    if (currentFromNNBatchIndex == batchSize - 1) {
      String fileName = fromNNReader.nextLine();

      currentFromNNBatchIndex = -1;

      currentFromNNBatchFileStream.close();
      currentFromNNBatchFileStream = new FileInputStream(fileName);
    }

    Datum datum = Datum.parseFrom(currentFromNNBatchFileStream);

    if (datum == null) {
      currentFromNNBatchIndex = batchSize - 1;
    } else {
      ++currentFromNNBatchIndex;
    }

    return datum;
  }

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
