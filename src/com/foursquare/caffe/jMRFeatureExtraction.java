package com.foursquare.caffe;

import caffe.*;
import java.io.*;
import java.lang.*;
import java.util.*;

public class jMRFeatureExtraction {
  private native int startFeatureExtraction(String pretrainedBinaryProto, String featureExtractionProto);

  private native void stopFeatureExtraction();

  private Thread featureExtractionThread = null;
  private volatile int featureExtractionReturnCode = -1;

  public native String getInputPipePath();
  public native String getOutputPipePath();

  public final static String shareMemoryFsPath = "/dev/shm";
  public final static int batchSize = 50;

  public jMRFeatureExtraction() throws Exception { }

  public RandomAccessFile toNNFile = new RandomAccessFile(getInputPipePath(), "rw");
  public RandomAccessFile fromNNFile = new RandomAccessFile(getOutputPipePath(), "rw");

  private int currentToNNBatchId = 0;
  private int currentToNNBatchIndex = -1;
  private String currentToNNBatchFileNamePrefix = "foursquare_pcv1_in_";
  private FileOutputStream currentToNNBatchFileStream =
    new FileOutputStream(shareMemoryFsPath + "/" + currentToNNBatchFileNamePrefix + currentToNNBatchId);

  public void writeDatum(Caffe.Datum datum) throws Exception {
    if (currentToNNBatchIndex == batchSize - 1) {
      toNNFile.write((currentToNNBatchFileNamePrefix + currentToNNBatchId + "\n").getBytes());

      currentToNNBatchIndex = -1;
      ++currentToNNBatchId;

      currentToNNBatchFileStream.close();
      currentToNNBatchFileStream =
        new FileOutputStream(currentToNNBatchFileNamePrefix + currentToNNBatchId);
    }

    datum.writeTo(currentToNNBatchFileStream);
    ++currentToNNBatchIndex;
  }

  private int currentFromNNBatchIndex = -1;
  private String currentFromNNBatchFileNamePrefix = "foursquare_pcv1_out_";
  private FileInputStream currentFromNNBatchFileStream = null;

  public Caffe.Datum readDatum() throws Exception {
    if (currentFromNNBatchFileStream == null || currentFromNNBatchIndex == batchSize - 1) {
      String fileName = fromNNFile.readLine();

      currentFromNNBatchIndex = -1;

      currentFromNNBatchFileStream.close();
      currentFromNNBatchFileStream = new FileInputStream(fileName);
    }

    Caffe.Datum datum = Caffe.Datum.parseFrom(currentFromNNBatchFileStream);

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
