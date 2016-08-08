package com.foursquare.caffe;

import caffe.*;
import com.google.protobuf.*;
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

  public final static int batchSize = 50;

  public jMRFeatureExtraction() throws Exception { }

  public RandomAccessFile toNNFile = new RandomAccessFile(getInputPipePath(), "rw");
  public RandomAccessFile fromNNFile = new RandomAccessFile(getOutputPipePath(), "rw");

  private int currentToNNBatchId = 0;
  private int currentToNNBatchIndex = -1;
  private String currentToNNBatchFileNamePrefix = "/dev/shm/foursquare_pcv1_in_";
  private FileOutputStream currentToNNBatchFileStream =
    new FileOutputStream(currentToNNBatchFileNamePrefix + currentToNNBatchId);

  public void writeDatum(Caffe.Datum datum) throws Exception {
    if (currentToNNBatchIndex == batchSize - 1) {
      currentToNNBatchFileStream.close();
      toNNFile.write((currentToNNBatchFileNamePrefix + currentToNNBatchId + "\n").getBytes());

      currentToNNBatchIndex = -1;
      ++currentToNNBatchId;

      currentToNNBatchFileStream =
        new FileOutputStream(currentToNNBatchFileNamePrefix + currentToNNBatchId);
    }

    datum.writeDelimitedTo(currentToNNBatchFileStream);
    ++currentToNNBatchIndex;
  }

  private String fileName = null;
  private FileInputStream currentFromNNBatchFileStream = null;
  private CodedInputStream cis = null;

  public Caffe.Datum readDatum() throws Exception {
    if (currentFromNNBatchFileStream == null) {
      if (fileName != null) {
        new File(fileName).delete();
      }

      fileName = fromNNFile.readLine();

      currentFromNNBatchFileStream = new FileInputStream(fileName);
      cis = CodedInputStream.newInstance(currentFromNNBatchFileStream);
    }

    // TODO(zen): avoid copying
    int size = cis.readRawVarint32();
    byte[] rawBytes = cis.readRawBytes(size);
    Caffe.Datum datum = Caffe.Datum.newBuilder().mergeFrom(rawBytes).build();

    if (datum == null) {
      currentFromNNBatchFileStream.close();
      currentFromNNBatchFileStream = null;

      return readDatum();
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
