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

  public Caffe.Datum readDatum() throws Exception {
    if (currentFromNNBatchFileStream == null) {
      if (fileName != null) {
        new File(fileName).delete();
      }

      fileName = fromNNFile.readLine();

      currentFromNNBatchFileStream = new FileInputStream(fileName);
    }

    Caffe.Datum datum = Caffe.Datum.parseDelimitedFrom(currentFromNNBatchFileStream);

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
        try {
          featureExtractionReturnCode = startFeatureExtraction(pretrainedBinaryProto, featureExtractionProto);
        } catch(Exception e) {
          featureExtractionReturnCode = 1;
        }
      }
    });

    featureExtractionThread.start();
  }

  public int stop() throws Exception {
    stopFeatureExtraction();
    Thread.sleep(1000);

    try {
      featureExtractionThread.join(5000);
    } catch(Exception e) {
      return -1;
    } finally { 
      System.err.println("Closing pipes");

      toNNFile.close();
      fromNNFile.close();
    }

    return featureExtractionReturnCode;
  }

  static {
    try {
      System.load(new File("libcaffe.so").getCanonicalPath());
      System.load(new File("libcaffe_jni.so").getCanonicalPath());
    } catch (Exception e) { }
  }
}
