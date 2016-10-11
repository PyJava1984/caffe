package com.foursquare.caffe;

import caffe.*;
import com.google.protobuf.*;
import java.io.*;
import java.lang.*;
import java.util.*;

public class jMRFeatureExtraction {
  private native int startFeatureExtraction(String pretrainedBinaryProto, String featureExtractionProto);

  private native int runFeatureExtraction(String pretrainedBinaryProto, String featureExtractionProto);

  private native String processBatch(String batchFilePath);

  private native void stopFeatureExtraction();

  private Thread featureExtractionThread = null;
  private volatile int featureExtractionReturnCode = -1;

  public native String getInputPipePath();
  public native String getOutputPipePath();

  public final static int batchSize = 50;

  protected jMRFeatureExtraction() throws Exception { }

  private static jMRFeatureExtraction _instance = null;
  public static jMRFeatureExtraction Instance() throws Exception {
    if (_instance == null) {
      _instance = new jMRFeatureExtraction();
    }

    return _instance;
  }

  private native String _resizeRawImage(byte[] rawImage);

  public static byte[] readFully(InputStream imageStream) throws IOException {
    byte[] buffer = new byte[16 * 1024];
    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

    int read;
    while ((read = imageStream.read(buffer)) > 0)
    {
      outputStream.write(buffer, 0, read);
    }

    return outputStream.toByteArray();
  }

  public Caffe.Datum resizeRawImage(InputStream imageStream) {
    try {
      String resultFileName = _resizeRawImage(readFully(imageStream));

      if (resultFileName == "") {
        return null;
      }

      FileInputStream resultFileStream = new FileInputStream(resultFileName);
      Caffe.Datum datum = Caffe.Datum.parseFrom(resultFileStream);

      resultFileStream.close();

      new File(resultFileName).delete();

      return datum;
    } catch (Exception ex) {
      return null;
    }
  }

  public List<Caffe.Datum> processBatch(Iterator<Caffe.Datum> batch) throws Exception {
    String batchFileName = currentToNNBatchFileNamePrefix + UUID.randomUUID();
    FileOutputStream batchStream = new FileOutputStream(batchFileName);
    int batchSize = 0;

    while (batch.hasNext()) {
      Caffe.Datum datum = batch.next();
      datum.writeDelimitedTo(batchStream);
      ++batchSize;
    }

    batchStream.close();

    int retry = 0;
    String resultFileName = processBatch(batchFileName);
    while ((batchFileName == null || batchFileName == "") && retry++ < 10) {
      Thread.sleep(1000);
      resultFileName = processBatch(batchFileName);
    }

    if (retry == 10) {
      throw new Exception("Failed to process batch");
    }

    new File(batchFileName).delete();

    FileInputStream resultFileStream = new FileInputStream(resultFileName);
    List<Caffe.Datum> results = new ArrayList<Caffe.Datum>();
    Caffe.Datum datum = Caffe.Datum.parseDelimitedFrom(resultFileStream);

    while (datum != null) {
      results.add(datum);

      datum = Caffe.Datum.parseDelimitedFrom(resultFileStream);
    }

    resultFileStream.close();

    new File(resultFileName).delete();

    if (results.size() != batchSize) {
      throw new Exception("Input size and output size do not match.");
    }

    return results;
  }

  public RandomAccessFile toNNFile = null;
  public RandomAccessFile fromNNFile = null;

  private int currentToNNBatchId = 0;
  private int currentToNNBatchIndex = -1;
  private String currentToNNBatchFileNamePrefix =
    "/dev/shm/foursquare_pcv1_in_" + UUID.randomUUID().toString() + "_";
  private FileOutputStream currentToNNBatchFileStream = null;

  private Boolean _init = false;
  private void init() throws Exception {
    toNNFile = new RandomAccessFile(getInputPipePath(), "rw");
    fromNNFile = new RandomAccessFile(getOutputPipePath(), "rw");
    currentToNNBatchFileStream =
      new FileOutputStream(currentToNNBatchFileNamePrefix + currentToNNBatchId);

    _init = true;
  }

  public void writeDatum(Caffe.Datum datum) throws Exception {
    if (!_init) {
      init();
      _init = true;
    }

    if (currentToNNBatchIndex == batchSize - 1) {
      currentToNNBatchFileStream.close();
      toNNFile.write((currentToNNBatchFileNamePrefix + currentToNNBatchId + '\n').getBytes());

      currentToNNBatchIndex = -1;
      ++currentToNNBatchId;

      // Throttling, never pile up more than 30 batches in share memory
      while (currentToNNBatchId - currentFromNNBatchId > 30) {
        Thread.sleep(100);
      }

      currentToNNBatchFileStream =
        new FileOutputStream(currentToNNBatchFileNamePrefix + currentToNNBatchId);
    }

    datum.writeDelimitedTo(currentToNNBatchFileStream);
    ++currentToNNBatchIndex;
  }

  private String fileName = null;
  private FileInputStream currentFromNNBatchFileStream = null;
  private int currentFromNNBatchId = -1;

  public Caffe.Datum readDatum() throws Exception {
    if (!_init) {
      init();
      _init = true;
    }

    if (currentFromNNBatchFileStream == null) {
      if (fileName != null) {
        new File(fileName).delete();
      }

      fileName = fromNNFile.readLine();
      
      currentFromNNBatchFileStream = new FileInputStream(fileName);

      String[] parts = fileName.split("_");
      currentFromNNBatchId = Integer.parseInt(parts[parts.length - 1]);
    }

    Caffe.Datum datum = Caffe.Datum.parseDelimitedFrom(currentFromNNBatchFileStream);

    if (datum == null) {
      currentFromNNBatchFileStream.close();
      currentFromNNBatchFileStream = null;

      return readDatum();
    }

    return datum;
  }

  private Boolean started = false;
  public void start(String pretrainedBinaryProto, String featureExtractionProto) {
    if (!started) {
      startFeatureExtraction(pretrainedBinaryProto, featureExtractionProto);
      started = true;
    }
  }

  private Boolean asyncStarted = false;
  public void startAsync(String pretrainedBinaryProto, String featureExtractionProto) {
    if (!asyncStarted) {
      featureExtractionThread = new Thread(new Runnable()
      {
        public void run() {
          try {
            featureExtractionReturnCode = runFeatureExtraction(pretrainedBinaryProto, featureExtractionProto);
          } catch (Exception e) {
            featureExtractionReturnCode = 1;
          }
        }
      });

      featureExtractionThread.start();
      asyncStarted = true;
    }
  }

  public void stop() {
    if (started) {
      stopFeatureExtraction();
      started = false;
    }
  }

  public int stopAsync() throws Exception {
    if (asyncStarted) {
      // As we have no control of the last batch size here, we cannot wait it to reach 0.
      if (currentToNNBatchId - currentFromNNBatchId > 1) {
        Thread.sleep(100);
      }

      if (currentToNNBatchId - currentFromNNBatchId == 1) {
        // Wait 1 second for the last batch, but should not block.
        Thread.sleep(1000);
      }

      stopFeatureExtraction();

      try {
        featureExtractionThread.join(1000);
      } catch (Exception e) {
        asyncStarted = false;

        return -1;
      } finally {
        System.err.println("Closing pipes");

        toNNFile.close();
        fromNNFile.close();
      }

      asyncStarted = false;

      if (currentToNNBatchId != currentFromNNBatchId) {
        throw new Exception("Last batch is left");
      }

      return featureExtractionReturnCode;
    } else {
      return 1;
    }
  }

  static {
    try {
      System.loadLibrary("caffe");
      System.loadLibrary("caffe_jni");
    } catch (Exception e) { }
  }
}
