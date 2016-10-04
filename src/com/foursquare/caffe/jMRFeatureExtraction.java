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

  public jMRFeatureExtraction() throws Exception { }

  public native void initializeS3(String accessKey, String secretKey, String s3Bucket);

  public native void destroyS3();

  private native String _processS3Files(String[] photoIds, String[] s3Files);

  public List<Caffe.Datum> processS3Files(
    String[] photoIds,
    String[] s3Files
  ) throws Exception {
    String resultFileName = _processS3Files(photoIds, s3Files);

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

    FileInputStream resultFileStream = new FileInputStream(resultFileName);
    List<Caffe.Datum> results = new ArrayList<Caffe.Datum>();
    Caffe.Datum datum = Caffe.Datum.parseDelimitedFrom(resultFileStream);

    while (datum != null) {
      results.add(datum);

      datum = Caffe.Datum.parseDelimitedFrom(resultFileStream);
    }

    resultFileStream.close();

    new File(batchFileName).delete();
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
  private String currentToNNBatchFileNamePrefix = "/dev/shm/foursquare_pcv1_in_";
  private FileOutputStream currentToNNBatchFileStream = null;

  private Boolean init = false;
  private void init() throws Exception {
    toNNFile = new RandomAccessFile(getInputPipePath(), "rw");
    fromNNFile = new RandomAccessFile(getOutputPipePath(), "rw");
    currentToNNBatchFileStream =
      new FileOutputStream(currentToNNBatchFileNamePrefix + currentToNNBatchId);

    init = true;
  }

  public void writeDatum(Caffe.Datum datum) throws Exception {
    if (!init) {
      init();
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
    if (!init) {
      init();
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

  public void start(String pretrainedBinaryProto, String featureExtractionProto) {
    startFeatureExtraction(pretrainedBinaryProto, featureExtractionProto);
  }

  public void startAsync(String pretrainedBinaryProto, String featureExtractionProto) {
    featureExtractionThread = new Thread(new Runnable() {
      public void run() {
        try {
          featureExtractionReturnCode = runFeatureExtraction(pretrainedBinaryProto, featureExtractionProto);
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
      // Full AWS so in explicit way
      System.loadLibrary("aws-cpp-sdk-access-management");
      System.loadLibrary("aws-cpp-sdk-acm");
      System.loadLibrary("aws-cpp-sdk-apigateway");
      System.loadLibrary("aws-cpp-sdk-application-autoscaling");
      System.loadLibrary("aws-cpp-sdk-autoscaling");
      System.loadLibrary("aws-cpp-sdk-cloudformation");
      System.loadLibrary("aws-cpp-sdk-cloudfront");
      System.loadLibrary("aws-cpp-sdk-cloudhsm");
      System.loadLibrary("aws-cpp-sdk-cloudsearchdomain");
      System.loadLibrary("aws-cpp-sdk-cloudsearch");
      System.loadLibrary("aws-cpp-sdk-cloudtrail");
      System.loadLibrary("aws-cpp-sdk-codecommit");
      System.loadLibrary("aws-cpp-sdk-codedeploy");
      System.loadLibrary("aws-cpp-sdk-codepipeline");
      System.loadLibrary("aws-cpp-sdk-cognito-identity");
      System.loadLibrary("aws-cpp-sdk-cognito-idp");
      System.loadLibrary("aws-cpp-sdk-cognito-sync");
      System.loadLibrary("aws-cpp-sdk-config");
      System.loadLibrary("aws-cpp-sdk-core");
      System.loadLibrary("aws-cpp-sdk-datapipeline");
      System.loadLibrary("aws-cpp-sdk-devicefarm");
      System.loadLibrary("aws-cpp-sdk-directconnect");
      System.loadLibrary("aws-cpp-sdk-dms");
      System.loadLibrary("aws-cpp-sdk-ds");
      System.loadLibrary("aws-cpp-sdk-dynamodb");
      System.loadLibrary("aws-cpp-sdk-ec2");
      System.loadLibrary("aws-cpp-sdk-ecr");
      System.loadLibrary("aws-cpp-sdk-ecs");
      System.loadLibrary("aws-cpp-sdk-elasticache");
      System.loadLibrary("aws-cpp-sdk-elasticbeanstalk");
      System.loadLibrary("aws-cpp-sdk-elasticfilesystem");
      System.loadLibrary("aws-cpp-sdk-elasticloadbalancing");
      System.loadLibrary("aws-cpp-sdk-elasticloadbalancingv2");
      System.loadLibrary("aws-cpp-sdk-elasticmapreduce");
      System.loadLibrary("aws-cpp-sdk-elastictranscoder");
      System.loadLibrary("aws-cpp-sdk-email");
      System.loadLibrary("aws-cpp-sdk-es");
      System.loadLibrary("aws-cpp-sdk-events");
      System.loadLibrary("aws-cpp-sdk-firehose");
      System.loadLibrary("aws-cpp-sdk-gamelift");
      System.loadLibrary("aws-cpp-sdk-glacier");
      System.loadLibrary("aws-cpp-sdk-iam");
      System.loadLibrary("aws-cpp-sdk-identity-management");
      System.loadLibrary("aws-cpp-sdk-importexport");
      System.loadLibrary("aws-cpp-sdk-inspector");
      System.loadLibrary("aws-cpp-sdk-iot");
      System.loadLibrary("aws-cpp-sdk-kinesisanalytics");
      System.loadLibrary("aws-cpp-sdk-kinesis");
      System.loadLibrary("aws-cpp-sdk-kms");
      System.loadLibrary("aws-cpp-sdk-lambda");
      System.loadLibrary("aws-cpp-sdk-logs");
      System.loadLibrary("aws-cpp-sdk-machinelearning");
      System.loadLibrary("aws-cpp-sdk-marketplacecommerceanalytics");
      System.loadLibrary("aws-cpp-sdk-meteringmarketplace");
      System.loadLibrary("aws-cpp-sdk-mobileanalytics");
      System.loadLibrary("aws-cpp-sdk-monitoring");
      System.loadLibrary("aws-cpp-sdk-opsworks");
      System.loadLibrary("aws-cpp-sdk-queues");
      System.loadLibrary("aws-cpp-sdk-rds");
      System.loadLibrary("aws-cpp-sdk-redshift");
      System.loadLibrary("aws-cpp-sdk-route53domains");
      System.loadLibrary("aws-cpp-sdk-route53");
      System.loadLibrary("aws-cpp-sdk-s3");
      System.loadLibrary("aws-cpp-sdk-sdb");
      System.loadLibrary("aws-cpp-sdk-servicecatalog");
      System.loadLibrary("aws-cpp-sdk-snowball");
      System.loadLibrary("aws-cpp-sdk-sns");
      System.loadLibrary("aws-cpp-sdk-sqs");
      System.loadLibrary("aws-cpp-sdk-ssm");
      System.loadLibrary("aws-cpp-sdk-storagegateway");
      System.loadLibrary("aws-cpp-sdk-sts");
      System.loadLibrary("aws-cpp-sdk-support");
      System.loadLibrary("aws-cpp-sdk-swf");
      System.loadLibrary("aws-cpp-sdk-transfer");
      System.loadLibrary("aws-cpp-sdk-waf");
      System.loadLibrary("aws-cpp-sdk-workspaces");
      
      System.loadLibrary("caffe");
      System.loadLibrary("caffe_jni");
    } catch (Exception e) { }
  }
}
