// Copyright 2016 Foursquare Labs Inc. All Rights Reserved.

package com.foursquare.caffe.test

import caffe.Caffe.Datum
import caffe.Foursquare.SerializedMessage
import collection.JavaConverters._
import com.foursquare.caffe.jMRFeatureExtraction
import com.google.protobuf.ByteString
import java.io.{File, FileInputStream, FileOutputStream}
import java.awt.image.{BufferedImage, DataBufferByte}
import javax.imageio.ImageIO
import scala.io.Source

object jMRFeatureExtractionTestApp extends App {
  val featureExtraction = new jMRFeatureExtraction
  val outputStream = new FileOutputStream(featureExtraction.getOutputPipePath)
  val inputStream = new FileInputStream(featureExtraction.getInputPipePath)

  featureExtraction.start("", "")

  val fileListPath = args(0)
  val source = Source.fromFile(fileListPath)
  val fileList = source.getLines.toVector
  source.close

  val img = ImageIO.read(new File(fileList.head))
  val byteArray = img.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
  val datum = Datum.newBuilder
    .setData(ByteString.copyFrom(byteArray))
    .setHeight(img.getHeight)
    .setWidth(img.getWidth)
    .setChannels(3)
    .build()
  val serializedDatum = datum.toByteString.toStringUtf8
  val serializedMessage = SerializedMessage.newBuilder
    .setSerializedMessage(serializedDatum)
    .build()

  serializedMessage.writeTo(outputStream)

  // TODO(zen): add flush method to force sync.
  val ret = featureExtraction.stop()

  val serializedResult = SerializedMessage.parseFrom(inputStream)
  val serializedFeatures = serializedResult.getSerializedMessage
  val featureDatum = Datum.parseFrom(ByteString.copyFromUtf8(serializedFeatures))

  featureDatum.getFloatDataList.asScala.foreach(d => println(s"$d "))

  println(ret)

  inputStream.close()
  outputStream.close()
}
