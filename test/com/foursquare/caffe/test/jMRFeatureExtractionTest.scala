// Copyright 2016 Foursquare Labs Inc. All Rights Reserved.

package com.foursquare.caffe.test

import caffe.Caffe.Datum
import collection.JavaConverters._
import com.foursquare.caffe.jMRFeatureExtraction
import com.google.protobuf.ByteString
import java.io.{File, FileInputStream, FileOutputStream}
import java.awt.image.{BufferedImage, DataBufferByte}
import javax.imageio.ImageIO
import scala.io.Source

object jMRFeatureExtractionTestApp extends App {
  val featureExtraction = new jMRFeatureExtraction

  featureExtraction.start(args(1), args(2))

  val fileListPath = args(0)
  val source = Source.fromFile(fileListPath)
  val fileList = source.getLines

  (0 until 1).foreach(idx => {
    val f = fileList.next.split(' ')(0)
    val img = ImageIO.read(new File(f))
    val byteArray = img.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
    val datum = Datum.newBuilder
      .setData(ByteString.copyFrom(byteArray))
      .setHeight(img.getHeight)
      .setWidth(img.getWidth)
      .setChannels(3)
      .build()

    featureExtraction.writeDatum(datum)
  })

  source.close

  // TODO(zen): add flush method to force sync.
  val ret = featureExtraction.stop()

  val featureDatum = featureExtraction.readDatum()

  featureDatum.getFloatDataList.asScala.foreach(d => println(s"$d "))

  println(ret)

  inputStream.close()
  outputStream.close()
}
