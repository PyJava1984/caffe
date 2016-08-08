// Copyright 2016 Foursquare Labs Inc. All Rights Reserved.

package com.foursquare.caffe.test

import caffe.Caffe.Datum
import collection.JavaConverters._
import com.foursquare.caffe.jMRFeatureExtraction
import com.google.protobuf.ByteString
import java.awt.image.{BufferedImage, DataBufferByte}
import java.io.{File, FileInputStream, FileOutputStream}
import java.lang.{Runnable, Thread}
import javax.imageio.ImageIO
import scala.io.Source

object jMRFeatureExtractionTestApp extends App {
  val featureExtraction = new jMRFeatureExtraction
  val fileListPath = args(0)
  val source = Source.fromFile(fileListPath)
  val fileList = source.getLines
  val resultWriter = new PrintWriter(args.lift(1).getOrElse("jMRFeatureExtractionTestApp_result.txt"))

  featureExtraction.start(
    args.lift(2).getOrElse("bvlc_googlenet.caffemodel"),
    args.lift(3).getOrElse("train_val.prototxt")
  )

  val worker = new Thread(new Runnable() {
    void run() {
      val featureDatum = featureExtraction.readDatum()

      featureDarum.getFloatDataList.asScala.foreach(d => resultWriter.println(s"$d "))
    }
  })

  worker.start()

  fileList.foreach(f => {
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

  // TODO(zen): add flush method to force sync.
  val ret = featureExtraction.stop()

  println(ret)

  worker.interrupt()

  resultWriter.close
  source.close

  inputStream.close
  outputStream.close
}
