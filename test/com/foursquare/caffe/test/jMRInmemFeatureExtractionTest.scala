// Copyright 2016 Foursquare Labs Inc. All Rights Reserved.

package com.foursquare.caffe.test

import caffe.Caffe.Datum
import collection.JavaConverters._
import com.foursquare.caffe.jMRFeatureExtraction
import com.google.protobuf.ByteString
import java.awt.image.{BufferedImage, DataBufferByte}
import java.io.{File, FileInputStream, FileOutputStream, PrintWriter}
import java.lang.{Runnable, Thread}
import javax.imageio.ImageIO
import scala.io.Source

object jMRInmemFeatureExtractionTestApp extends App {
  val featureExtraction = jMRFeatureExtraction.Instance()
  val fileListPath = args(0)
  val source = Source.fromFile(fileListPath)
  val fileList = source.getLines
  val resultWriter = new PrintWriter(args.lift(1).getOrElse("jMRFeatureExtractionTestApp_result.txt"))

  featureExtraction.start(
    args.lift(2).getOrElse("bvlc_googlenet.caffemodel"),
    args.lift(3).getOrElse("train_val.prototxt")
  )

  Thread.sleep(1000)

  val batches = fileList.flatMap(f => {
    try {
      val img = ImageIO.read(new File(f))
      val byteArray = img.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData
      Some(Datum.newBuilder
        .setData(ByteString.copyFrom(byteArray))
        .setHeight(img.getHeight)
        .setWidth(img.getWidth)
        .setChannels(3)
        .build())
    } catch {
      case e: Exception => {
        print(s"Failed to open $f")
        None
      }
    }
  }).grouped(50)

  batches.foreach(batch => {
    val results = featureExtraction.processBatch(batch.toIterator.asJava)

    results.asScala.foreach(featureDatum => {
      featureDatum.getFloatDataList.asScala.foreach(d => resultWriter.print(s"$d "))
      resultWriter.println
    })
  })

  source.close

  // TODO(zen): add flush method to force sync.
  val ret = featureExtraction.stop()

  println(ret)

  resultWriter.close

  println("Result writer closed")

  System.exit(0)
}
