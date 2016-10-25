package knub.master_thesis.welda

import java.io.File
import java.util

import breeze.linalg.DenseVector
import jMEF._
import knub.master_thesis.Args
import org.apache.commons.exec.{CommandLine, DefaultExecutor}
import org.apache.commons.io.FileUtils
import weka.clusterers.EM
import weka.core.{Attribute, DenseInstance, Instance, Instances}

import scala.collection.JavaConverters._

class GaussianMixtureWELDA(p: Args) extends ReplacementWELDA(p) {

    override def transformVector(a: Array[Double]): Array[Double] = a

    override def getFolderName(): String = {
        s"${p.modelFileName}.$embeddingName.welda.gaussian-mixture." +
            s"pca-$PCA_DIMENSIONS." +
            s"des-$DISTRIBUTION_ESTIMATION_SAMPLES." +
            s"lambda-${LAMBDA.toString.replace('.', '-')}"
    }

    /**
      * WITH WEKA: Does not work, only one-dimensional
      */

//    override def estimateDistributionParameters(): Unit = {
//        val topTopicVectors = getTopTopicVectorsWithWords()
//        topTopicVectors.map { topVectors =>
//            val attributes = new util.ArrayList[Attribute](PCA_DIMENSIONS)
//            for (i <- 0 until PCA_DIMENSIONS)
//                attributes.add(new Attribute(s"v$i"))
//            val instances = new Instances("words", attributes, DISTRIBUTION_ESTIMATION_SAMPLES)
//            topVectors.foreach { case (word, a) =>
//                instances.add(new DenseInstance(1.0, a))
//            }
//            val gaussians = new EM
//            gaussians.setNumClusters(-1)
//            gaussians.setSeed(21011991)
//
//            gaussians.buildClusterer(instances)
//            println(gaussians.toString)
//            null
//        }
//    }
    /**
      * WITH scikit
      */
    var mixtureModels: Array[GaussianMixture] = _
    override def estimateDistributionParameters(): Unit = {
        println("Estimating distribution parameters")

        val iterationFolder = new File(s"$folder/$currentIteration")
        iterationFolder.mkdir()

        val topTopicVectors = getTopTopicVectorsWithWords()
        topTopicVectors.zipWithIndex.map { case (topVectors, idx) =>
            val topicFile = new File(s"${iterationFolder.getAbsolutePath}/$idx")
            println(topicFile.getAbsolutePath)
            val linesInFile = topVectors.map { case (w, v) =>
                s"$w\t${v.mkString("\t")}"
            }
            FileUtils.writeLines(topicFile, "UTF-8", linesInFile.asJava)

            val cl = new CommandLine("/opt/anaconda3/envs/py27/bin/python")
            cl.addArgument("gaussian_mixture.py")
            cl.addArgument(topicFile.getAbsolutePath)

            val exec = new DefaultExecutor
            exec.setExitValue(0)
            exec.execute(cl)

//            val points = topVectors.map { case (w, v) =>
//                val pVector = new PVector(v.length)
//                pVector.array = v
//                pVector
//            }.toArray
//            val clusters = KMeans.run(points, 2)
//            var gaussians = BregmanSoftClustering.initialize(clusters, new MultivariateGaussian)
//            gaussians = BregmanSoftClustering.run(points, gaussians)
//            //            points.foreach { point =>
//            //                println(gaussians.mixtureNumber(point))
//            //            }
//            gaussians.param
//            gaussians
            null
        }
        System.exit(1)
    }

    override def sampleFromDistribution(topicId: Int): DenseVector[Double] = {
        null
    }

    /**
      * WITH jMEF
      */
//    var mixtureModels: Array[MixtureModel] = _
//    override def estimateDistributionParameters(): Unit = {
//        println("Estimating distribution parameters")
//        val topTopicVectors = getTopTopicVectorsWithWords()
//        mixtureModels = topTopicVectors.map { topVectors =>
//            val points = topVectors.map { case (w, v) =>
//                val pVector = new PVector(v.length)
//                pVector.array = v
//                pVector
//            }.toArray
//            val clusters = KMeans.run(points, 2)
//            var gaussians = BregmanSoftClustering.initialize(clusters, new MultivariateGaussian)
//            gaussians = BregmanSoftClustering.run(points, gaussians)
////            points.foreach { point =>
////                println(gaussians.mixtureNumber(point))
////            }
//            gaussians.param
//            gaussians
//        }
//    }
//
//    override def sampleFromDistribution(topicId: Int): DenseVector[Double] = {
//        val a = mixtureModels(topicId).drawRandomPoints(1)(0).array
//        a.foreach { v =>
//            assert(!v.isNaN, s"sampled value is nan in ${a.deep.toString}")
//        }
//        new DenseVector(a)
//    }

}
