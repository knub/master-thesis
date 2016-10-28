package knub.master_thesis.welda

import java.util
import java.util.Date

import breeze.linalg.{DenseVector, NotConvergedException}
import breeze.stats.distributions.MultivariateGaussian
import knub.master_thesis.Args
import knub.master_thesis.util.Sampler
import weka.clusterers.XMeans
import weka.core._

import scala.collection.mutable

case class GaussianMixture(mixture: Array[Double], gaussians: Array[MultivariateGaussian])

class GaussianKMeansWELDA(p: Args) extends GaussianWELDA(p) {

    override val PCA_DIMENSIONS = 6

    var gaussianMixtures: Array[GaussianMixture] = _

    override def getFolderName(): String = {
        s"${p.modelFileName}.$embeddingName.welda.gaussian-top.FOO." +
            s"distance-${p.weldaDistanceFunction}." +
            s"lambda-${p.lambda.toString.replace('.', '-')}"
    }

    override def getTopTopicVectors(): Array[Seq[Array[Double]]] = {
        val topTopicVectors = super.getTopTopicVectors()
        topTopicVectors.map { topicVectors =>
            val words = topicVectors.map { vector =>
                val sumDistance = topicVectors.map { otherVector =>
                    Math.sqrt(vector.zip(otherVector).map { case (a, b) => (a - b) * (a - b) }.sum)
                }.sum
                (vector, sumDistance)
            }.sortBy(_._2)
            words
                .take(10)
                .map(_._1)
        }
    }


//    override def estimateDistributionParameters() = {
//        val topTopicVectors = getTopTopicVectors()
//
//        val clusterCount = mutable.Map[Int, Int]().withDefaultValue(0)
//        gaussianMixtures = topTopicVectors.indices.map { topicId =>
//            val topicVectors = topTopicVectors(topicId)
//            val (xmeans, instances) = clusterWords(topicVectors)
//
//            val NUM_CLUSTERS = xmeans.numberOfClusters()
//            clusterCount(NUM_CLUSTERS) += 1
//            var mixture = new Array[Double](NUM_CLUSTERS)
//            val assignments = (0 until instances.size()).map { i => xmeans.clusterInstance(instances.get(i)) }
//            assignments.foreach { a =>
//                mixture(a) += 1.0
//            }
//            if (NUM_CLUSTERS > 1)
//                println(s"\tTopic $topicId: ${mixture.deep}")
//
//            val gaussians = mixture.indices.flatMap { i =>
//                val vectors = topicVectors.zip(assignments).filter(_._2 == i).map(_._1)
//                val (meanVector, covarianceMatrix) = determineGaussianParameters(vectors)
//                try {
//                    Some(mixture(i), new MultivariateGaussian(meanVector, covarianceMatrix))
//                } catch {
//                    case e: NotConvergedException =>
//                        println(s"\tTopic $topicId-$i: ERROR - Did not converge for ${vectors.size} points")
//                        None
//                }
//            }.toArray
//            mixture = gaussians.map(_._1)
//            mixture.indices.foreach { i =>
//                mixture(i) /= mixture.sum
//            }
//            assert(Math.abs(mixture.sum - 1.0) < 0.0001, s"mixture should add to 1 but adds to ${mixture.sum}")
//            GaussianMixture(mixture, gaussians.map(_._2))
//        }.toArray
//        println(
//            s"\t\t${new Date}: " +
//                clusterCount.toList.sortBy(_._1).map { case (clusters, count) => s"$clusters cluster/s: $count"}.mkString(", "))
//    }

    def clusterWords(topicVectors: Seq[Array[Double]]): (XMeans, Instances) = {
        val attributes = new util.ArrayList[Attribute]()
        for (i <- 0 until PCA_DIMENSIONS) {
            attributes.add(new Attribute(i.toString))
        }
        val instances = new Instances("vectors", attributes, DISTRIBUTION_ESTIMATION_SAMPLES)
        topicVectors.foreach { vector =>
            val inst = new DenseInstance(1.0, vector)
            instances.add(inst)
        }
        val xmeans = new XMeans()
        xmeans.setMaxIterations(10)
        xmeans.setMinNumClusters(1)
        xmeans.setMaxNumClusters(4)
        xmeans.buildClusterer(instances)

        (xmeans, instances)
    }

//    override def sampleFromDistribution(topicId: Int): DenseVector[Double] = {
//        val gaussianMixture = gaussianMixtures(topicId)
//        val mixture = Sampler.nextDiscrete(gaussianMixture.mixture)
//        gaussianMixture.gaussians(mixture).sample()
//    }

}
