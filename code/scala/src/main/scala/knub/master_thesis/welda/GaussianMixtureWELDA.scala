package knub.master_thesis.welda

import breeze.linalg.DenseVector
import jMEF._
import knub.master_thesis.Args

class GaussianMixtureWELDA(p: Args) extends ReplacementWELDA(p) {

    var mixtureModels: Array[MixtureModel] = _
    override def transformVector(a: Array[Double]): Array[Double] = a

    override def getFolderName(): String = {
        s"${p.modelFileName}.$embeddingName.welda.gaussian-mixture." +
            s"pca-$PCA_DIMENSIONS." +
            s"des-$DISTRIBUTION_ESTIMATION_SAMPLES." +
            s"lambda-${LAMBDA.toString.replace('.', '-')}"
    }

    override def estimateDistributionParameters(): Unit = {
        println("Estimating distribution parameters")
        val topTopicVectors = getTopTopicVectors()
        mixtureModels = topTopicVectors.map { topVectors =>
            val points = topVectors.map { v =>
                val pVector = new PVector(v.length)
                pVector.array = v
                pVector
            }.toArray
            val clusters = KMeans.run(points, 2)
            var gaussians = BregmanSoftClustering.initialize(clusters, new MultivariateGaussian)
            gaussians = BregmanSoftClustering.run(points, gaussians)
            points.foreach { point =>
                println(gaussians.mixtureNumber(point))
            }
            gaussians
        }
    }

    override def sampleFromDistribution(topicId: Int): DenseVector[Double] = {
        new DenseVector(mixtureModels(topicId).drawRandomPoints(1)(0).array)
    }

}
