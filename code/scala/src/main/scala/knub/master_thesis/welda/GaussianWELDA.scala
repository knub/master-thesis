package knub.master_thesis.welda

import java.io.File

import be.tarsos.lsh.Vector
import breeze.linalg.{DenseMatrix, DenseVector, _}
import breeze.stats.distributions.{MultivariateGaussian, RandBasis}
import knub.master_thesis.Args
import org.apache.commons.math3.random.MersenneTwister

class GaussianWELDA(p: Args) extends ReplacementWELDA(p) {

    var gaussianDistributions: Array[MultivariateGaussian] = _

    override def getFolderName(): String = {
        s"${p.modelFileName}.$embeddingName.welda.gaussian." +
            s"pca-$PCA_DIMENSIONS." +
            s"des-$DISTRIBUTION_ESTIMATION_SAMPLES." +
            s"lambda-${LAMBDA.toString.replace('.', '-')}"
    }

    override def transformVector(a: Array[Double]): Array[Double] = a

    override def estimateDistributionParameters() = {
        val topTopicVectors = getTopTopicVectors()

        val gaussianParameters = topTopicVectors.map(determineGaussianParameters)
        gaussianDistributions = gaussianParameters.map { case (meanVector, covarianceMatrix) =>
            new MultivariateGaussian(meanVector, covarianceMatrix)
        }
    }

    override def sampleFromDistribution(topicId: Int): DenseVector[Double] = {
        gaussianDistributions(topicId).sample()
    }


    def determineGaussianParameters(vectors: Seq[Array[Double]]): (DenseVector[Double], DenseMatrix[Double]) = {
        val mean = new Array[Double](PCA_DIMENSIONS)
        for (i <- 0 until PCA_DIMENSIONS) {
            var meanAtDim = 0.0
            vectors.foreach { a =>
                meanAtDim += a(i)
            }
            mean(i) = meanAtDim / vectors.size
        }
        val meanVector = new DenseVector(mean)
        var covariance = new DenseMatrix[Double](PCA_DIMENSIONS, PCA_DIMENSIONS)
        vectors.foreach { vectorValues =>
            val vector = new DenseVector(vectorValues)
            val normalized = vector - meanVector
            val result = normalized * normalized.t
            covariance += result
        }
        covariance *= 1.0 / (vectors.size - 1)
        (meanVector, covariance)
    }

    def testCovarianceMatrixCalculation() = {
        def getSamples(): IndexedSeq[DenseVector[Double]] = {
            val mean = DenseVector(Array(1.0, 2.0))
            val cov = DenseMatrix(Array(1.0, 0.0), Array(0.0, 1.0))
            val dist = new MultivariateGaussian(mean, cov)(new RandBasis(new MersenneTwister(21011991)))

            val samples = for (i <- 0 until 5000) yield dist.sample()
            samples
        }
        val samples = getSamples()
        val samplesMatrix = DenseMatrix(samples: _*)

        val numDimensions = 2
        val mean = new Array[Double](numDimensions)
        for (i <- 0 until numDimensions) {
            var meanAtDim = 0.0
            samples.foreach { a =>
                meanAtDim += a(i)
            }
            mean(i) = meanAtDim / samples.size
        }
        val meanVector = new DenseVector(mean)


        var covariance = new DenseMatrix[Double](numDimensions, numDimensions)
        samples.foreach { sample =>
            val diff = sample - meanVector
            val result = diff * diff.t
            covariance += result
        }
        covariance *= 1.0 / (samples.size - 1)

        println("Mean:")
        println(meanVector)
        println("Covariance Matrix:")
        println(covariance)
        val foo = samplesMatrix.t * samplesMatrix
        println(1.0 / (samples.size - 1) * foo)
    }
}
