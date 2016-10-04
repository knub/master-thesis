package knub.master_thesis.welda

import java.io.File

import breeze.linalg._
import breeze.linalg.{DenseMatrix, DenseVector, norm, svd}
import breeze.stats.distributions.Rand
import etomica.math.SpecialFunctions
import knub.master_thesis.Args
import knub.master_thesis.math.Bessel
import org.apache.commons.math3.distribution.BetaDistribution

import scala.collection.mutable

case class Vmf(kappa: Double, mu: DenseVector[Double])

class VmfWELDA(p: Args) extends ReplacementWELDA(p) {

    var vmfs: Array[Vmf] = _

    override val PCA_DIMENSIONS = 10
    override val DISTRIBUTION_ESTIMATION_SAMPLES = 20
    val KAPPA_FACTOR_FOR_MORE_CONCENTRATION = p.kappaFactor

    val vmfDim = PCA_DIMENSIONS - 1
    val betaDist = new BetaDistribution(vmfDim / 2.0, vmfDim / 2.0)

    override def init(): Unit = {
        super.init()
        folder = s"${p.modelFileName}.$embeddingName.welda.vmf." +
            s"distance-$DISTANCE_FUNCTION." +
            s"lambda-${LAMBDA.toString.replace('.', '-')}." +
            s"kappafactor-$KAPPA_FACTOR_FOR_MORE_CONCENTRATION"
        new File(folder).mkdir()
    }

    override def transformVector(a: Array[Double]): Array[Double] = {
        val v = new DenseVector(a)
        val n = norm(v)
        (v / n).toArray
    }

    override def estimateDistributionParameters(): Unit = {
        val topTopicVectors = getTopTopicVectors()

        vmfs = topTopicVectors.map { vectors =>
            val sum = new DenseVector[Double](PCA_DIMENSIONS)
            vectors.foreach { v =>
                sum += new DenseVector(v)
            }
            val mu = sum / norm(sum)
            val R = norm(sum) / vectors.length
            val Rsquared = R * R

            val kappa = R * (PCA_DIMENSIONS - Rsquared) / (1 - Rsquared)
            val finalKappa = try {
                val kappa1 = refineKappa(R, kappa)
                val kappa2 = refineKappa(R, kappa1)
                kappa2
//                println(s"kappa = $kappa, kappa1 = $kappa1, kappa2 = $kappa2")
            } catch {
                case e: RuntimeException =>
                    kappa
            }
            val concentratedKappa = finalKappa * KAPPA_FACTOR_FOR_MORE_CONCENTRATION
            Vmf(concentratedKappa, mu)
        }
    }

    def refineKappa(R: Double, kappa: Double): Double = {
        val ApKappa = Ap(kappa, PCA_DIMENSIONS)
        val refinedKappa = kappa - (ApKappa - R) / (1.0 - ApKappa * ApKappa - (PCA_DIMENSIONS - 1) / kappa * ApKappa)
        refinedKappa
    }

    private def Ap(kappa: Double, p: Int): Double = {
        val foo1 = SpecialFunctions.besselI(p / 2, kappa) / SpecialFunctions.besselI(-1 + p / 2, kappa)
//        val foo2 = Bessel.bessi(p / 2, kappa) / Bessel.bessi(-1 + p / 2, kappa)
//        println(s"foo1 = $foo1, foo2 = $foo2")
        foo1
    }

    def rW(kappa: Double): Double = {
        val b = vmfDim / (Math.sqrt(4 * kappa * kappa + vmfDim * vmfDim) + 2 * kappa)
        val x = (1 - b) / (1 + b)
        val c = kappa * x + vmfDim * Math.log(1 - x * x)

        while (true) {
            val z = betaDist.sample()
            val w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            val u = Math.random()
            if (kappa * w + vmfDim * Math.log(1 - x * w) - c >= Math.log(u)) {
                return w
            }
        }
        throw new RuntimeException
    }

    def rvMF(kappa: Double, mu: DenseVector[Double]): DenseVector[Double] = {
        assert(mu.length == PCA_DIMENSIONS)
        val w = rW(kappa)
        val v = sampleTangentUnit(mu)

        Math.sqrt(1 - w * w) * v + w * mu
    }

    def sampleTangentUnit(mu: DenseVector[Double]): DenseVector[Double] = {
        val mat = mu.asDenseMatrix.t

        val svd.SVD(u, _, _) = svd(mat)
        val nu = DenseMatrix.rand(mat.rows, 1, Rand.gaussian)

        val foo1 = u(::, 1 to -1)
        val foo2 = nu(1 to -1, ::)
        val x = (foo1 * foo2).toDenseVector
        val normX = norm(x)
        x / normX
    }

    override def sampleFromDistribution(topicId: Int): DenseVector[Double] = {
        val Vmf(kappa, mu) = vmfs(topicId)
        val sample = rvMF(kappa, mu)
//        if (topicId == 0)
//            println(s"Sample: $sample")
        sample
    }

    override def fileBaseName: String = s"$folder/welda"
}
