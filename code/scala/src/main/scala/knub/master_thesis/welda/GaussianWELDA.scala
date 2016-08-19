package knub.master_thesis.welda

import java.io.File
import java.util

import be.tarsos.lsh.{CommandLineInterface, LSH, Vector}
import breeze.linalg.{DenseMatrix, DenseVector, _}
import breeze.stats.distributions.{MultivariateGaussian, RandBasis}
import de.uni_potsdam.hpi.coheel.util.Timer
import knub.master_thesis.Args
import knub.master_thesis.util.Sampler
import org.apache.commons.math3.random.MersenneTwister
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.nd4j.linalg.cpu.nativecpu.NDArray

import scala.collection.JavaConverters._
import scala.collection.mutable

class GaussianWELDA(p: Args) extends BaseWELDA(p) {
    val PCA_DIMENSIONS = 10
    val GAUSSIAN_OVER_TOP_N = 20
    override val TOPIC_OUTPUT_EVERY = 1

    // value sets, over how many top words the gaussians are calculated
    // how often to sample from the gaussians
    val LAMBDA = p.lambda
    val DISTANCE_FUNCTION = p.gaussianWELDADistanceFunction

    println(s"LAMBDA is set to ${LAMBDA.toString}")
    println(s"Distance Function is set to $DISTANCE_FUNCTION")
    assert(LAMBDA >= 0.0, "lambda must be at least zero")
    assert(LAMBDA <= 1.0, "lambda must be at most one")

    var word2Vec: WordVectors = _
    var pcaVectors: mutable.Map[String, Array[Double]] = _
//    var numDimensions: Int = _

    var nrReplacedWordsSuccessful = 0L
    var nrReplacedWordsTries = 0L

    var lsh: LSH = _

    var folder: String = _

    override def init(): Unit = {
        super.init()
        word2Vec = WordVectorSerializer.loadTxtVectors(new File(s"${p.modelFileName}.$embeddingName.embedding.txt"))
        println(s"Loaded vectors have ${word2Vec.getWordVector("house").length} dimensions")
        pcaVectors = mutable.Map()

        val embeddingsList = new mutable.ArrayBuffer[Array[Double]](word2IdVocabulary.size)
        val vocabulary: Array[String] = word2IdVocabulary.keys.toArray
        vocabulary.foreach { word =>
            val embeddingDimensions = word2Vec.getWordVector(word)
            embeddingsList += embeddingDimensions
        }
        val rows = embeddingsList.toArray
        val M = DenseMatrix(rows: _*)
        println(s"Dimensions of embedding matrix = ${M.rows}, ${M.cols}")

        val pcaM = pca(M, PCA_DIMENSIONS)

        val lshVectors = for (i <- vocabulary.indices) yield {
            val vec = pcaM(i, ::).t.toArray
            val word = vocabulary(i)
            pcaVectors(word) = vec
            new Vector(word, vec)
        }

        println(s"Words in vocabulary: ${word2Vec.vocab().numWords()}, words in LSH: ${lshVectors.size} (should be equal)")

        val hashFamily = CommandLineInterface.getHashFamily(0.0, DISTANCE_FUNCTION, PCA_DIMENSIONS)
        lsh = new LSH(new util.ArrayList(lshVectors.asJava), hashFamily)
        DISTANCE_FUNCTION match {
            case "cos" =>
                lsh.buildIndex(32, 4)
            case "l2" =>
                lsh.buildIndex(4, 4)
        }

        folder = s"${p.modelFileName}.$embeddingName.welda.gaussian.distance-$DISTANCE_FUNCTION.lambda-${LAMBDA.toString.replace('.', '-')}"
        new File(folder).mkdir()
    }

    //noinspection AccessorLikeMethodIsEmptyParen
    private def getTopTopicWords(): Array[Seq[String]] = {
        val result = new Array[Seq[String]](p.numTopics)
        for (tIndex <- 0 until p.numTopics) {
            val topicWordProbs = mutable.Map[Int, Double]()
            for (wIndex <- 0 until vocabularySize) {
                val pro = topicWordCountLDA(tIndex)(wIndex)
                topicWordProbs(wIndex) = pro
            }
            val mostLikelyWords = topicWordProbs.toSeq.sortBy(-_._2).take(GAUSSIAN_OVER_TOP_N)
            result(tIndex) = for (wordId <- mostLikelyWords) yield {
                id2WordVocabulary(wordId._1)
            }
        }
        result
    }

    var gaussianDistributions: Array[MultivariateGaussian] = _

    def pca(m: DenseMatrix[Double], numDimensions: Int): DenseMatrix[Double] = {
        val rawPCA = breeze.linalg.princomp(m).scores
        val pcaResult = rawPCA(::, 0 until numDimensions)
        pcaResult
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

    def estimateGaussians() = {
        val topTopicWords = getTopTopicWords()
        val topTopicVectors = topTopicWords.map { topWords =>
            topWords.flatMap { topWord =>
                if (pcaVectors.contains(topWord))
                    Some(pcaVectors(topWord))
                else
                    None
            }
        }

        val gaussianParameters = topTopicVectors.map(determineGaussianParameters)
        gaussianDistributions = gaussianParameters.map { case (meanVector, covarianceMatrix) =>
            new MultivariateGaussian(meanVector, covarianceMatrix)
        }
    }

    override def sampleSingleIteration(): Unit = {
        estimateGaussians()
        for (docIdx <- 0 until p.numDocuments) {
            val docSize = corpusWords(docIdx).size
            for (wIndex <- 0 until docSize) {
                // determine topic first
                val topicId = corpusTopics(docIdx).get(wIndex)
                // now determine the word we "observe"
                val wordId = if (Sampler.nextCoinFlip(LAMBDA)) {
                    // sample a vector from the multivariate gaussian
                    // use vector form here to get nearest word to a sampled vector
                    val vectorSample = gaussianDistributions(topicId).sample()
//                    Timer.start("FINDING WORD2VEC")
//                    val sample = new NDArray(Array(vectorSample.data))
//                    val nearestWordWord2Vec = word2Vec.wordsNearest(sample, 1).iterator().next()
//                    Timer.end("FINDING WORD2VEC")
//                    Timer.start("FINDING LSH")
                    val nearestNeighbours = lsh.query(new Vector("", vectorSample.data), 1)
                    val nearestWordLsh = if (nearestNeighbours.isEmpty) "<NULL>" else nearestNeighbours.get(0).getKey
//                    Timer.end("FINDING LSH")
//                    println(s"$nearestWordLsh -- $nearestWordWord2Vec")
                    nrReplacedWordsTries += 1
                    word2IdVocabulary.get(nearestWordLsh) match {
                        case Some(id) =>
                            nrReplacedWordsSuccessful += 1
                            id
                        case None =>
                            corpusWords(docIdx).get(wIndex)
                    }
                } else {
                    corpusWords(docIdx).get(wIndex)
                }

                docTopicCount(docIdx)(topicId) -= 1
                topicWordCountLDA(topicId)(wordId) -= 1
                sumTopicWordCountLDA(topicId) -= 1

                for (tIndex <- 0 until p.numTopics) {
                    multiPros(tIndex) =
                        (docTopicCount(docIdx)(tIndex) + alpha(tIndex)) * (topicWordCountLDA(tIndex)(wordId) + beta) /
                            (sumTopicWordCountLDA(tIndex) + betaSum)
                }
                val newTopicId = Sampler.nextDiscrete(multiPros)

                docTopicCount(docIdx)(newTopicId) += 1
                topicWordCountLDA(newTopicId)(wordId) += 1
                sumTopicWordCountLDA(newTopicId) += 1

                // update topic
                corpusTopics(docIdx).set(wIndex, newTopicId)
            }
//            println(s"How often does replacing work: ${nrReplacedWordsSuccessful.toDouble / nrReplacedWordsTries}")
            Timer.printAll()
        }

    }


    override def fileBaseName: String = s"$folder/welda"

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
