package knub.master_thesis.welda

import java.io.File
import java.util

import be.tarsos.lsh.{CommandLineInterface, LSH, Vector}
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.stats.distributions.{MultivariateGaussian, RandBasis}
import de.uni_potsdam.hpi.coheel.util.Timer
import knub.master_thesis.Args
import knub.master_thesis.util.Sampler
import org.apache.commons.math3.random.MersenneTwister
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors

import scala.collection.mutable

class GaussianWELDA(p: Args) extends BaseWELDA(p) {

    override val TOPIC_OUTPUT_EVERY = 1
    // value sets, over how many top words the gaussians are calculated
    val GAUSSIAN_OVER_TOP_N = 250
    // how often to sample from the gaussians
    val LAMBDA = p.lambda
    println(s"LAMBDA is set to ${LAMBDA.toString}")
    assert(LAMBDA >= 0.0, "lambda must be at least zero")
    assert(LAMBDA <= 1.0, "lambda must be at most one")

    // after each generation, re-estimate gaussians

    var word2Vec: WordVectors = _
    var numDimensions: Int = _

    var nrFirstWordNotInVocab = 0L
    var nrReplacedWords = 0L

    var lsh: LSH = _

    override def init(): Unit = {
        super.init()
        word2Vec = WordVectorSerializer.loadTxtVectors(new File(p.embeddingFileName + ".txt"))

        val embeddings = new util.ArrayList[be.tarsos.lsh.Vector](word2IdVocabulary.size)
        word2IdVocabulary.keys.foreach { word =>
            val embeddingDimensions = word2Vec.getWordVector(word)
            embeddings.add(new Vector(word, embeddingDimensions))
        }
        val hashFamily = CommandLineInterface.getHashFamily(0.0, "cos", embeddings.get(0).getDimensions)
        lsh = new LSH(embeddings, hashFamily)
        lsh.buildIndex(16, 4)

        numDimensions = word2Vec.getWordVector("this").length
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

    def estimateGaussians() = {
        val topTopicWords = getTopTopicWords()
        val topTopicVectors = topTopicWords.map { topWords =>
            topWords.flatMap { topWord =>
                if (word2Vec.hasWord(topWord))
                    Some(word2Vec.getWordVector(topWord))
                else
                    None
            }
        }
        val meanVectors = topTopicVectors.map { topVectors =>
            val mean = new Array[Double](numDimensions)
            for (i <- 0 until numDimensions) {
                var meanAtDim = 0.0
                topVectors.foreach { a =>
                    meanAtDim += a(i)
                }
                mean(i) = meanAtDim / topVectors.size
            }
            new DenseVector(mean)
        }
        val covariances = topTopicVectors.zipWithIndex.map { case (topVectors, topicId) =>
            val meanVector = meanVectors(topicId)
            var covariance = new DenseMatrix[Double](numDimensions, numDimensions)
            topVectors.foreach { topVector =>
                val topVectorAsMatrix = new DenseVector(topVector)
                val diff = topVectorAsMatrix - meanVector
                val result = diff * diff.t
                covariance += result
            }
            covariance *= 1.0 / (topVectors.size - 1)
        }
        gaussianDistributions = (0 until p.numTopics).map { i =>
            new MultivariateGaussian(meanVectors(i), covariances(i))
        }.toArray

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
//                    Timer.start("SAMPLING")
                    val vectorSample = gaussianDistributions(topicId).sample()
//                    Timer.end("SAMPLING")
//                    Timer.start("CONVERTING")
//                    val sample = new NDArray(Array(vectorSample.data))
//                    Timer.end("CONVERTING")
//                    Timer.start("FINDING LSH")
                    val nearestNeighbours = lsh.query(new Vector("to search", vectorSample.data), 1)
                    if (nearestNeighbours.isEmpty) {
                        val nearestWordLsh = if (nearestNeighbours.isEmpty) "<NULL>" else nearestNeighbours.get(0).getKey
//                        Timer.end("FINDING LSH")
//                        Timer.start("FINDING WORD2VEC")
//                        val nearestWordWord2Vec = word2Vec.wordsNearest(sample, 1).iterator().next()
//                        Timer.end("FINDING WORD2VEC")
//                        println(s"$nearestWordLsh -- $nearestWordWord2Vec")
                        nrReplacedWords += 1
                        word2IdVocabulary.get(nearestWordLsh) match {
                            case Some(id) =>
                                id
                            case None =>
                                nrFirstWordNotInVocab += 1
                                corpusWords(docIdx).get(wIndex)
                        }
                    } else {
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
//            println(s"How often does replacing work: ${nrFirstWordNotInVocab.toDouble / nrReplacedWords}")
            Timer.printAll()
        }

    }

    override def fileBaseName: String = s"${p.modelFileName}.$embeddingName.welda.gaussian.lambda-${LAMBDA.toString.replace('.', '-')}"

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

        numDimensions = 2
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
