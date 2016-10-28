package knub.master_thesis.welda

import java.io.File
import java.nio.file.{Files, Path, Paths}
import java.util

import be.tarsos.lsh.{CommandLineInterface, LSH, Vector}
import breeze.linalg.{DenseMatrix, DenseVector}
import knub.master_thesis
import knub.master_thesis.Args
import master_thesis.util.{Sampler, Timer, Word2VecUtils}
import org.apache.commons.io.FileUtils
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import weka.core.{Attribute, DenseInstance, Instance, Instances}
import weka.core.neighboursearch.KDTree

import scala.collection.mutable
import scala.collection.JavaConverters._


class WordInstance(weight: Double, attValues: Array[Double], val word: String)
    extends DenseInstance(weight, attValues) {
    override def copy(): Instance = {
        this
    }
}

abstract class ReplacementWELDA(p: Args) extends BaseWELDA(p) {
    new File(".").listFiles().filter(_.getName.contains("be.tarsos")).foreach { f => println(f); f.delete() }

    // dimensions to sample embeddings down to, so we can estimate the parameters of the distribution from a few samples only
    val PCA_DIMENSIONS = p.pcaDimensions
    // how many samples to estimate the distribution
    val DISTRIBUTION_ESTIMATION_SAMPLES = p.distributionEstimationSamples
    override val TOPIC_OUTPUT_EVERY = 1

    val LAMBDA = p.lambda
    val DISTANCE_FUNCTION = p.weldaDistanceFunction

    assert(LAMBDA >= 0.0, "lambda must be at least zero")
    assert(LAMBDA <= 1.0, "lambda must be at most one")

    var pcaVectors: mutable.Map[String, Array[Double]] = _

    var lsh: LSH = _
    var tree: KDTree = _
    var instances: Instances = _

    var totalWords: Long = 0L
    var totalReplaced: Long = 0L

    var folder: String = _

    override def fileBaseName: String = s"$folder/welda"

    def getFolderName(): String

    // Needs to be taken from stopword topic
    val newStopwords = List(
        "1", "2", "also", "anyone", "article", "back", "believe", "ca", "'d", "even",
        "find", "first", "get", "go", "going", "good", "got", "know", "last", "like",
        "'ll", "'m", "make", "many", "may", "much", "never", "new", "one", "people",
        "'re", "really", "right", "'s", "said", "say", "see", "since", "someone", "something",
        "still", "sure", "take", "think", "time", "two", "us", "use", "used", "using",
        "'ve", "want", "way", "well", "without", "work", "would", "writes"
    )

    var newStopwordIds: Set[Int] = _

    override def init(): Unit = {
        super.init()
        val word2Vec = WordVectorSerializer.loadTxtVectors(new File(s"${p.modelFileName.replaceAll("/model$", "/")}$embeddingName.restricted.vocab.embedding.txt"))
//        println(s"Loaded vectors have ${word2Vec.getWordVector("house").length} dimensions")
        pcaVectors = mutable.Map()

        val embeddingsList = new mutable.ArrayBuffer[Array[Double]](word2IdVocabulary.size)
        val vocabulary: Array[String] = word2IdVocabulary.keys.toArray
        vocabulary.foreach { word =>
            val actualWord = Word2VecUtils.findActualWord(word2Vec, word)
            val embeddingDimensions = word2Vec.getWordVector(actualWord)
            if (embeddingDimensions == null) {
                throw new RuntimeException(word)
            }
            embeddingsList += embeddingDimensions
        }
        val rows = embeddingsList.toArray
        val M = DenseMatrix(rows: _*)
//        println(s"Dimensions of embedding matrix = ${M.rows}, ${M.cols}")

        val pcaM = pca(M, PCA_DIMENSIONS)

        initNearestNeighbourSearch(vocabulary, pcaM)


        /*
         * Give an intuition about sampling
         */
//        val topWords = getTopTopicWords()
//        estimateDistributionParameters()
//
//        val NR_SAMPLES = 3
//        for (j <- 0 until numTopics) {
//            println(s"Topic: ${topWords(j)}")
//            for (i <- 0 until NR_SAMPLES) {
//                val word = sampleAndFindWord(j)
//                println(s"Sample: $word")
//            }
//        }

        newStopwordIds = newStopwords.flatMap { word =>
            val id = word2IdVocabulary.get(word)
//            if (id.isEmpty)
//                println(word)
            id
        }.toSet

        folder = getFolderName()
        val folderFile = new File(folder)
        val parentFolderFile = folderFile.getParentFile
        val existingFolder = parentFolderFile.listFiles().find { f =>
            f.isDirectory && f.getAbsolutePath.startsWith(folderFile.getAbsolutePath)
        }

        if (existingFolder.nonEmpty && new File(f"${existingFolder.get.getAbsolutePath}/welda.iteration-${p.numIterations}%03d.topics").exists()) {
            throw new RuntimeException(s"Experiment ${existingFolder.get.getName} already existing")
        } else {
            folderFile.mkdir()
            super.init()
        }
    }

    def initNearestNeighbourSearch(vocabulary: Array[String], pcaM: DenseMatrix[Double]): Unit = {
        val lshVectors = for (i <- vocabulary.indices) yield {
            val vec = transformVector(pcaM(i, ::).t.toArray)
            val word = vocabulary(i)
            pcaVectors(word) = vec
            new Vector(word, vec)
        }

        if (PCA_DIMENSIONS <= 5) {
            val attributes = new util.ArrayList[Attribute]()
            val numFeatures = pcaM.cols

            for (i <- 0 until numFeatures)
                attributes.add(new Attribute(s"$i"))
            instances = new Instances("kdtree", attributes, vocabulary.length)

            for (i <- vocabulary.indices) yield {
                val vec = transformVector(pcaM(i, ::).t.toArray)
                val inst = new WordInstance(1.0, vec, vocabulary(i))
                instances.add(inst)
            }
            tree = new KDTree
            tree.setInstances(instances)
        } else {
            val hashFamily = CommandLineInterface.getHashFamily(0.0, DISTANCE_FUNCTION, PCA_DIMENSIONS)
            lsh = new LSH(new util.ArrayList(lshVectors.asJava), hashFamily)
            DISTANCE_FUNCTION match {
                case "cos" =>
                    lsh.buildIndex(32, 4)
                case "l2" =>
                    lsh.buildIndex(4, 4)
            }
        }
    }

    def transformVector(a: Array[Double]): Array[Double]

    def sampleAndFindWord(topicId: Int): String = {
        if (PCA_DIMENSIONS <= 5) {
//            Timer.start("kdtree")
            val sample = sampleFromDistribution(topicId)
            val inst = new DenseInstance(1.0, sample.data)
            inst.setDataset(instances)
            val result = tree.nearestNeighbour(inst)
            val word = result.asInstanceOf[WordInstance].word
//            println(word)
//            Timer.end("kdtree")
            word
        } else {
            val maxTries = 10
            var currentTry = 0
            while (currentTry < maxTries) {
                currentTry += 1
                // sample a vector from the multivariate gaussian
                // use vector form here to get nearest word to a sampled vector
                val vectorSample = sampleFromDistribution(topicId)
//                Timer.start("FINDING WORD2VEC")
//                val sample = new NDArray(Array(vectorSample.data))
//                val nearestWordWord2Vec = word2Vec.wordsNearest(sample, 1).iterator().next()
//                Timer.end("FINDING WORD2VEC")
//                Timer.start("FINDING LSH")
                val nearestNeighbours = lsh.query(new Vector("", vectorSample.data), 1)
//                Timer.end("FINDING LSH")
//                println(s"$nearestWordLsh -- $nearestWordWord2Vec")

                if (!nearestNeighbours.isEmpty)
                    return nearestNeighbours.get(0).getKey
            }
//            println(s"WARNING: Exceeded tries in topic $topicId")
            "NONE"
        }
    }

    override def sampleSingleIteration(): Unit = {
        estimateDistributionParameters()
        var wordsInIteration = 0L
        var topic0Count = 0
        var nrReplacedWordsSuccessful = 0
        for (docIdx <- 0 until p.numDocuments) {
            val docSize = corpusWords(docIdx).size
            for (wIndex <- 0 until docSize) {
                wordsInIteration += 1
                // determine topic first
                val topicId = corpusTopics(docIdx).get(wIndex)
                if (topicId == 0)
                    topic0Count += 1
                val originalWordId = corpusWords(docIdx).get(wIndex)
                // now determine the word we "observe"
                val wordId = if (Sampler.nextCoinFlip(LAMBDA) || (LAMBDA != 0.0 && topicId == 0) ||
                    (LAMBDA != 0.0 && newStopwordIds.contains(originalWordId))) {
                    val sampledWord = sampleAndFindWord(topicId)
                    if (sampledWord == "NONE") {
                        originalWordId
                    } else {
                        nrReplacedWordsSuccessful += 1
                        word2IdVocabulary(sampledWord)
                    }
                } else {
                    originalWordId
                }

                docTopicCount(docIdx)(topicId) -= 1
                topicWordCountLDA(topicId)(wordId) -= 1
                sumTopicWordCountLDA(topicId) -= 1

                for (tIndex <- 0 until numTopics) {
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
//            Timer.printAll()
        }
        totalReplaced += nrReplacedWordsSuccessful
        totalWords += wordsInIteration
        if (p.diagnosisMode)
            println(s"Given lambda: $LAMBDA, actual lambda: ${nrReplacedWordsSuccessful.toDouble / wordsInIteration}, topic0 = ${topic0Count.toDouble / wordsInIteration}")
    }

    def pca(m: DenseMatrix[Double], numDimensions: Int): DenseMatrix[Double] = {
        val rawPCA = breeze.linalg.princomp(m).scores
        val pcaResult = rawPCA(::, 0 until numDimensions)
        pcaResult
    }

    //noinspection AccessorLikeMethodIsEmptyParen
    def getTopTopicWords(): Array[Seq[String]] = {
        val result = new Array[Seq[String]](numTopics)
        for (tIndex <- 0 until numTopics) {
            val topicWordProbs = mutable.Map[Int, Double]()
            for (wIndex <- 0 until vocabularySize) {
                val pro = topicWordCountLDA(tIndex)(wIndex)
                topicWordProbs(wIndex) = pro
            }
            val mostLikelyWords = topicWordProbs.toSeq.sortBy(-_._2).take(DISTRIBUTION_ESTIMATION_SAMPLES)
            result(tIndex) = for (wordId <- mostLikelyWords) yield {
                id2WordVocabulary(wordId._1)
            }
        }
        result
    }

    def getTopTopicVectors(): Array[Seq[Array[Double]]] = {
        val topTopicWords = getTopTopicWords()
        val topTopicVectors = topTopicWords.map { topWords =>
            topWords.flatMap { topWord =>
                if (pcaVectors.contains(topWord))
                    Some(pcaVectors(topWord))
                else
                    None
            }
        }
        topTopicVectors
    }

    def getTopTopicVectorsWithWords(): Array[Seq[(String, Array[Double])]] = {
        val topTopicWords = getTopTopicWords()
        val topTopicVectors = topTopicWords.map { topWords =>
            topWords.flatMap { topWord =>
                if (pcaVectors.contains(topWord))
                    Some((topWord, pcaVectors(topWord)))
                else
                    None
            }
        }
        topTopicVectors
    }


    override def inference(): Unit = {
        super.inference()
        val f = Paths.get(folder)
        val actualLambda = Math.round(totalReplaced * 100.0 / totalWords) / 100.0
        println(s"actualLambda = $actualLambda")
        val newFolderName = folder + s".lambdaact-${actualLambda.toString.replace('.', '-')}"
        println(s"Rename \n${new File(folder).getName} to \n${new File(newFolderName).getName}")

        if (new File(newFolderName).exists()) {
            FileUtils.deleteDirectory(new File(newFolderName))
        }
        try {
            Files.move(Paths.get(folder), Paths.get(newFolderName))
        } catch {
            case e: java.nio.file.FileAlreadyExistsException =>
                println(s"ERROR: File $newFolderName already exists, keeping $folder")
        }
    }

    def estimateDistributionParameters(): Unit
    def sampleFromDistribution(topicId: Int): DenseVector[Double]

}
