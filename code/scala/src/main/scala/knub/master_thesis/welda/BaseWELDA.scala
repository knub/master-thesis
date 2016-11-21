package knub.master_thesis.welda

import java.io.{BufferedReader, File, FileReader}
import java.nio.file.Paths

import cc.mallet.topics.ParallelTopicModel
import com.carrotsearch.hppc.IntArrayList
import knub.master_thesis.{Args, util}
import util.{Date, TopicModelWriter}
import scala.collection.JavaConverters._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.util.Random

abstract class BaseWELDA(val p: Args) {
    val embeddingName = Paths.get(p.embeddingFileName).getFileName.toString

    val writer = new TopicModelWriter(this)

    var alpha: Array[Double] = null
    var beta: Double = .0
    var betaSum: Double = .0
    var alphaSum: Double = .0
    var numTopics: Int = -1

    var BURN_IN = p.numIterations / 2
    var LAG = 5

    var vocabularySize: Int = _

    var docTopicCount: Array[Array[Int]] = _
    var docWordCount: Array[Int] = _
    var topicWordCountLDA: Array[Array[Int]] = _
    var sumTopicWordCountLDA: Array[Int] = _
    var multiPros: Array[Double] = _

    var corpusWords: ArrayBuffer[IntArrayList] = _
    var corpusTopics: ArrayBuffer[IntArrayList] = _

    var topicAssignmentSamples: ArrayBuffer[ArrayBuffer[IntArrayList]] = ArrayBuffer()

    var word2IdVocabulary: mutable.Map[String, Int] = null
    var id2WordVocabulary: mutable.Map[Int, String] = null

    def init(): Unit = {
        val info = loadTopicModelInfo()
        alpha = info.alpha
        beta = info.beta
        betaSum = info.betaSum
        numTopics = info.numTopics
        alphaSum = alpha.sum

        vocabularySize = Source.fromFile(s"${p.modelFileName}.$embeddingName.restricted.vocab").getLines().size
//        println(s"Topic model loaded, vocabularySize = $vocabularySize")

        docTopicCount = Array.ofDim[Int](p.numDocuments, numTopics)
        docWordCount = new Array[Int](p.numDocuments)
        topicWordCountLDA = Array.ofDim[Int](numTopics, vocabularySize)
        sumTopicWordCountLDA = new Array[Int](numTopics)

        multiPros = new Array[Double](numTopics)

        corpusWords = new mutable.ArrayBuffer[IntArrayList](p.numDocuments)
        corpusTopics = new mutable.ArrayBuffer[IntArrayList](p.numDocuments)

        readCorpus(p.modelFileName, p.embeddingFileName)
    }

    def loadTopicModelInfo(): TopicModelInfo = {
        System.out.println ("Loading topic model info " + p.modelFileName)
        val tm = ParallelTopicModel.read(new File(p.modelFileName))
        val alph = tm.getAlphabet
        val vocabSize = alph.size()
        val betaSum = vocabSize * beta
        new TopicModelInfo (tm.alpha, tm.beta, betaSum, tm.numTopics)
    }

    private def buildId2WordVocabulary(word2IdVocabulary: mutable.Map[String, Int]): mutable.Map[Int, String] = {
        val result = mutable.Map[Int, String]()
        for ((key, value) <- word2IdVocabulary) {
            result.put(value, key)
        }
        result
    }

    private def readWord2IdVocabulary(modelFileName: String): mutable.Map[String, Int] = {
        val brAlphabet = new BufferedReader(new FileReader(modelFileName + "." + embeddingName + ".restricted.vocab"))
        val result = mutable.Map[String, Int]()
        var line = brAlphabet.readLine()
        var i = 0
        while (line != null) {
            result.put(line, i)
            i += 1
            line = brAlphabet.readLine()
        }
        brAlphabet.close()
//        println(s"Finished reading vocabulary, ${result.size} words")
        result
    }

    private def readCorpus(modelFileName: String, embeddingFileName: String) {
        val r = new Random(21011991)
        var docId = 0
        word2IdVocabulary = readWord2IdVocabulary(modelFileName)
        id2WordVocabulary = buildId2WordVocabulary(word2IdVocabulary)
        val brDocument = new BufferedReader(new FileReader(modelFileName + "." + embeddingName + ".restricted"))
        var lineNr = 0
        var documentWords = new IntArrayList()
        var documentTopics = new IntArrayList()
        var line: String = brDocument.readLine()
        while (line != null) {
            if (line == "##") {
                if (documentWords.size > 0) {
                    corpusWords += documentWords
                    corpusTopics += documentTopics
                    documentWords = new IntArrayList()
                    documentTopics = new IntArrayList()
                    if (docId % 5000 == 0) {
                        println(docId)
                    }
                    docId += 1
                }
            } else {
                try {
                    val wordId = java.lang.Integer.parseInt(line.substring(0, 6))
                    val topicId = if (p.randomTopicInitialization) {
                        r.nextInt(numTopics)
                    } else {
                        java.lang.Integer.parseInt(line.substring(7, 13))
                    }
                    topicWordCountLDA(topicId)(wordId) += 1
                    sumTopicWordCountLDA(topicId) += 1
                    docTopicCount(docId)(topicId) += 1
                    docWordCount(docId) += 1
                    documentWords.add(wordId)
                    documentTopics.add(topicId)
                } catch {
                    case e: Exception =>
                        println(line)
                        println("lineNr = " + lineNr)
                        println("line = <" + line + ">")
                        throw e
                }
                lineNr += 1
            }
            line = brDocument.readLine()
        }
    }

    val TOPIC_OUTPUT_EVERY = 50

    var currentIteration: Int = 0

    def inference(): Unit = {
        for (iter <- 0 until p.numIterations) {
            currentIteration = iter
            if (p.saveStep > 0 && iter % p.saveStep == 0 && iter < p.numIterations) {
                writer.write(iter)
            }
            if (iter % TOPIC_OUTPUT_EVERY == 0) {
                println(s"\t${Date.date()}: It. ${iter + 1} (lambda=${p.lambda}, " +
                    s"embedding=${new File(p.embeddingFileName).getName}, " +
                    s"pca=${this.asInstanceOf[ReplacementWELDA].PCA_DIMENSIONS}, " +
                    s"samples=${this.asInstanceOf[ReplacementWELDA].DISTRIBUTION_ESTIMATION_SAMPLES})")
            }
            sampleSingleIteration()
            if (iter + 1 >= BURN_IN && (iter + 1) % LAG == 0) {
                storeCopyOfCurrentTopics()
            }
        }

        println(s"${Date.date()}: Aggregating topic assignments")
        aggregateTopicAssignments()
        println(s"${Date.date()}: Done aggregating topic assignments")


        writer.writeParameters()
        println("Sampling completed!")
        writer.write(p.numIterations)
    }

    def aggregateTopicAssignments(): Unit = {
        corpusTopics.indices.foreach { docIdx =>
            for (wordIdx <- 0 until corpusTopics(docIdx).size()) {
                val topicAssignments = topicAssignmentSamples.map { sample =>
                    sample(docIdx).get(wordIdx)
                }
                val maxTopic = topicAssignments.groupBy(identity).maxBy(_._2.size)._1
                corpusTopics(docIdx).set(wordIdx, maxTopic)
            }
        }
        docTopicCount = Array.ofDim[Int](p.numDocuments, numTopics)
        topicWordCountLDA = Array.ofDim[Int](numTopics, vocabularySize)
        sumTopicWordCountLDA = new Array[Int](numTopics)
        corpusWords.indices.foreach { docIdx =>
            for (wordIdx <- 0 until corpusTopics(docIdx).size()) {
                val word = corpusWords(docIdx).get(wordIdx)
                val topic = corpusTopics(docIdx).get(wordIdx)
                docTopicCount(docIdx)(topic) += 1
                topicWordCountLDA(topic)(word) += 1
                sumTopicWordCountLDA(topic) += 1
            }
        }
    }

    def storeCopyOfCurrentTopics(): ArrayBuffer[ArrayBuffer[IntArrayList]] = {
        val sampleCorpusTopics = new mutable.ArrayBuffer[IntArrayList](p.numDocuments)
        corpusTopics.foreach { t => sampleCorpusTopics += t.clone() }
        topicAssignmentSamples += sampleCorpusTopics
    }

    def sampleSingleIteration(): Unit

    def fileBaseName: String
}
