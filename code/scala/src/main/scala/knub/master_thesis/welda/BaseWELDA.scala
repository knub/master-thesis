package knub.master_thesis.welda

import java.io.{BufferedReader, File, FileReader}
import java.nio.file.Paths

import cc.mallet.topics.ParallelTopicModel
import com.carrotsearch.hppc.IntArrayList
import knub.master_thesis.Args
import knub.master_thesis.util.TopicModelWriter

import scala.collection.mutable
import scala.io.Source

abstract class BaseWELDA(val p: Args) {
    val embeddingName = Paths.get(p.embeddingFileName).getFileName.toString

    val writer = new TopicModelWriter(this)

    val TopicModelInfo(alpha, beta, betaSum) = loadTopicModelInfo()
    val vocabularySize = Source.fromFile(s"${p.modelFileName}.$embeddingName.restricted.alphabet").getLines().size
    println(s"Topic model loaded, vocabularySize = $vocabularySize")
    val alphaSum = alpha.sum


    val docTopicCount = Array.ofDim[Int](p.numDocuments, p.numTopics)
    val docWordCount = new Array[Int](p.numDocuments)
    val topicWordCountLDA = Array.ofDim[Int](p.numTopics, vocabularySize)
    val sumTopicWordCountLDA = new Array[Int](p.numTopics)

    val multiPros = new Array[Double](p.numTopics)

    val corpusWords = new mutable.ArrayBuffer[IntArrayList](p.numDocuments)
    val corpusTopics = new mutable.ArrayBuffer[IntArrayList](p.numDocuments)

    var word2IdVocabulary: mutable.Map[String, Int] = null
    var id2WordVocabulary: mutable.Map[Int, String] = null

    readCorpus(p.modelFileName, p.embeddingFileName)

    def loadTopicModelInfo(): TopicModelInfo = {
        System.out.println ("Loading topic model info " + p.modelFileName)
        val tm = ParallelTopicModel.read(new File(p.modelFileName))
        val alph = tm.getAlphabet
        val vocabSize = alph.size()
        //        val vocabularySize = determineVocabularySize(tm, getVectorWords(pathToVectorWords))
        val betaSum = vocabSize * beta
        new TopicModelInfo (tm.alpha, tm.beta, betaSum)
    }


    private def buildId2WordVocabulary(word2IdVocabulary: mutable.Map[String, Int]): mutable.Map[Int, String] = {
        val result = mutable.Map[Int, String]()
        for ((key, value) <- word2IdVocabulary) {
            result.put(value, key)
        }
        result
    }

    private def readWord2IdVocabulary(modelFileName: String): mutable.Map[String, Int] = {
        val brAlphabet = new BufferedReader(new FileReader(modelFileName + "." + embeddingName + ".restricted.alphabet"))
        val result = mutable.Map[String, Int]()
        var line = brAlphabet.readLine()
        while (line != null) {
            val split = line.split("#")
            result.put(split(0), split(1).toInt)
            line = brAlphabet.readLine()
        }
        brAlphabet.close()
        println(s"Finished reading vocabulary, ${result.size} words")
        result
    }

    private def readCorpus(modelFileName: String, embeddingFileName: String) {
        println("Reading corpus")
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
                    val topicId = java.lang.Integer.parseInt(line.substring(7, 13))
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

    def inference(): Unit = {
        for (iter <- 0 until p.numIterations) {
            if (p.saveStep > 0 && iter % p.saveStep == 0 && iter < p.numIterations) {
                System.out.println("\t\tSaving the output from the " + iter + "^{th} sample")
                writer.write(f"$iter%03d")
            }
            if (iter % 50 == 0) {
                println("\tWELDA sampling iteration: " + iter)
            }
            sampleSingleIteration()
        }
        writer.writeParameters()
        println("Sampling completed!")
        writer.write(p.numIterations.toString)
    }

    def sampleSingleIteration(): Unit

    def fileBaseName: String
}
