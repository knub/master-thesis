package knub.master_thesis.welda

import java.io.{BufferedReader, File, FileReader}
import java.nio.file.Paths

import cc.mallet.topics.ParallelTopicModel
import com.carrotsearch.hppc.IntArrayList
import knub.master_thesis.Args
import knub.master_thesis.util.{FreeMemory, Sampler, TopicModelWriter}
import org.deeplearning4j.models.word2vec.Word2Vec
import org.nd4j.linalg.cpu.nativecpu.NDArray

import scala.collection.mutable
import scala.io.Source

case class TopicModelInfo(alpha: Array[Double], beta: Double, betaSum: Double)

class WordEmbeddingLDA(val p: Args) {

    val LAMBDA = p.lambda
    assert(LAMBDA >= 0.0, "lambda must be at least zero")
    assert(LAMBDA <= 1.0, "lambda must be at most one")
    val TM_SIM_THRESHOLD = 0.4
    val WE_SIM_THRESHOLD = 0.6

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

    val replacementWords = mutable.HashMap[Int, Array[Int]]()
    val replacementProbabilities = mutable.HashMap[Int, Array[Double]]()
    readCorpus(p.modelFileName, p.embeddingFileName)
    prepareReplacements()

    System.out.println("Memory: " + FreeMemory.get(true, 5) + " MB")


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

    def prepareReplacements(): Unit = {
        println("Reading replacements")
        val lines = Source.fromFile(p.modelFileName + "." + embeddingName + ".similarities-most-similar.with-tm").getLines().toList

        case class SimilaritiesLine(word: String, similarWord: String, embeddingSim: Double, topicModelSim: Double)
        val parsedLines = lines.map { line =>
            val split = line.split('\t')
            SimilaritiesLine(split(0), split(1), split(2).toDouble, split(3).toDouble)
        }

        parsedLines.groupBy(_.word).foreach { case (word, similars) =>
            val wordId = word2IdVocabulary(word.toLowerCase())
            // filter low topic similarity
            val filteredSimilars = similars.filter { simLine =>
                simLine.embeddingSim > WE_SIM_THRESHOLD
            }.take(1)
//            val filteredSimilars = similars
            if (filteredSimilars.nonEmpty) {
                replacementWords(wordId) = filteredSimilars.map { simLine =>
                    word2IdVocabulary(simLine.similarWord.toLowerCase())
                }.toArray
                replacementProbabilities(wordId) = filteredSimilars.map(_.embeddingSim).toArray
            }
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

    def sampleSingleIteration() {
        for (docIdx <- 0 until p.numDocuments) {
//            if (docIdx % 100000 == 0) {
//                System.out.print(docIdx + " ")
//                System.out.flush()
//            }
            val docSize = corpusWords(docIdx).size
            for (wIndex <- 0 until docSize) {
                val wordId = if (Sampler.nextCoinFlip(LAMBDA)) {
                    val originalWordId = corpusWords(docIdx).get(wIndex)
                    replacementProbabilities.get(originalWordId) match {
                        case Some(replacementProbs) =>
                            val chosenIndex = Sampler.nextDiscrete(replacementProbs)
                            replacementWords(originalWordId)(chosenIndex)
                        case None =>
                            originalWordId
                    }
                } else {
                    corpusWords(docIdx).get(wIndex)
                }
                val topicId = corpusTopics(docIdx).get(wIndex)

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
        }
    }


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

//    private def determineVocabularySize(tm: ParallelTopicModel , vectorWords: mutable.Set[String]): Int = {
//        var size = 0
//        val it = tm.getAlphabet.iterator()
//        while (it.hasNext) {
//            val w = it.next().asInstanceOf[String]
//            if (vectorWords.contains(w)) {
//                size += 1
//            }
//        }
//        size
//    }
//    private def getVectorWords(pathToVectorWords: String): mutable.Set[String] = {
//        val vectorWords = mutable.Set[String]()
//        val br = new BufferedReader(new FileReader(pathToVectorWords))
//        var word: String = br.readLine()
//        while (word != null) {
//            vectorWords.add(word)
//            word = br.readLine()
//        }
//        vectorWords
//    }
}


