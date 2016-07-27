package knub.master_thesis

import java.io.{BufferedReader, File, FileReader}

import cc.mallet.topics.ParallelTopicModel
import cc.mallet.types.Alphabet
import com.carrotsearch.hppc.IntArrayList
import knub.master_thesis.util.{FreeMemory, Sampler, TopicModelWriter}

import scala.collection.mutable

case class TopicModelInfo(wordAlphabet: Alphabet, vocabularySize: Int)

class WordEmbeddingLDA(val p: Args) {

    val writer = new TopicModelWriter(this)

    val TopicModelInfo(wordAlphabet, vocabularySize) = loadTopicModelInfo()
    val docTopicCount = Array.ofDim[Int](p.numDocuments, p.numTopics)
    val sumDocTopicCount = new Array[Int](p.numDocuments)
    val topicWordCountLDA = Array.ofDim[Int](p.numTopics, vocabularySize)
    val sumTopicWordCountLDA = new Array[Int](p.numTopics)

    val multiPros = new Array[Double](p.numTopics)

    val alphaSum = p.numTopics * p.alpha
    val betaSum = vocabularySize * p.beta

    val corpusWords = new mutable.ArrayBuffer[IntArrayList](p.numDocuments)
    val corpusTopics = new mutable.ArrayBuffer[IntArrayList](p.numDocuments)

    var word2IdVocabulary: mutable.Map[String, Int] = null
    var id2WordVocabulary: mutable.Map[Int, String] = null
    readCorpus(p.modelFileName)

    System.out.println("Memory: " + FreeMemory.get(true, 5) + " MB")

    private def readCorpus(pathToTopicModel: String) {
        println("Reading corpus")
        var docId = 0
        val brAlphabet = new BufferedReader(new FileReader(pathToTopicModel + ".lflda.alphabet"))
        word2IdVocabulary = readWord2IdVocabulary(brAlphabet.readLine())
        brAlphabet.close()
        id2WordVocabulary = buildId2WordVocabulary(word2IdVocabulary)
        val brDocument = new BufferedReader(new FileReader(pathToTopicModel + ".lflda"))
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
                    sumDocTopicCount(docId) += 1
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
        println("Running Gibbs sampling inference: ")

        for (iter <- 0 until p.numIterations) {
            println("\tLFLDA sampling iteration: " + iter)
            sampleSingleIteration()

            if (p.saveStep > 0 && iter % p.saveStep == 0 && iter < p.numIterations) {
                System.out.println("\t\tSaving the output from the " + iter + "^{th} sample")
                writer.write(String.valueOf(iter))
            }
        }
        println("Sampling completed!")
        writer.writeParameters()
        System.out.println("Writing output from the last sample ...")
        writer.write(p.numIterations.toString)
    }

    def sampleSingleIteration() {
        println("\t\tRunning iteration ...")
        for (docIdx <- 0 until p.numDocuments) {
//            if (docIdx % 100000 == 0) {
//                System.out.print(docIdx + " ")
//                System.out.flush()
//            }
            val docSize = corpusWords(docIdx).size
            for (wIndex <- 0 until docSize) {
                val wordId = corpusWords(docIdx).get(wIndex)
                val topicId = corpusTopics(docIdx).get(wIndex)

                docTopicCount(docIdx)(topicId) -= 1
                topicWordCountLDA(topicId)(wordId) -= 1
                sumTopicWordCountLDA(topicId) -= 1

                for (tIndex <- 0 until p.numTopics) {
                    multiPros(tIndex) =
                        (docTopicCount(docIdx)(tIndex) + p.alpha) * (topicWordCountLDA(tIndex)(wordId) + p.beta) /
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
        new TopicModelInfo (alph, vocabSize)
    }


    private def buildId2WordVocabulary(word2IdVocabulary: mutable.Map[String, Int]): mutable.Map[Int, String] = {
        val result = mutable.Map[Int, String]()
        for ((key, value) <- word2IdVocabulary) {
            result.put(value, key)
        }
        result
    }

    private def readWord2IdVocabulary(line: String): mutable.Map[String, Int] = {
        val result = mutable.Map[String, Int]()
        val split = line.split(" ")
        for (alphabetEntry <- split) {
            val innerSplit = alphabetEntry.split("#")
            result.put(innerSplit(0), java.lang.Integer.parseInt(innerSplit(1)))
        }
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


