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

class SimpleSimBasedReplacementWELDA(p: Args) extends BaseWELDA(p) {

    val LAMBDA = p.lambda
    assert(LAMBDA >= 0.0, "lambda must be at least zero")
    assert(LAMBDA <= 1.0, "lambda must be at most one")
    val TM_SIM_THRESHOLD = 0.4
    val WE_SIM_THRESHOLD = 0.6

    val replacementWords = mutable.HashMap[Int, Array[Int]]()
    val replacementProbabilities = mutable.HashMap[Int, Array[Double]]()
    prepareReplacements()
    System.out.println("Memory: " + FreeMemory.get(true, 5) + " MB")

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

    override def sampleSingleIteration() {
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

    override def fileBaseName: String = s"${p.modelFileName}.$embeddingName.welda.simple.lambda-${LAMBDA.toString.replace('.', '-')}"
}
