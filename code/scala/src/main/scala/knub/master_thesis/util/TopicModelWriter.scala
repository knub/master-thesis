package knub.master_thesis.util

import java.io.BufferedWriter
import java.io.FileWriter
import java.io.IOException
import java.util
import java.util.Map
import java.util.Set
import java.util.TreeMap

import scala.collection.JavaConverters._
import knub.master_thesis.WordEmbeddingLDA

import scala.collection.mutable

class TopicModelWriter(private val model: WordEmbeddingLDA) {

    val params = model.p

    def writeParameters() {
        val writer = new BufferedWriter(new FileWriter(params.modelFileName + ".welda.params"))
        writer.write("-model" + "\t" + "WELDA")
        writer.write("\n-topicmodel" + "\t" + params.modelFileName)
        writer.write("\n-ntopics" + "\t" + params.numTopics)
        writer.write("\n-alpha" + "\t" + params.alpha)
        writer.write("\n-beta" + "\t" + params.beta)
        writer.write("\n-niters" + "\t" + params.numIterations)
        if (params.saveStep > 0) writer.write("\n-sstep" + "\t" + params.saveStep)
        writer.close()
    }

    def writeTopicAssignments(name: String) {
        val writer = new BufferedWriter(new FileWriter(params.modelFileName + ".welda-" + name + ".topic-assignments"))
        for (dIndex <- 0 until params.numDocuments) {
            val docSize = model.corpusWords(dIndex).size
            for (wIndex <- 0 until docSize) {
                writer.write(model.corpusTopics(dIndex).get(wIndex) + " ")
            }
            writer.write("\n")
        }
        writer.close()
    }

    def writeTopTopicalWords(name: String) {
        val writer = new BufferedWriter(new FileWriter(params.modelFileName + ".welda-" + name + ".topics"))
        for (tIndex <- 0 until params.numTopics) {
            writer.write(String.valueOf(tIndex))
            var topicWordProbs = mutable.Map[Integer, Double]()
            for (wIndex <- 0 until model.vocabularySize) {
                val pro = (model.topicWordCountLDA(tIndex)(wIndex) + params.beta) /
                        (model.sumTopicWordCountLDA(tIndex) + model.betaSum)
                topicWordProbs(wIndex) = pro
            }
            val TOP_WORDS = 10
            val mostLikelyWords = topicWordProbs.toSeq.sortBy(-_._2).take(TOP_WORDS)
//            topicWordProbs = FuncUtils.sortByValueDescending(topicWordProbs)
//            val mostLikelyWords = topicWordProbs.keySet
            var count = 0
            for (wordId <- mostLikelyWords) {
                writer.write(" " + model.id2WordVocabulary(wordId._1))
                count += 1
            }
            writer.write("\n")
        }
        writer.close()
    }

    def writeTopicWordPros(name: String) {
        val writer = new BufferedWriter(new FileWriter(params.modelFileName + ".welda-" + name + ".phi"))
        for (t <- 0 until params.numTopics) {
            for (w <- 0 until model.vocabularySize) {
                val pro = (model.topicWordCountLDA(t)(w) + params.beta) /
                        (model.sumTopicWordCountLDA(t) + model.betaSum)
                writer.write(pro + " ")
            }
            writer.write("\n")
        }
        writer.close()
    }

    def writeDocTopicPros(name: String) {
        val writer = new BufferedWriter(new FileWriter(params.modelFileName + ".welda-" + name + ".theta"))
        for (i <- 0 until params.numDocuments) {
            for (j <- 0 until params.numTopics) {
                val pro = (model.docTopicCount(i)(j) + params.alpha) / (model.sumDocTopicCount(i) + model.alphaSum)
                writer.write(pro + " ")
            }
            writer.write("\n")
        }
        writer.close()
    }

    def write(name: String) {
        writeTopTopicalWords(name)
        if (name == "final") {
            writeDocTopicPros(name)
            writeTopicAssignments(name)
            writeTopicWordPros(name)
        }
    }
}

