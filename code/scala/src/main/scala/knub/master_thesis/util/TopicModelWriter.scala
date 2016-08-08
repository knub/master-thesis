package knub.master_thesis.util

import java.io.{BufferedWriter, FileWriter}

import knub.master_thesis.WordEmbeddingLDA

import scala.collection.mutable

class TopicModelWriter(private val model: WordEmbeddingLDA) {

    val params = model.p

    val modelFileName = s"${params.modelFileName}.${model.embeddingName}"

    def writeParameters() {
        val writer = new BufferedWriter(new FileWriter(modelFileName + ".welda.params"))
        writer.write("-model" + "\t" + "WELDA")
        writer.write("\n-topicmodel" + "\t" + params.modelFileName)
        writer.write("\n-ntopics" + "\t" + params.numTopics)
        writer.write("\n-alpha" + "\t" + model.alpha.deep)
        writer.write("\n-beta" + "\t" + model.beta)
        writer.write("\n-niters" + "\t" + params.numIterations)
        if (params.saveStep > 0) writer.write("\n-sstep" + "\t" + params.saveStep)
        writer.close()
    }

    def writeTopicAssignments(name: String) {
        val writer = new BufferedWriter(new FileWriter(modelFileName + ".welda-" + name + ".topic-assignments"))
        for (dIndex <- 0 until params.numDocuments) {
            val docSize = model.corpusWords(dIndex).size
            for (wIndex <- 0 until docSize) {
                writer.write(model.corpusTopics(dIndex).get(wIndex) + " ")
            }
            writer.write("\n")
        }
        writer.close()
    }

    def writeTopTopicalWords(name: String): Unit = {
        val writer = new BufferedWriter(new FileWriter(modelFileName + ".welda-" + name + ".topics"))
        for (tIndex <- 0 until params.numTopics) {
            writer.write(String.valueOf(tIndex))
            val topicWordProbs = mutable.Map[Int, Double]()
            for (wIndex <- 0 until model.vocabularySize) {
                // TODO: Think about what is better sorting?
//                val pro = (model.topicWordCountLDA(tIndex)(wIndex) + params.beta) /
//                        (model.sumTopicWordCountLDA(tIndex) + model.betaSum)
                val pro = model.topicWordCountLDA(tIndex)(wIndex)
                topicWordProbs(wIndex) = pro
            }
            val TOP_WORDS = 10
            val mostLikelyWords = topicWordProbs.toSeq.sortBy(-_._2).take(TOP_WORDS)
//            topicWordProbs = FuncUtils.sortByValueDescending(topicWordProbs)
//            val mostLikelyWords = topicWordProbs.keySet
            for (wordId <- mostLikelyWords) {
                writer.write(" " + model.id2WordVocabulary(wordId._1))
            }
            writer.write("\n")
        }
        writer.close()
    }

    def writeTopicWordPros(name: String) {
        val writer = new BufferedWriter(new FileWriter(modelFileName + ".welda-" + name + ".phi"))
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

    def writeDocTopicProbs(name: String) {
        val writer = new BufferedWriter(new FileWriter(modelFileName + ".welda-" + name + ".theta"))
        for (i <- 0 until params.numDocuments) {
            for (j <- 0 until params.numTopics) {
                val pro = (model.docTopicCount(i)(j) + params.alpha) / (model.docWordCount(i) + model.alphaSum)
                writer.write(pro + " ")
            }
            writer.write("\n")
        }
        writer.close()
    }

    def write(name: String) {
        writeTopTopicalWords(name)
        if (name == model.p.numIterations.toString) {
            writeDocTopicProbs(name)
            writeTopicAssignments(name)
            writeTopicWordPros(name)
        }
    }
}

