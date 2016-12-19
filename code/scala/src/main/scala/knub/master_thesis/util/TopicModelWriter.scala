package knub.master_thesis.util

import java.io.{BufferedWriter, FileWriter}

import knub.master_thesis.welda.{BaseWELDA, ReplacementWELDA}

import scala.collection.mutable

class TopicModelWriter(private val model: ReplacementWELDA) {

    val params = model.p

    def baseName = model.fileBaseName

    def writeParameters(): Unit = {
        val writer = new BufferedWriter(new FileWriter(baseName + ".welda.params"))
        writer.write("-model" + "\t" + "WELDA")
        writer.write("\n-topicmodel" + "\t" + params.modelFileName)
        writer.write("\n-ntopics" + "\t" + model.numTopics)
        writer.write("\n-alpha" + "\t" + model.alpha.deep)
        writer.write("\n-beta" + "\t" + model.beta)
        writer.write("\n-niters" + "\t" + params.numIterations)
        if (params.saveStep > 0) writer.write("\n-sstep" + "\t" + params.saveStep)
        writer.close()
    }

    def writeTopicAssignments(name: String): Unit = {
        val writer = new BufferedWriter(new FileWriter(baseName + s".$name.topic-assignments"))
        for (dIndex <- 0 until params.numDocuments) {
            val docSize = model.corpusWords(dIndex).size
            for (wIndex <- 0 until docSize) {
                writer.write(model.corpusTopics(dIndex).get(wIndex) + " ")
            }
            writer.write("\n")
        }
        writer.close()
    }

    def writeTopTopicalWords(name: String, topWords: Int = 10): Unit = {
        val writer = if (topWords == 10)
            new BufferedWriter(new FileWriter(baseName + s".$name.topics"))
        else
            new BufferedWriter(new FileWriter(baseName + s".$name.topics.$topWords"))

        for (tIndex <- 0 until model.numTopics) {
            writer.write(String.valueOf(tIndex))
            val topicWordProbs = mutable.Map[Int, Double]()
            for (wIndex <- 0 until model.vocabularySize) {
                // TODO: Think about what is better sorting?
//                val pro = (model.topicWordCountLDA(tIndex)(wIndex) + params.beta) /
//                        (model.sumTopicWordCountLDA(tIndex) + model.betaSum)
                val pro = model.topicWordCountLDAAveraged(tIndex)(wIndex)
                topicWordProbs(wIndex) = pro
            }
            val mostLikelyWords = topicWordProbs.toSeq.sortBy(-_._2).take(topWords)
//            topicWordProbs = FuncUtils.sortByValueDescending(topicWordProbs)
//            val mostLikelyWords = topicWordProbs.keySet
            for (wordId <- mostLikelyWords) {
                writer.write(" " + model.id2WordVocabulary(wordId._1))
            }
            writer.write("\n")
        }
        writer.close()
    }

    def writeTopicWordPros(name: String): Unit = {
        val writer = new BufferedWriter(new FileWriter(baseName + s".$name.phi"))
        for (t <- 0 until model.numTopics) {
            for (w <- 0 until model.vocabularySize) {
                val pro = (model.topicWordCountLDAAveraged(t)(w) + params.beta) /
                        (model.sumTopicWordCountLDAAveraged(t) + model.betaSum)
                writer.write(pro + " ")
            }
            writer.write("\n")
        }
        writer.close()
    }

    def writeDocTopicProbs(name: String): Array[Array[Double]] = {
        val writer = new BufferedWriter(new FileWriter(baseName + s".$name.document-topics"))
        val docTopicMatrix = Array.ofDim[Double](params.numDocuments, model.numTopics)
        for (i <- 0 until params.numDocuments) {
            for (j <- 0 until model.numTopics) {
                val pro = (model.docTopicCountAveraged(i)(j) + model.alpha(j)) / (model.docWordCount(i) + model.alpha.sum)
                docTopicMatrix(i)(j) = pro
                if (j == 0)
                    writer.write(pro.toString)
                else
                    writer.write(" " + pro)
            }
            writer.write("\n")
        }
        writer.close()
        docTopicMatrix
    }

    def writePercentageOfTopWords(name: String, docTopicMatrix: Array[Array[Double]]): Unit = {
        val topWords = 10
        val writer = new BufferedWriter(new FileWriter(baseName + s".$name.percentage-of-top-words"))

        for (tIndex <- 0 until model.numTopics) {
            writer.write(String.valueOf(tIndex))
            val topicWordProbs = mutable.Map[Int, Double]()
            for (wIndex <- 0 until model.vocabularySize) {
                val pro = model.topicWordCountLDAAveraged(tIndex)(wIndex)
                topicWordProbs(wIndex) = pro
            }
            val mostLikelyWords = topicWordProbs.toSeq.sortBy(-_._2).take(topWords).map(_._1)
            for (wordId <- mostLikelyWords) {
                writer.write(" " + model.id2WordVocabulary(wordId))
            }
            writer.write("\n")

            val mostLikelyWordsSet = mostLikelyWords.toSet
            // find topics where tIndex is prominent
            val filtered = docTopicMatrix.zipWithIndex.filter { case (topicProbs, idx) =>
                topicProbs.zipWithIndex.maxBy(_._1)._2 == tIndex
            }.map(_._2)
            if (filtered.isEmpty) {
                writer.write("  Topic is not dominant topic anywhere\n")
            } else {
                var topWordCount = 0
                var totalCount = 0
                filtered.foreach { docId =>
                    val doc = model.corpusWords(docId)
                    totalCount += doc.size()
                    var i = 0
                    while (i < doc.size()) {
                        val w = doc.get(i)
                        if (mostLikelyWordsSet.contains(w)) {
                            topWordCount += 1
                        }
                        i += 1
                    }
                }
                writer.write(f"  $topWordCount / $totalCount = ${100 * topWordCount.toDouble / totalCount}%.2f %%\n")
            }
        }
        writer.close()
    }

    def writeRuntime() = {
        val endTime = System.currentTimeMillis()
        val durationInSec = (endTime - model.startTime) / 1000
        val writer = new BufferedWriter(new FileWriter(baseName + s".runtime"))
        writer.write(durationInSec.toString)
        writer.close()
    }

    def write(it: Int) {
        val name = f"iteration-$it%03d"
        writeTopTopicalWords(name)
        val docTopicMatrix = writeDocTopicProbs(name)
        if (it == model.p.numIterations) {
            writeTopTopicalWords(name, 500)
            writeTopicAssignments(name)
            writeTopicWordPros(name)
            writePercentageOfTopWords(name, docTopicMatrix)
            writeRuntime()
        }
    }
}

