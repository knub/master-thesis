package knub.master_thesis

import java.io.{File, PrintWriter}

import cc.mallet.topics.ParallelTopicModel
import cc.mallet.types.FeatureSequence

import scala.collection.mutable
import scala.io.Source

case class WordConcept(word: String, concept: String)


class TopicModelResult(val model: ParallelTopicModel) {
    // for using bags
    implicit val m1 = mutable.Bag.configuration.compact[Int]

    def save(fileName: String): Unit = {
        model.write(new File(fileName))
    }

    // The data alphabet maps word IDs to strings
    val dataAlphabet = model.alphabet

    /**
      * Show the words and topics in the given instance
      */
    def showInstance(instanceIdx: Int): Unit = {
        val tokens = model.getData.get(instanceIdx).instance.getData.asInstanceOf[FeatureSequence]
        val topics = model.getData.get(instanceIdx).topicSequence

        for (i <- 0 until tokens.getLength) {
            print(s"${dataAlphabet.lookupObject(tokens.getIndexAtPosition(i))}-${topics.getIndexAtPosition(i)} ")
        }
        println()
    }

    def getNormalizedWordTopics: Array[Array[Double]] = {
        val result = Array.ofDim[Double](model.numTypes, model.numTopics)
        for (wordType <- 0 until model.numTypes) {
            val topicCounts = model.typeTopicCounts(wordType)
            var index = 0
            while (index < topicCounts.length && topicCounts(index) > 0) {
                val topic = topicCounts(index) & model.topicMask
                val count = topicCounts(index) >> model.topicBits
                result(wordType)(topic) += count
                index += 1
            }
        }
        for (wordType <- 0 until model.numTypes) {
            for (topic <- 0 until model.numTopics) {
                result(wordType)(topic) += model.beta
            }
        }
        val topicNormalizers = Array.ofDim[Double](model.numTopics)
        for (topic <- 0 until model.numTopics) {
            topicNormalizers(topic) = 1.0 / (model.tokensPerTopic(topic) + model.numTypes * model.beta)
        }
        for (topic <- 0 until model.numTopics; wordType <- 0 until model.numTypes) {
            result(wordType)(topic) *= topicNormalizers(topic)
        }
        for (i <- result.indices) {
            result(i) = normalized(result(i))
        }
        result
    }

    private def normalized(a: Array[Double]): Array[Double] = {
        val s = a.sum
        for (i <- a.indices) {
            a(i) = a(i) / s
        }
        a
    }

    def showTopWordsPerTopics(): Unit = {
//        val stdout = new PrintWriter(System.out)
//        model.printDocumentTopics(stdout)
//        model.printTopicDocuments(stdout)
        model.printTopWords(System.out, 10, false)
    }

    def displayTopWords(numWords: Int = 10): String = {

        val orderedTopicIndices = model.tokensPerTopic.zipWithIndex.sortBy(-_._1)

        val out = new StringBuilder()
        val topicSortedWords = model.getSortedWords
        out.append(s"topic topic-count ${(0 until 10).map(_.toString).mkString(" ")}\n")
//        for (topic <- 0 until model.numTopics) {
        for ((tokenCount, topic) <- orderedTopicIndices) {
            val sortedWords = topicSortedWords.get(topic)
            var word = 0
            val iterator = sortedWords.iterator()

            out.append(s"$topic $tokenCount")
            while (iterator.hasNext && word < numWords) {
                val info = iterator.next()
                out.append(" " + dataAlphabet.lookupObject(info.getID))
                word += 1
            }
            out.append("\n")
        }
        out.toString
    }

    /**
      * Returns the set of the best words from all topics
      * @param numWords How many words for each topic
      * @return Set of the highest scoring words from all topics
      */
    def getTopWords(numWords: Int): Array[String] = {
        val s = mutable.ArrayBuffer[String]()
        val topicSortedWords = model.getSortedWords
        for (topic <- 0 until model.numTopics) {
            val sortedWords = topicSortedWords.get(topic)
            var word = 0
            val iterator = sortedWords.iterator()

            while (iterator.hasNext && word < numWords) {
                val info = iterator.next()
                s += dataAlphabet.lookupObject(info.getID).asInstanceOf[String]
                word += 1
            }
        }
        s.toArray
    }

    def findBestTopicsForWord(word: String, nrTopics: Int = 3): Array[Int] = {
        // The format for typeTopicCounts array is
        //  the topic in the rightmost bits
        //  the count in the remaining (left) bits.
        // Since the count is in the high bits, sorting (desc)
        //  by the numeric value of the int guarantees that
        //  higher counts will be before the lower counts.
        val idx = dataAlphabet.lookupIndex(word)
        model.typeTopicCounts(idx).take(nrTopics).filter(_ != 0).map(_ & model.topicMask)

        // OLD IMPLEMENTATION -- SLOW
//        val wordId = dataAlphabet.lookupIndex(word)
//        val topicSortedWords = model.getSortedWords
//        topicSortedWords.zipWithIndex.maxBy { case (topic, _) =>
//            topic.iterator()
//                .find { idSorter => idSorter.getID == wordId }
//                .map(_.getWeight)
//                .getOrElse(0.0)
//        }._2
    }

    def estimateTopicDistribution(): Unit = {
        // Estimate the topic distribution of the first instance,
        //  given the current Gibbs state.
        val topicDistribution = model.getTopicProbabilities(0)
        // Get an array of sorted sets of word ID/count pairs
        val topicSortedWords = model.getSortedWords

        // Show top 5 words in topics with proportions for the first document
        for (topic <- 0 until model.numTopics) {
            val iterator = topicSortedWords.get(topic).iterator()

            println(f"$topic\t${topicDistribution(topic)}%.3f\t")
            var rank = 0
            while (iterator.hasNext && rank < 5) {
                val idCountPair = iterator.next()
                println(s"${dataAlphabet.lookupObject(idCountPair.getID)} (${idCountPair.getWeight}%.0f) ")
                rank += 1
            }
        }

        // Create a new instance with high probability of topic 0
        val topicZeroText = new java.lang.StringBuilder()
        val iterator = topicSortedWords.get(0).iterator()

        var rank = 0
        while (iterator.hasNext && rank < 5) {
            val idCountPair = iterator.next()
            topicZeroText.append(dataAlphabet.lookupObject(idCountPair.getID) + " ")
            rank += 1
        }

//        // Create a new instance named "test instance" with empty target and source fields.
//        val testing = new InstanceList(PreprocessingPipe.pipe)
//        testing.addThruPipe(new Instance(topicZeroText.toString, null, "test instance", null))
//
//        val inferencer = model.getInferencer
//        val testProbabilities = inferencer.getSampledDistribution(testing.get(0), 10, 1, 5)
//        System.out.println("0\t" + testProbabilities(0))
    }
    def conceptCategorization(args: Args): Unit = {
        val modelFile = new File(args.modelFileName)
        val purityTextFile = new File(modelFile.getCanonicalPath + ".purity")

        val pw = new PrintWriter(purityTextFile)
        val conceptCategorizationFile =
            args.conceptCategorizationFileName
        val concepts = Source.fromFile(conceptCategorizationFile).getLines().map { line =>
            val split = line.split("\t")
            WordConcept(split(0), split(1))
        }.toList.groupBy(_.concept)

        val purities = Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        concepts.foreach { case (concept, wordConcepts) =>
            val words = wordConcepts.map(_.word)
            pw.println(s"Concept: $concept (${words.size} words): ${words.mkString(" ")}")
            for (n <- 1 to 5) {
                val wordTopics = words.map { word =>
                    (word, findBestTopicsForWord(word, nrTopics = n))
                }
                val topics = mutable.Bag[Int]()
                topics ++= wordTopics.flatMap(_._2)

                val topicWithMaxMultiplicity = topics.multiplicities.maxBy(_._2)._1
                val missingWords = wordTopics
                    .filter { case (word, currentWordTopics) => !currentWordTopics.contains(topicWithMaxMultiplicity) }
                    .map(_._1)
                val conceptPurity = topics.maxMultiplicity * 100.0 / words.length
                purities(n) += conceptPurity * words.size

                pw.println(f"$n-purity: $conceptPurity%.1f %% -- missing words: $missingWords")
            }
        }
        pw.println("#" * 100)
        for (n <- 1 to 5) {
            purities(n) = purities(n) / concepts.values.map(_.size).sum
        }
        for (n <- 1 to 5) {
            pw.println(f"$n-purity: ${purities(n)}%.1f %%")
        }
        pw.println("#" * 100)
        pw.println(purities(1))

        pw.close()
    }

}
