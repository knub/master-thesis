package knub.master_thesis

import java.io._
import java.util.Comparator

import cc.mallet.topics.ParallelTopicModel
import com.google.common.collect.MinMaxPriorityQueue

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.io.Source

case class Args(
    mode: String = "",
    modelFileName: String = "/home/knub/Repositories/master-thesis/models/topic-models/topic.model",
    dataFolderName: String = "/home/knub/Repositories/master-thesis/code/resources/plain-text-test",
    createNewModel: Boolean = false,
    stopWordsFileName: String = "../resources/stopwords.txt",
    conceptCategorizationFileName: String = "../../data/concept-categorization/battig_concept-categorization.tsv",
    numThreads: Int = 2,
    numTopics: Int = 256,
    numIterations: Int = 50)

object Main {

    implicit val m1 = mutable.Bag.configuration.compact[Int]

    val parser = new scopt.OptionParser[Args]("topic-models") {
        head("topic-models", "0.0.1")

        cmd("topic-model").action { (_, c) => c.copy(mode = "topic-model") }
        cmd("text-preprocessing").action { (_, c) => c.copy(mode = "text-preprocessing") }

        opt[String]('m', "model-file-name").action { (x, c) =>
            c.copy(modelFileName = x) }
        opt[String]('d', "data-folder-name").action { (x, c) =>
            c.copy(dataFolderName = x) }
        opt[String]("stop-words").action { (x, c) =>
            c.copy(stopWordsFileName = x) }
        opt[String]("concept-categorization").action { (x, c) =>
            c.copy(conceptCategorizationFileName = x) }
        opt[Int]("num-threads").action { (x, c) =>
            c.copy(numThreads = x) }
        opt[Int]("num-topics").action { (x, c) =>
            c.copy(numTopics = x) }
        opt[Int]("num-iterations").action { (x, c) =>
            c.copy(numIterations = x) }
        opt[Unit]("create-new-model").action { (_, c) =>
            c.copy(createNewModel = true) }
    }

    def main(args: Array[String]): Unit = {
        parser.parse(args, Args()) match {
            case Some(config) =>
                run(config)
            case None =>
        }
    }

    def run(args: Args): Unit = {
        args.mode match {
            case "topic-model" =>
                val res =
                    if (args.createNewModel)
                        trainAndSaveNewModel(args)
                    else
                        loadExistingModel(args.modelFileName)

                analyzeResult(res, args)
            case "text-preprocessing" =>
                writeArticlesTextFile(args)
        }
    }

    def kullbackLeibler(p: Array[Double], q: Array[Double]): Double = {
        var sum = 0.0
        for (i <- p.indices) {
            if (p(i) != 0.0 && q(i) != 0.0)
                sum += p(i) * Math.log(p(i) / q(i))
        }
        sum
    }

    case class WordConcept(word: String, concept: String)
    case class WordPair(word1: String, word2: String, divergence: Double)
    class WordPairComparator(ascending: Int = 1) extends Comparator[WordPair] {
        override def compare(wp1: WordPair, wp2: WordPair): Int = {
            if (wp1.divergence == wp2.divergence)
                0
            else if (wp1.divergence < wp2.divergence)
                -1 * ascending
            else
                +1 * ascending
        }
    }
    def analyzeResult(res: TopicModelResult, args: Args): Unit = {
        val frequentWords = Source.fromFile("../../data/vocab.txt").getLines().toArray

        writeTopWordsToTextFile(res, args)
        conceptCategorization(res, args)
        val topicProbs = writeTopicProbsToFile(res, args, frequentWords.toSet)
        findWordPairs(res, topicProbs)


        /*
         * WHAT TO DO:
         * Find variance of words with highest and lowest variance
         * Find word pairs, that have high sim in topic space (kl divergence) but large diff in WE
         */
    }

    def findWordPairs(res: TopicModelResult, topicProbs: Array[Array[Double]]): Unit = {
        val topQueue = MinMaxPriorityQueue.orderedBy(new WordPairComparator)
            .maximumSize(100)
            .create[WordPair]()
        val bottomQueue = MinMaxPriorityQueue.orderedBy(new WordPairComparator(-1))
            .maximumSize(100)
            .create[WordPair]()

        var time = System.currentTimeMillis()

        val topWords = res.getTopWords(30).filter { word =>
            res.dataAlphabet.lookupIndex(word, false) >= 0
        }.toArray

        println(s"Top-words: ${topWords.length}")

        // precompute alphabet mapping for performance (lookupIndex uses locks)
        val alphabetMapping = mutable.Map[String, Int]()
        topWords.foreach { word =>
            alphabetMapping += word -> res.dataAlphabet.lookupIndex(word, false)
        }

        val wordCount = topWords.length
        val totalNrPairs = wordCount.toLong * (wordCount - 1) / 2 // all combinations of size 2

        var c = 0
        for (i <- 0 until wordCount) {
            val wordI = topWords(i)
            val idxI = alphabetMapping(wordI)
            for (j <- 0 until i) {
                if (c % (totalNrPairs / 1000) == 0) {
                    val secondsSinceLast = Math.round((System.currentTimeMillis() - time) / 1000.0)
                    time = System.currentTimeMillis()
                    println(f"${100.0 * c / totalNrPairs}%.1f %% ($secondsSinceLast secs)")
                }
                c += 1

                val wordJ = topWords(j)
                val idxJ = alphabetMapping(wordJ)
                val divergence: Double = jensenShannonDivergence(topicProbs(idxI), topicProbs(idxJ))
                topQueue.add(WordPair(wordI, wordJ, divergence))
                bottomQueue.add(WordPair(wordI, wordJ, divergence))
            }
        }

        println("Most similar words:")
        topQueue.toList.sortBy(_.divergence).foreach { wordPair =>
            val word1 = wordPair.word1
            val word2 = wordPair.word2
            println(f"${wordPair.divergence}%.9f $word1 - $word2")
        }
        println("Most dissimilar words:")
        bottomQueue.toList.sortBy(-_.divergence).foreach { wordPair =>
            val word1 = wordPair.word1
            val word2 = wordPair.word2
            println(f"${wordPair.divergence}%.9f $word1 - $word2")
        }
    }

    def jensenShannonDivergence(p: Array[Double], q: Array[Double]): Double = {
        val m = new Array[Double](p.length)
        for (k <- p.indices)
            m(k) = 0.5 * (p(k) + q(k))

        val divergence = kullbackLeibler(p, m) + kullbackLeibler(q, m)
        //                        val divergence = Maths.jensenShannonDivergence(p, q)
        divergence
    }

    def writeTopicProbsToFile(res: TopicModelResult, args: Args, frequentWords: Set[String]): Array[Array[Double]] = {
        val modelFile = new File(args.modelFileName)
        val topicProbsFile = new File(modelFile.getCanonicalPath + ".topic-probs")

        val m = res.getWordTopics
        println(s"Topics: ${m(0).length}")
        println(s"Tokens: ${res.dataAlphabet.size}")

        val pw = new PrintWriter(topicProbsFile)
        pw.write(s"word,${(0 until res.model.numTopics).mkString(",")}\n")
        res.dataAlphabet.iterator().foreach { word =>
            if (frequentWords.contains(word)) {
                val idx = res.dataAlphabet.lookupIndex(word)
                val topicProbs = m(idx)

                pw.write(s"$word,${topicProbs.mkString(",")}\n")
            }
        }

        pw.close()
        m
    }

    def writeTopWordsToTextFile(res: TopicModelResult, args: Args): Unit = {
        val modelFile = new File(args.modelFileName)
        val modelTextFile = new File(modelFile.getCanonicalPath + ".ssv")
        val pw = new PrintWriter(modelTextFile)
        pw.write(res.displayTopWords(10))
        pw.close()
    }

    def conceptCategorization(res: TopicModelResult, args: Args): Unit = {
        val modelFile = new File(args.modelFileName)
        val purityTextFile = new File(modelFile.getCanonicalPath + ".purity")

        val out = new StringBuilder()
        val conceptCategorizationFile =
            args.conceptCategorizationFileName
        val concepts = Source.fromFile(conceptCategorizationFile).getLines().map { line =>
            val split = line.split("\t")
            WordConcept(split(0), split(1))
        }.toList.groupBy(_.concept)

        val purities = Array(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        concepts.foreach { case (concept, wordConcepts) =>
            val words = wordConcepts.map(_.word)
            out.append(s"Concept: $concept (${words.size} words): ${words.mkString(" ")}\n")
            for (n <- 1 to 5) {
                val wordTopics = words.map { word =>
                    (word, res.findBestTopicsForWord(word, nrTopics = n))
                }
                val topics = mutable.Bag[Int]()
                topics ++= wordTopics.flatMap(_._2)

                val topicWithMaxMultiplicity = topics.multiplicities.maxBy(_._2)._1
                val missingWords = wordTopics
                    .filter { case (word, currentWordTopics) => !currentWordTopics.contains(topicWithMaxMultiplicity) }
                    .map(_._1)
                val conceptPurity = topics.maxMultiplicity * 100.0 / words.length
                purities(n) += conceptPurity * words.size

                out.append(f"$n-purity: $conceptPurity%.1f %% -- missing words: $missingWords\n")
            }
        }
        out.append("#" * 100 + "\n")
        for (n <- 1 to 5) {
            purities(n) = purities(n) / concepts.values.map(_.size).sum
        }
        for (n <- 1 to 5) {
            out.append(f"$n-purity: ${purities(n)}%.1f %%\n")
        }
        out.append("#" * 100 + "\n")
        out.append(purities(1))


        val pw = new PrintWriter(purityTextFile)
        pw.write(out.toString())
        pw.close()
    }

    def trainAndSaveNewModel(args: Args): TopicModelResult = {
        val tp = new TopicModel(args)
        val res = tp.run(args.dataFolderName, args.stopWordsFileName)
        res.save(args.modelFileName)
        res
    }

    def loadExistingModel(modelFileName: String): TopicModelResult = {
        new TopicModelResult(ParallelTopicModel.read(new File(modelFileName)))
    }

    def writeArticlesTextFile(args: Args): Unit = {
        val wpti = new WikiPlainTextIterator(args.dataFolderName)
        val writer = new OutputStreamWriter(new FileOutputStream(args.modelFileName), "UTF-8")
        wpti.foreach { article =>
            writer.write(article.getData.asInstanceOf[String])
            writer.write("\n")
        }

    }

}
