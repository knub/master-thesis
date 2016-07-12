package knub.master_thesis

import java.io._
import java.util.Comparator

import cc.mallet.topics.ParallelTopicModel
import com.google.common.collect.MinMaxPriorityQueue
import knub.master_thesis.probabilistic.Divergence.jensenShannonDivergence

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
    numIterations: Int = 50) {

    def getPrintWriterFor(extension: String): PrintWriter = {
        val modelFile = new File(modelFileName)
        val modelTextFile = new File(modelFile.getCanonicalPath + extension)
        val pw = new PrintWriter(modelTextFile)
        pw
    }
}

object Main {

    val parser = new scopt.OptionParser[Args]("topic-models") {
        head("topic-models", "0.0.1")

        cmd("topic-model").action { (_, c) => c.copy(mode = "topic-model") }
        cmd("text-preprocessing").action { (_, c) => c.copy(mode = "text-preprocessing") }
        cmd("word-similarity").action { (_, c) => c.copy(mode = "word-similarity") }

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
                if (args.createNewModel) {
//                    for (alpha <- List(1.0 / 10.0,
//                        1.0 / 100.0,
//                        1.0 / 500.0)) {
//                        for (beta <- List(1.0 / 10.0,
//                            1.0 / 100.0,
//                            1.0 / 500.0)) {
//                            val f = args.modelFileName.replace(".model", s".alpha-${alpha.toString.replace(".", "0")}.beta-${beta.toString.replace(".", "0")}.model")
//                            val res = trainAndSaveNewModel(args, alpha, beta)
//                            analyzeResult(res, args.copy(modelFileName = f))
//                        }
//                    }
//                    val f = args.modelFileName.replace(".model", s".alpha-${alpha.toString.replace(".", "0")}.beta-${beta.toString.replace(".", "0")}.model")
                    val res = trainAndSaveNewModel(args, 1.0 / 256.0, 1.0 / 256.0)
                }
                else {
                    val res = loadExistingModel(args.modelFileName)
                    analyzeResult(res, args)
                }
            case "text-preprocessing" =>
                writeArticlesTextFile(args)
            case "word-similarity" =>
                val res = loadExistingModel(args.modelFileName)
                calculateWordSimilarity(args, res)
        }
    }

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

    case class WordSimilarity(word1: String, word2: String, humanSim: Double, topicSim: Double = -1.0)
    def calculateWordSimilarity(args: Args, res: TopicModelResult): Unit = {
        for (simType <- List("similarity", "relatedness")) {
            println(s"Word $simType")
            val lines = Source.fromFile(
                s"../../data/word-similarity/wordsim353_sim_rel/wordsim_${simType}_goldstandard.txt").getLines()
            val sims = lines.map { line =>
                val split = line.split("\t")
                WordSimilarity(split(0), split(1), split(2).toDouble)
            }
            val topicProbs = res.getNormalizedWordTopics
            val topicSims = sims.flatMap { sim =>
                val idx1 = res.dataAlphabet.lookupIndex(sim.word1.toLowerCase(), false)
                val idx2 = res.dataAlphabet.lookupIndex(sim.word2.toLowerCase(), false)
                if (idx1 != -1 && idx2 != -1) {
                    val divergence = jensenShannonDivergence(topicProbs(idx1), topicProbs(idx2))
                    List(sim.copy(topicSim = 1 - divergence))
                } else {
                    if (idx1 == -1)
                        println(sim.word1)
                    if (idx2 == -1)
                        println(sim.word2)
                    None
                }
            }
            val pw = args.getPrintWriterFor(s".wordsim353-$simType")
            pw.println(List("word1", "word2", "human-sim", "topic-sim").mkString("\t"))
            topicSims.foreach { topicSim =>
                pw.println(f"${topicSim.word1}\t${topicSim.word2}\t${topicSim.humanSim}\t${topicSim.topicSim * 10}%.2f")
            }
            pw.close()
        }
    }

    def analyzeResult(res: TopicModelResult, args: Args): Unit = {
        val frequentWords = Source.fromFile("../../data/vocab.txt").getLines().toArray

        println("Write top words")
        writeTopWordsToTextFile(res, args)
        println("Write topic probs")
        val topicProbs = writeTopicProbsToFile(res, args, frequentWords.toSet)
        println("Find word pairs")
        findWordPairs(res, args, frequentWords, topicProbs)

        /*
         * WHAT TO DO:
         * Find variance of words with highest and lowest variance
         * Find word pairs, that have high sim in topic space (kl divergence) but large diff in WE
         * For each word, find similars and dissimilars
         * Sample similarity
         */
    }

    def findWordPairs(res: TopicModelResult, args: Args, frequentWordsRaw: Array[String], topicProbs: Array[Array[Double]]): Unit = {
        val topWordsRaw = res.getTopWords(10)

        val topWordsAlphabet = mutable.Map[String, Int]()
        topWordsRaw.foreach { word => topWordsAlphabet += word -> res.dataAlphabet.lookupIndex(word, false) }
        val frequentWordsAlphabet = mutable.Map[String, Int]()
        frequentWordsRaw.foreach { word => frequentWordsAlphabet += word -> res.dataAlphabet.lookupIndex(word, false) }

        val topWords = topWordsRaw.filter { word => topWordsAlphabet(word) >= 0}
        val frequentWords = frequentWordsRaw.filter { word => frequentWordsAlphabet(word) >= 0}.take(150000)

        val topWordsCount = topWords.length
        val frequentWordsCount = frequentWords.length
        println(s"Top-words: $topWordsCount")
        println(s"Frequent-words: $frequentWordsCount")

        val progress = new util.Progress(topWordsCount.toLong * frequentWordsCount.toLong, -1)
        val pw = args.getPrintWriterFor(".similars")

        for (i <- 0 until topWordsCount) {
            val wordI = topWords(i)
            val idxI = topWordsAlphabet(wordI)
            val probsI = topicProbs(idxI)
            val topQueue = MinMaxPriorityQueue.orderedBy(new WordPairComparator)
                .maximumSize(10)
                .create[WordPair]()

            for (j <- 0 until frequentWordsCount) {
                progress.report_progress()

                val wordJ = frequentWords(j)
                if (wordI != wordJ) {
                    val idxJ = frequentWordsAlphabet(wordJ)
                    val probsJ = topicProbs(idxJ)
                    val divergence: Double = jensenShannonDivergence(probsI, probsJ)
                    topQueue.add(WordPair(wordI, wordJ, divergence))
                }
            }
            topQueue.toList.sortBy(_.divergence).foreach { wordPair =>
                val word1 = wordPair.word1
                val word2 = wordPair.word2
                pw.println(f"${wordPair.divergence}%.9f\tSIM\t$word1\t$word2")
            }
        }
        pw.close()
    }

    def writeTopicProbsToFile(res: TopicModelResult, args: Args, frequentWords: Set[String]): Array[Array[Double]] = {
        val pw = args.getPrintWriterFor(".topic-probs-normalized")

        val m = res.getNormalizedWordTopics
        println(s"Topics: ${m(0).length}")
        println(s"Tokens: ${res.dataAlphabet.size}")

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
        val pw = args.getPrintWriterFor(".ssv")
        pw.write(res.displayTopWords(10))
        pw.close()
    }


    def trainAndSaveNewModel(args: Args, alpha: Double, beta: Double): TopicModelResult = {
        val tp = new TopicModel(args, alpha, beta)
        val res = tp.run(args.dataFolderName, args.stopWordsFileName)
        val f = args.modelFileName//.replace(".model", s".alpha-${alpha.toString.replace(".", "0")}.beta-${beta.toString.replace(".", "0")}.model")
        res.save(f)
        res
    }

    def loadExistingModel(modelFileName: String): TopicModelResult = {
        new TopicModelResult(ParallelTopicModel.read(new File(modelFileName)))
    }

    def writeArticlesTextFile(args: Args): Unit = {
        val wpti = OnlyNormalPagesIterator.normalPagesIterator(new WikiPlainTextIterator(args.dataFolderName))
        val writer = new OutputStreamWriter(new FileOutputStream(args.modelFileName), "UTF-8")
        wpti.foreach { article =>
            writer.write(article.getData.asInstanceOf[String])
            writer.write("\n")
        }

    }

}
