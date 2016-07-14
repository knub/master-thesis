package knub.master_thesis

import java.io._

import cc.mallet.topics.ParallelTopicModel
import knub.master_thesis.probabilistic.Divergence._

import scala.collection.JavaConversions._
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

    case class WordSimilarity(word1: String, word2: String, humanSim: Double, topicSim: Double = -1.0)
    def calculateWordSimilarity(args: Args, res: TopicModelResult): Unit = {
        case class SimilarityObject(simFile: String, fileEnding: String, extractor: Array[String] => WordSimilarity)
        val scws =
            SimilarityObject("SCWS/ratings.txt", ".scws", { x =>
                WordSimilarity(x(1).toLowerCase(), x(3).toLowerCase(), x(7).toDouble)
            })
        val wordsim353All =
            SimilarityObject("wordsim353_sim_rel/wordsim_all_goldstandard.txt", ".wordsim353-all",
                { x => WordSimilarity(x(0).toLowerCase(), x(1).toLowerCase(), x(2).toDouble) } )
        val wordsim353Sim =
            SimilarityObject("wordsim353_sim_rel/wordsim_similarity_goldstandard.txt", ".wordsim353-sim",
                { x => WordSimilarity(x(0).toLowerCase(), x(1).toLowerCase(), x(2).toDouble) } )
        val wordsim353Rel =
            SimilarityObject("wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt", ".wordsim353-rel",
                { x => WordSimilarity(x(0).toLowerCase(), x(1).toLowerCase(), x(2).toDouble) } )

        val topicProbs = res.getNormalizedWordTopics
        for (simObject <- List(scws, wordsim353All, wordsim353Sim, wordsim353Rel)) {
            println(simObject.simFile)

            val lines = Source.fromFile(s"../../data/word-similarity/${simObject.simFile}").getLines().toList
            val sims = lines.map { line =>
                val split = line.split("\t")
                simObject.extractor(split)
            }

            for (simFunction <- List(simSum, simBhattacharyya, simHelling, simJensenShannon)) {
                println(simFunction.name)
                val topicSims = sims.flatMap { sim =>
                    val idx1 = res.dataAlphabet.lookupIndex(sim.word1, false)
                    val idx2 = res.dataAlphabet.lookupIndex(sim.word2, false)
                    if (idx1 != -1 && idx2 != -1) {
                        val divergence = simFunction.sim(topicProbs(idx1), topicProbs(idx2))
                        val similarity = 1 - divergence
                        List(sim.copy(topicSim = similarity))
                    } else {
                        if (idx1 == -1)
                            println(sim.word1)
                        if (idx2 == -1)
                            println(sim.word2)
                        None
                    }
                }
                val pw = args.getPrintWriterFor(s"${simObject.fileEnding}-${simFunction.name}")
                pw.println(List("word1", "word2", "human-sim", s"topic-sim-${simFunction.name}").mkString("\t"))
                topicSims.foreach { topicSim =>
                    pw.println(f"${topicSim.word1}\t${topicSim.word2}\t${topicSim.humanSim}\t${topicSim.topicSim * 10}%.2f")
                }
                pw.close()
            }
        }
    }

    def analyzeResult(res: TopicModelResult, args: Args): Unit = {
        val frequentWords = Source.fromFile("../../data/vocab.txt").getLines().toArray

//        println("Write top words")
//        writeTopWordsToTextFile(res, args)
        println("Write topic probs")
        val topicProbs = writeTopicProbsToFile(res, args, frequentWords.toSet)
        val simFinder = new SimilarWordFinder(res, args, frequentWords, topicProbs)
        simFinder.run()

        /*
         * WHAT TO DO:
         * Find variance of words with highest and lowest variance
         * Find word pairs, that have high sim in topic space (kl divergence) but large diff in WE
         * For each word, find similars and dissimilars
         * Sample similarity
         */
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
