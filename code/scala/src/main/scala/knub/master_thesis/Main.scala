package knub.master_thesis

import java.io._
import java.util.regex.Pattern

import cc.mallet.pipe.CharSequence2TokenSequence
import cc.mallet.topics.ParallelTopicModel
import cc.mallet.types.{Instance, TokenSequence}
import knub.master_thesis.probabilistic.Divergence._

import scala.collection.JavaConversions._
import scala.collection.parallel.ForkJoinTaskSupport
import scala.io.Source

case class Args(
    mode: String = "",
    modelFileName: String = "/home/knub/Repositories/master-thesis/models/topic-models/topic.model",
    dataFolderName: String = "/home/knub/Repositories/master-thesis/code/resources/plain-text-test",
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

        List("topic-model-create", "topic-model-load",
            "text-preprocessing", "word-similarity",
            "supply-tm-similarity"
        ).foreach { mode =>
            cmd(mode).action { (_, c) => c.copy(mode = mode) }
        }

        opt[String]('m', "model-file-name").action { (x, c) =>
            c.copy(modelFileName = x) }
        opt[String]('d', "data-folder-name").required().action { (x, c) =>
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
            case "topic-model-create" =>
                val res = trainAndSaveNewModel(args, 1.0 / 256.0, 1.0 / 256.0)
                println(res.displayTopWords(10))
            case "topic-model-load" =>
                val res = loadExistingModel(args.modelFileName)
                analyzeResult(res, args)
            case "text-preprocessing" =>
                writeArticlesTextFile(args)
            case "word-similarity" =>
                val res = loadExistingModel(args.modelFileName)
                calculateWordSimilarity(args, res)
            case "supply-tm-similarity" =>
                val res = loadExistingModel(args.modelFileName)
                supplyTopicModelSimilarity(args, res)
        }
    }

    def supplyTopicModelSimilarity(args: Args, res: TopicModelResult): Unit = {
        val (topicProbs, _) = res.getNormalizedWordTopics
        val simFunction = simBhattacharyya
        println(simFunction.name)

        val fileWithTmSims = new File(args.dataFolderName + ".with-tm")
        val pw = new PrintWriter(fileWithTmSims)

        Source.fromFile(args.dataFolderName).getLines().foreach { line =>
            val Array(word1, word2, embeddingProb) = line.split("\t")
            val idx1 = res.dataAlphabet.lookupIndex(word1, false)
            val idx2 = res.dataAlphabet.lookupIndex(word2, false)
            if (idx1 != -1 && idx2 != -1) {
                val divergence = simFunction.sim(topicProbs(idx1), topicProbs(idx2))
                val topicSimilarity = 1 - divergence
                pw.println(s"$word1\t$word2\t$embeddingProb\t$topicSimilarity")
            } else {
                if (idx1 == -1)
                    println(word1)
                if (idx2 == -1)
                    println(word2)
                None
            }
        }
        pw.close()
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

        val (topicProbs, _) = res.getNormalizedWordTopics
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

        println("Write top words")
        writeTopWordsToTextFile(res, args)
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

        // last value is absolute word probability
        val (m, wordProbs) = res.getNormalizedWordTopics
        println(s"Topics: ${m(0).length}")
        println(s"Tokens: ${res.dataAlphabet.size}")

        pw.write(s"word,${(0 until res.model.numTopics).mkString(",")},word-prob\n")
        val par = res.dataAlphabet.iterator().toList.par
        par.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(10))
        par.foreach { word =>
            if (frequentWords.contains(word)) {
                val idx = res.dataAlphabet.lookupIndex(word, false)
                val topicProbs = m(idx)

                pw.write(s"$word,${topicProbs.mkString(",")},${wordProbs(idx)}\n")
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
        val instancesIterator = DataIterators.getIteratorForDataFolderName(args.dataFolderName)
        val tp = new TopicModel(args, alpha, beta, instancesIterator)
        val res = tp.run(args.dataFolderName, args.stopWordsFileName)
        val f = args.modelFileName
        res.save(f)
        res
    }

    def loadExistingModel(modelFileName: String): TopicModelResult = {
        new TopicModelResult(ParallelTopicModel.read(new File(modelFileName)))
    }

    def writeArticlesTextFile(args: Args): Unit = {
        val iterator = DataIterators.getIteratorForDataFolderName(args.dataFolderName)
        val converter = new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}"))


        val writer = new OutputStreamWriter(new FileOutputStream(args.modelFileName), "UTF-8")
        iterator.foreach { article =>
            val convertedInstance = converter.pipe(article)
            convertedInstance.getData.asInstanceOf[TokenSequence].foreach { token =>
                writer.write(article.getData.asInstanceOf[String])
                writer.write("\n")
            }
            writer.write(article.getData.asInstanceOf[String])
            writer.write("\n")
        }
        writer.close()
    }

}
