package knub.master_thesis

import java.io._
import java.nio.file.Paths
import java.util.regex.Pattern

import cc.mallet.pipe.CharSequence2TokenSequence
import cc.mallet.topics.ParallelTopicModel
import cc.mallet.types.TokenSequence
import knub.master_thesis.preprocessing.DataIterators
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
    numIterations: Int = 50,
    numDocuments: Int = -1,
    alpha: Double = 0.01,
    beta: Double = 0.01,
    saveStep: Int = 50,
    inspectFileSuffix: String = "###") {
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
            "supply-tm-similarity", "embedding-lda",
            "inspect-topic-evolution"
        ).foreach { mode =>
            cmd(mode).action { (_, c) => c.copy(mode = mode) }
        }

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
        opt[Int]("num-documents").action { (x, c) =>
            c.copy(numDocuments = x) }
        opt[Double]("alpha").action { (x, c) =>
            c.copy(alpha = x) }
        opt[Double]("beta").action { (x, c) =>
            c.copy(beta = x) }
        opt[String]("inspect-file-suffix").action { (x, c) =>
            c.copy(inspectFileSuffix = x) }
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
                val res = trainAndSaveNewModel(args, args.alpha, args.beta)
                println("Top words")
                writeTopWordsToTextFile(res, args)
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
            case "embedding-lda" =>
                val embeddingLDA = new WordEmbeddingLDA(args)
                embeddingLDA.inference()
            case "inspect-topic-evolution" =>
                inspectTopicEvolution(args)
        }
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

    def analyzeResult(res: TopicModelResult, args: Args): Unit = {
        val frequentWords = Source.fromFile(args.modelFileName + ".vocab").getLines().toArray

        println("Write top words")
        writeTopWordsToTextFile(res, args)
        println("Write vocabulary")
        writeVocabulary(res, args)
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

    def writeTopWordsToTextFile(res: TopicModelResult, args: Args): Unit = {
        val pw = args.getPrintWriterFor(".ssv")
        pw.write(res.displayTopWords(10))
        pw.close()
    }

    def writeVocabulary(res: TopicModelResult, args: Args) = {
        val pw = args.getPrintWriterFor(".vocab")
        res.dataAlphabet.toArray.foreach { word =>
            pw.println(word)
        }
        pw.close()
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


    def calculateWordSimilarity(args: Args, res: TopicModelResult): Unit = {
        case class WordSimilarity(word1: String, word2: String, humanSim: Double, topicSim: Double = -1.0)
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


    val START_STRIKETHROUGH = "\u001B[09m"
    val STOP_STRIKETHROUGH = "\u001B[29m"
    val START_BOLD = "\u001B[01m"
    val STOP_BOLD = "\u001B[22m"
    case class Topic(id: Int, words: scala.collection.mutable.Buffer[String])
    def inspectTopicEvolution(args: Args): Unit = {
        val fileSuffix = s"${args.inspectFileSuffix}-"
        val topicEvolutionFiles = new File(new File(args.modelFileName).getParent).listFiles().filter { file =>
            file.getAbsolutePath.startsWith(args.modelFileName) && file.getAbsolutePath.contains(fileSuffix)
        }.sorted
        val source = Source.fromFile(args.modelFileName + ".ssv").getLines.drop(1)
        val topics = source.toBuffer[String].map { line =>
            val split = line.split(" ")
            Topic(split(0).toInt, split.drop(2).toBuffer)
        }.sortBy(_.id).toArray

        val numTopics = topics.length

        val changedTopicsPerEvolution = topicEvolutionFiles.map { topicFile =>
            readTopicsFromLFLDA(topicFile.getAbsolutePath)
        }

        for (i <- 0 until numTopics) {
            val originalWords = topics(i).words
            for (j <- changedTopicsPerEvolution.indices) {
                val changedWords = changedTopicsPerEvolution(j)(i).words
                // hacky way to parse the iteration number from the file name
                val fileName = Paths.get(topicEvolutionFiles(j).getAbsolutePath).getFileName.toString.split('.').dropRight(1).last.replace(fileSuffix, "")
                print(f"${i + 1}%2d $fileName ")
                for (word <- originalWords) {
                    if (changedWords.contains(word)) {
                        print(word)
                    } else {
                        print(s"$START_STRIKETHROUGH$word$STOP_STRIKETHROUGH")
                    }
                    print(" ")
                }
                changedWords.filter { word => !originalWords.contains(word) }.foreach { word =>
                    print(s"$START_BOLD$word$STOP_BOLD ")
                }
                println()
            }
        }
    }

    def readTopicsFromLFLDA(fileName: String): Array[Topic] = {
        Source.fromFile(fileName).getLines().map { line =>
            val words = line.split(" ").toBuffer
            Topic(-1, words)
        }.toArray
    }
}
