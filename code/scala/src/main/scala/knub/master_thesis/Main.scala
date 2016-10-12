package knub.master_thesis

import java.io._
import java.nio.file.Paths
import java.util.Random
import java.util.regex.Pattern

import cc.mallet.pipe.CharSequence2TokenSequence
import cc.mallet.topics.ParallelTopicModel
import cc.mallet.types.{FeatureSequence, TokenSequence}
import knub.master_thesis.preprocessing.DataIterators
import knub.master_thesis.probabilistic.Divergence._
import knub.master_thesis.util.Word2VecUtils
import knub.master_thesis.welda.{GaussianKMeansWELDA, GaussianWELDA, SimpleSimBasedReplacementWELDA, VmfWELDA}
import org.apache.commons.io.FileUtils
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import weka.classifiers.functions.SMO
import weka.core.{Attribute, DenseInstance, Instances}

import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import scala.collection.parallel.ForkJoinTaskSupport
import scala.io.Source

case class Args(
    mode: String = "",
    modelFileName: String = "/home/knub/Repositories/master-thesis/models/topic-models/topic.model",
    embeddingFileName: String = "NOT SET",
    dataFolderName: String = "/home/knub/Repositories/master-thesis/code/resources/plain-text-test",
    stopWordsFileName: String = "../resources/stopwords.txt",
    conceptCategorizationFileName: String = "../../data/concept-categorization/battig_concept-categorization.tsv",
    inspectionFolder: String = "NOT SET",
    numThreads: Int = 2,
    numTopics: Int = -1,
    numIterations: Int = 50,
    numDocuments: Int = -1,
    alpha: Double = 0.02,
    beta: Double = 0.02,
    alpha0Boost: Double = 1.0,
    lambda: Double = -1.0,
    kappaFactor: Int = 5,
    saveStep: Int = 50,
    // replacement sampling
    pcaDimensions: Int = 10,
    distributionEstimationSamples: Int = 20,
    inspectFileContains: String = "###",
    weldaDistanceFunction: String = "cos") {

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
            "topic-model-experiment",
            "text-preprocessing", "word-similarity",
            "supply-tm-similarity", "welda-sim",
            "welda-gaussian", "welda-vmf", "welda-gaussian-kmeans",
            "inspect-topic-evolution",
            "20news-test", "20news-document-classification",
            "avg-embedding"
        ).foreach { mode =>
            cmd(mode).action { (_, c) => c.copy(mode = mode) }
        }

        opt[String]("model-file-name").action { (x, c) =>
            c.copy(modelFileName = x) }
        opt[String]("embedding-file-name").action { (x, c) =>
            c.copy(embeddingFileName = x) }
        opt[String]("data-folder-name").action { (x, c) =>
            c.copy(dataFolderName = x) }
        opt[String]("inspection-folder").action { (x, c) =>
            c.copy(inspectionFolder = x) }
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
        opt[Int]("save-step").action { (x, c) =>
            c.copy(saveStep = x) }
        opt[Double]("alpha").action { (x, c) =>
            c.copy(alpha = x) }
        opt[Double]("beta").action { (x, c) =>
            c.copy(beta = x) }
        opt[Double]("lambda").action { (x, c) =>
            c.copy(lambda = x) }
        opt[String]("inspect-file-contains").action { (x, c) =>
            c.copy(inspectFileContains = x) }
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
                /*
                for (alpha <- List(0.02)) {
                    for (beta <- List(0.02)) {
//                for (alpha <- List(0.002, 0.005, 0.01, 0.02, 0.05, 0.1)) {
//                    for (beta <- List(0.002, 0.005, 0.01, 0.02, 0.05, 0.1)) {
                        val folder = s"/data/wikipedia/2016-06-21/topic-models/topic.20news.250-1500" +
                            s".alpha-${alpha.toString.replace('.', '-')}" +
                            s".beta-${beta.toString.replace('.', '-')}"
                        println(folder)
                        new File(folder).mkdir()
                        if (!new File(s"$folder/model.ssv").exists()) {
                            val startTime = System.currentTimeMillis()
                            val newArgs = args.copy(alpha = alpha, beta = beta, modelFileName = s"$folder/model")
                            val res = trainAndSaveNewModel(newArgs)
                            val endTime = System.currentTimeMillis()
                            val duration = (endTime - startTime) / 1000
                            println(s"Learning took $duration s")
                            println("Write vocabulary")
                            writeVocabulary(res, newArgs)
                            println("Top words")
                            writeTopWordsToTextFile(res, newArgs)
                            println(res.displayTopWords(10))
                            FileUtils.writeStringToFile(new File(newArgs.modelFileName + ".runtime"), duration.toString)
                        } else {
                            println(s"$folder results already exist")
                        }
                    }
                }
                */

                val startTime = System.currentTimeMillis()
                val res = trainAndSaveNewModel(args)
                val endTime = System.currentTimeMillis()
                val duration = (endTime - startTime) / 1000
                println(s"Learning took $duration s")
                println("Write vocabulary")
                writeVocabulary(res, args)
                println("Top words")
                writeTopWordsToTextFile(res, args)
                println(res.displayTopWords(10))
                FileUtils.writeStringToFile(new File(args.modelFileName + ".runtime"), duration.toString)
            case "topic-model-load" =>
                val res = loadExistingModel(args.modelFileName)
                analyzeResult(res, args)
            case "topic-model-experiment" =>
                val alpha = 0.02
                val beta = 0.02

                val alpha0Boosts = List(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 100, 200, 300, 400)
                val THREADS = 5
                val alpha0BoostsPar = if (THREADS == 1) {
                    alpha0Boosts
                } else {
                    val par = alpha0Boosts.par
                    par.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(THREADS))
                    par
                }

                alpha0BoostsPar.foreach { alpha0Boost =>
                    val folder = s"/data/wikipedia/2016-06-21/topic-models/topic.20news.experiment.50-1500" +
                        s".alpha-${alpha.toString.replace('.', '-')}" +
                        s".beta-${beta.toString.replace('.', '-')}" +
                        f".alphazero-$alpha0Boost%03d"
                    println(folder)
                    new File(folder).mkdir()
                    if (!new File(s"$folder/model.ssv").exists()) {
                        val startTime = System.currentTimeMillis()
                        val newArgs = args.copy(alpha = alpha, beta = beta, alpha0Boost = alpha0Boost,
                            modelFileName = s"$folder/model")
                        val res = trainAndSaveNewModel(newArgs)
                        val documentTopics = res.model.getDocumentTopics(true, false)

                        val topic0AvailableCount = documentTopics.count { doc => doc(0) > 0.0 }
                        val topic0Available = topic0AvailableCount.toDouble / documentTopics.length
                        val topic0Average = documentTopics.map(_(0)).sum / documentTopics.length

                        val topic0 = res.displayTopWords(50).lines.slice(1, 2).next
                        println(f"alpha0Boost = $alpha0Boost% 4d, " +
                            f"topic available: $topic0AvailableCount% 6d/${documentTopics.length} = $topic0Available%.2f, " +
                            f"topic average: $topic0Average%.5f, $topic0")

                        val endTime = System.currentTimeMillis()
                        val duration = (endTime - startTime) / 1000
                        println(s"Learning took $duration s")
                        writeVocabulary(res, newArgs)
                        writeTopWordsToTextFile(res, newArgs)
                        FileUtils.writeStringToFile(new File(newArgs.modelFileName + ".runtime"), duration.toString)
                    } else {
                        println(s"$folder results already exist")
                    }
                }
            case "text-preprocessing" =>
                writeArticlesTextFile(args)
            case "word-similarity" =>
                val res = loadExistingModel(args.modelFileName)
                calculateWordSimilarity(args, res)
            case "supply-tm-similarity" =>
                val res = loadExistingModel(args.modelFileName)
                supplyTopicModelSimilarity(args, res)
            case "welda-sim" =>
                for (lambda <- List(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) {
                    println(lambda)
                    val weldaSim = new SimpleSimBasedReplacementWELDA(args.copy(lambda = lambda))
                    weldaSim.inference()
                }
            case "welda-gaussian" =>
                val weldaGaussian = new GaussianWELDA(args)
                weldaGaussian.init()
                weldaGaussian.inference()
                System.exit(1)
                val THREADS = 20
                val lambdas = List(0.1, 0.5, 0.2)
                val embeddings = List(
                    ("/data/wikipedia/2016-06-21/embedding-models/dim-200.skip-gram.embedding", 11295),
                    ("/data/wikipedia/2016-06-21/embedding-models/20news.dim-50.skip-gram.embedding", 11294)
//                    "/data/wikipedia/2016-06-21/embedding-models/google.embedding"
                )
                val samplingParams = List(
                    (2, 3), (2, 5), (2, 10), (2, 20), (2, 50),
                    (3, 6), (3, 10),(3, 20), (3, 50),
                    (5, 10), (5, 20), (5, 30), (5, 50),
                    (10, 50))

                val cases = for (embedding <- embeddings; lambda <- lambdas; samplingParam <- samplingParams)
                    yield (lambda, embedding, samplingParam)

                cases.foreach(println)
                val parCases = if (THREADS == 1) {
                    cases
                } else {
                    val parCases = cases.par
                    parCases.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(THREADS))
                    parCases
                }

                parCases.foreach { case (lambda, embedding, samplingParam) =>
                    println(s"Starting lambda = $lambda, embedding = $embedding")
                    try {
                        val weldaGaussian = new GaussianWELDA(
                            args.copy(lambda = lambda,
                                embeddingFileName = embedding._1,
                                numDocuments = embedding._2,
                                pcaDimensions = samplingParam._1,
                                distributionEstimationSamples = samplingParam._2))
                        weldaGaussian.init()
                        weldaGaussian.inference()
                        println(s"Finshed lambda = $lambda, embedding = $embedding")
                    } catch {
                        case e: Exception =>
                            println(s"Failed lambda = $lambda, embedding = $embedding")
                            println(e)
                    }
                }
            case "welda-gaussian-kmeans" =>
                val weldaGaussian = new GaussianKMeansWELDA(args)
                weldaGaussian.init()
                weldaGaussian.inference()
                System.exit(1)
                val THREADS = 15
                val lambdas = List(0.5, 0.6, 0.8, 1.0, 0.3, 0.05, 0.1, 0.2, 0.0)
                val embeddings = List(
                    ("/data/wikipedia/2016-06-21/embedding-models/dim-200.skip-gram.embedding", 11295),
                    ("/data/wikipedia/2016-06-21/embedding-models/20news.dim-50.skip-gram.embedding", 11294)
                )

                val cases = for (embedding <- embeddings; lambda <- lambdas)
                    yield (lambda, embedding)

                cases.foreach(println)
                val parCases = if (THREADS == 1) {
                    cases
                } else {
                    val parCases = cases.par
                    parCases.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(THREADS))
                    parCases
                }

                parCases.foreach { case (lambda, embedding) =>
                    println(s"Starting lambda = $lambda, embedding = $embedding")
                    try {
                        val weldaKmeansGaussian = new GaussianKMeansWELDA(
                            args.copy(lambda = lambda,
                                embeddingFileName = embedding._1,
                                numDocuments = embedding._2))
                        weldaKmeansGaussian.init()
                        weldaKmeansGaussian.inference()
                        println(s"Finshed lambda = $lambda, embedding = $embedding")
                    } catch {
                        case e: Exception =>
                            println(s"Failed lambda = $lambda, embedding = $embedding")
                            println(e)
                    }
                }
            case "welda-vmf" =>
                val THREADS = 30
                val lambdas = List(0.5, 0.6, 0.8, 1.0, 0.3, 0.05, 0.1, 0.2, 0.0)
                val kappaFactors = List(1, 2, 3, 5, 10, 20, 50, 100)
                val embeddings = List(
                    ("/data/wikipedia/2016-06-21/embedding-models/dim-200.skip-gram.embedding", 11295),
                    ("/data/wikipedia/2016-06-21/embedding-models/20news.dim-50.skip-gram.embedding", 11294)
                    //                    "/data/wikipedia/2016-06-21/embedding-models/google.embedding"
                )
                val cases = for (embedding <- embeddings; lambda <- lambdas; kappaFactor <- kappaFactors)
                    yield (lambda, embedding, kappaFactor)

                cases.foreach(println)
                val parCases = if (THREADS == 1) {
                    cases
                } else {
                    val parCases = cases.par
                    parCases.tasksupport = new ForkJoinTaskSupport(new scala.concurrent.forkjoin.ForkJoinPool(THREADS))
                    parCases
                }

                parCases.foreach { case (lambda, embedding, kappaFactor) =>
                    println(s"Starting lambda = $lambda, embedding = $embedding")
                    try {
                        val weldaVmf = new VmfWELDA(
                            args.copy(lambda = lambda,
                                embeddingFileName = embedding._1,
                                numDocuments = embedding._2,
                                kappaFactor = kappaFactor)
                        )
                        weldaVmf.init()
                        weldaVmf.inference()

                        println(s"Finshed lambda = $lambda, embedding = $embedding")
                    } catch {
                        case e: Exception =>
                            println(s"Failed lambda = $lambda, embedding = $embedding, $e: ${e.getMessage}")
                    }
                }
//                val weldaVmf = new VmfWELDA(args)
//                weldaVmf.init()
//                weldaVmf.inference()

            case "inspect-topic-evolution" =>
                inspectTopicEvolution(args)
            case "20news-test" =>
                run20NewsTest(args)
            case "20news-document-classification" =>
                run20NewsDocumentClassification(args)
            case "avg-embedding" =>
                val embeddingName = new File(args.embeddingFileName).getName
                val word2Vec = WordVectorSerializer.loadTxtVectors(
                    new File(s"${args.modelFileName.replaceAll("/model$", "/")}$embeddingName.restricted.vocab.embedding.txt"))
                val embeddingDimensions = word2Vec.getWordVector("house").length
                val tm = ParallelTopicModel.read(new File(args.modelFileName))
                val pw = new PrintWriter(s"${args.modelFileName}.$embeddingName.avg-embedding")

                val data = tm.getData
                data.foreach { doc =>
                    val avgVector = new Array[Double](embeddingDimensions)
                    val wordFeatures = doc.instance.getData.asInstanceOf[FeatureSequence]
                    val alph = doc.instance.getDataAlphabet

                    var nrWordsWithVectors = 0
                    wordFeatures.getFeatures.foreach { wordId =>
                        try {
                            val word = alph.lookupObject(wordId).asInstanceOf[String]
                            val actualWord = Word2VecUtils.findActualWord(word2Vec, word)
                            val wordVector = word2Vec.getWordVector(actualWord)
                            for (i <- 0 until embeddingDimensions) {
                                avgVector(i) += wordVector(i)
                            }
                            nrWordsWithVectors += 1
                        } catch {
                            case e: RuntimeException =>
                        }
                    }
                    for (i <- 0 until embeddingDimensions) {
                        avgVector(i) /= nrWordsWithVectors
                    }
                    val clazz = doc.instance.getTarget
                    pw.println(s"$clazz ${avgVector.mkString(" ")}")
                }
                pw.close()
        }
    }

    def trainAndSaveNewModel(args: Args): TopicModelResult = {
        val (instancesIterator, stopWordsFileName) = DataIterators.getIteratorForDataFolderName(args.dataFolderName)
        val tp = new TopicModel(args, args.alpha, args.beta, instancesIterator)
        println(s"Using stopword list from $stopWordsFileName")
        val res = tp.run(stopWordsFileName, args.alpha0Boost)
        res.save(args.modelFileName)
        res
    }

    def loadExistingModel(modelFileName: String): TopicModelResult = {
        new TopicModelResult(ParallelTopicModel.read(new File(modelFileName)))
    }

    def analyzeResult(res: TopicModelResult, args: Args): Unit = {
        println("Write vocabulary")
        writeVocabulary(res, args)
        val frequentWords = Source.fromFile(args.modelFileName + ".vocab").getLines().toArray

        println("Alpha")
        println(res.model.alpha.deep)
        println("Beta")
        println(res.model.beta)
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
        val (iterator, _) = DataIterators.getIteratorForDataFolderName(args.dataFolderName)
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
        val SIM_TYPE = "most-similar"
        val (topicProbs, _) = res.getNormalizedWordTopics
        val simFunction = simBhattacharyya
        println(simFunction.name)

        val embeddingName = Paths.get(args.embeddingFileName).getFileName.toString

        val fileWithTmSims = new File(s"${args.modelFileName}.$embeddingName.similarities-$SIM_TYPE.with-tm")
        val pw = new PrintWriter(fileWithTmSims)


        Source.fromFile(s"${args.modelFileName}.$embeddingName.similarities-$SIM_TYPE").getLines().foreach { line =>
            val Array(word1, word2, embeddingProb) = line.split("\t")
            val idx1 = res.dataAlphabet.lookupIndex(word1.toLowerCase(), false)
            val idx2 = res.dataAlphabet.lookupIndex(word2.toLowerCase(), false)
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


    val START_STRIKETHROUGH = "\u001B[03m"
    val STOP_STRIKETHROUGH = "\u001B[23m"
    val START_BOLD = "\u001B[01m"
    val STOP_BOLD = "\u001B[22m"
    case class Topic(id: Int, words: scala.collection.mutable.Buffer[String])
    def inspectTopicEvolution(args: Args): Unit = {
        val topicEvolutionFiles = new File(args.inspectionFolder).listFiles().filter { file =>
            file.getAbsolutePath.contains(args.inspectFileContains) &&
                file.getAbsolutePath.endsWith(".topics")
        }.sorted
        println(topicEvolutionFiles.map(_.getName).deep)
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
                val fileName = Paths.get(topicEvolutionFiles(j).getAbsolutePath).getFileName.toString.split('.').dropRight(1).last
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
            val words = line.split(" ").toBuffer.drop(1)
            Topic(-1, words)
        }.toArray
    }

    def run20NewsTest(args: Args): Unit = {
        val (instancesIterator, _) = DataIterators.getIteratorForDataFolderName(args.dataFolderName)
        val res = loadExistingModel(args.modelFileName)

        val pw = args.getPrintWriterFor("." + new File(args.dataFolderName).getName + ".predictions.lda")
        val inferencer = res.model.getInferencer
        instancesIterator.foreach { instance =>
            pw.println(inferencer.getSampledDistribution(instance, 20, 2, 10).mkString(" "))
        }
        pw.close()
    }

    def run20NewsDocumentClassification(args: Args): Unit = {
        val labels = Source.fromFile(args.dataFolderName)
            .getLines().map { line =>
            line.toInt
        }.toBuffer

        val instanceList = Source.fromFile(args.modelFileName).getLines().zipWithIndex.map { case (line, idx) =>
            val features = (line + " " + labels(idx).toString).split(" ").map(_.toDouble)
            val inst = new DenseInstance(1.0, features)
            inst
        }.toList
        val nrFeatures = instanceList.head.numValues() - 1

        assert(instanceList.size == labels.size)
        println(s"nrFeatures = $nrFeatures")

        val attributes = new java.util.ArrayList[Attribute]()
        for (i <- 0 until nrFeatures)
            attributes.add(new Attribute(i.toString))
        attributes.add(new Attribute("class", (0 until 20).map(_.toString).toList.asJava))

        val instances = new Instances("data", attributes, instanceList.size)
        instances.setClassIndex(nrFeatures)
        instanceList.foreach { instance =>
            instances.add(instance)
        }

        instances.randomize(new Random(21011991))
        println(s"nrInstances = ${instances.numInstances()}")

        val FOLDS = 10
        val percentages = for (i <- 0 until FOLDS) yield {
            val train = instances.trainCV(FOLDS, i)
            val test = instances.testCV(FOLDS, i)
//            println(train.numInstances())
//            println(test.numInstances())

            val smo = new SMO
            smo.buildClassifier(train)

            var nrCorrect = 0
            var nrIncorrect = 0
            test.enumerateInstances().toList.foreach { testInstance =>
                if (smo.classifyInstance(testInstance) == testInstance.classValue()) {
                    nrCorrect += 1
                } else {
                    nrIncorrect += 1
                }
            }

            val percentage = nrCorrect.toDouble / (nrCorrect + nrIncorrect)
//            println(s"$nrCorrect/${nrCorrect + nrIncorrect} = $percentage")
            percentage
        }

        println(s"Macro-averaged precision: ${percentages.sum / percentages.size}")

    }
}
