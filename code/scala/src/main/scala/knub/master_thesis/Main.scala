package knub.master_thesis

import java.io._

import cc.mallet.topics.ParallelTopicModel

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.io.Source

case class Args(
    mode: String = "",
    modelFileName: String = "/home/knub/Repositories/master-thesis/code/resources/topic.model",
    dataFolderName: String = "/home/knub/Repositories/master-thesis/code/resources/plain-text-test",
    createNewModel: Boolean = false,
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

                analyzeResult(res)
            case "text-preprocessing" =>
                writeArticlesTextFile(args)
        }
    }

    case class WordConcept(word: String, concept: String)
    def analyzeResult(res: TopicModelResult): Unit = {
        val conceptCategorizationFile =
            "../../data/concept-categorization/battig_concept-categorization.tsv"
        val concepts = Source.fromFile(conceptCategorizationFile).getLines().map { line =>
            val split = line.split("\t")
            WordConcept(split(0), split(1))
        }.toList.groupBy(_.concept)

        concepts.foreach { case (concept, wordConcepts) =>
            val words = wordConcepts.map(_.word)
            println(s"Concept: $concept (${words.size} words): ${words.mkString(" ")}")
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
                println(s"$n-purity: ${topics.maxMultiplicity * 100 / words.length} % -- missing words: $missingWords")
            }
        }
    }

    def trainAndSaveNewModel(args: Args): TopicModelResult = {
        val tp = new TopicModel(args)
        val res = tp.run(args.dataFolderName)
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
