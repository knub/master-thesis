package knub.master_thesis

import java.io.File

import cc.mallet.topics.ParallelTopicModel

case class Args(
    modelFileName: String = "/home/knub/Repositories/master-thesis/code/scala/resources/topic.model",
    dataFolderName: String = "/home/knub/Repositories/master-thesis/code/scala/resources/plain-text-test",
    createNewModel: Boolean = false)

object Main {

    val parser = new scopt.OptionParser[Args]("topic-models") {
        head("topic-models", "0.01")

        opt[String]('m', "model-file-name").action( (x, c) =>
            c.copy(modelFileName = x))
        opt[String]('d', "data-folder-name").action( (x, c) =>
            c.copy(dataFolderName = x))
        opt[Unit]("create-new-model").action( (_, c) =>
            c.copy(createNewModel = true))
    }

    def main(args: Array[String]): Unit = {
        parser.parse(args, Args()) match {
            case Some(config) =>
                run(config)
            case None =>
        }
    }

    def run(args: Args): Unit = {
        val res =
            if (args.createNewModel)
                trainAndSaveNewModel(args)
            else
                loadExistingModel(args.modelFileName)

        res.showTopWordsPerTopics()
    }

    def trainAndSaveNewModel(args: Args): TopicModelResult = {
        val tp = new TopicModel
        val res = tp.run(args.dataFolderName)
        res.save(args.modelFileName)
        res
    }

    def loadExistingModel(modelFileName: String): TopicModelResult = {
        new TopicModelResult(ParallelTopicModel.read(new File(modelFileName)))
    }

}
