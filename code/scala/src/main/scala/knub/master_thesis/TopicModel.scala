package knub.master_thesis

import java.nio.file.{Files, Paths}

import cc.mallet.topics.ParallelTopicModel
import cc.mallet.types.{Instance, InstanceList}

import scala.collection.JavaConverters._

class TopicModel {

    def run(dataFolderName: String): TopicModelResult = {
        val instances = new InstanceList(PreprocessingPipe.pipe)
        instances.addThruPipe(new WikiPlainTextIterator(dataFolderName))

        val numTopics = 100
        val model = new ParallelTopicModel(numTopics, 1.0, 0.01)
        model.addInstances(instances)
        model.setNumThreads(2)
        model.setNumIterations(50)
        model.estimate()

        new TopicModelResult(model)
    }
}
