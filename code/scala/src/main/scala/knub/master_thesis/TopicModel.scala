package knub.master_thesis

import cc.mallet.topics.ParallelTopicModel
import cc.mallet.types.{Instance, InstanceList}

class TopicModel(args: Args, alpha: Double, beta: Double, instancesIterator: java.util.Iterator[Instance]) {

    def run(dataFolderName: String, stopWordsFileName: String): TopicModelResult = {
        val instances = new InstanceList(PreprocessingPipe.pipe(stopWordsFileName))
        instances.addThruPipe(instancesIterator)

        val numTopics = args.numTopics
        val model = new ParallelTopicModel(numTopics, numTopics * alpha, beta)
        model.setOptimizeInterval(10)
        model.setBurninPeriod(100)
        model.addInstances(instances)
        model.setNumThreads(args.numThreads)
        model.setNumIterations(args.numIterations)
        model.showTopicsInterval = 0
        model.estimate()

        new TopicModelResult(model)
    }
}
