package knub.master_thesis

import cc.mallet.topics.ParallelTopicModel
import cc.mallet.types.InstanceList

class TopicModel(args: Args) {

    def run(dataFolderName: String): TopicModelResult = {
        val instances = new InstanceList(PreprocessingPipe.pipe)
        instances.addThruPipe(new WikiPlainTextIterator(dataFolderName))

        val numTopics = args.numTopics
        val model = new ParallelTopicModel(numTopics, 1.0, 1.0 / args.numTopics)
        model.addInstances(instances)
        model.setNumThreads(args.numThreads)
        model.setNumIterations(args.numIterations)
        model.showTopicsInterval = 0
        model.estimate()

        new TopicModelResult(model)
    }
}
