package knub.master_thesis

import cc.mallet.topics.ParallelTopicModel
import cc.mallet.types.InstanceList

class TopicModel(args: Args) {

    def run(dataFolderName: String, stopWordsFileName: String): TopicModelResult = {
        val instances = new InstanceList(PreprocessingPipe.pipe(stopWordsFileName))
        instances.addThruPipe(new WikiPlainTextIterator(dataFolderName))

        val numTopics = args.numTopics
        // alpha = 1 / 100, 1 / 10
        // beta - dependend on vocabulary size
        // longer documents? not all documents
        val model = new ParallelTopicModel(numTopics, 1.0, 1.0 / args.numTopics)
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
