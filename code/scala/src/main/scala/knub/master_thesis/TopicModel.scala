package knub.master_thesis

import java.util

import cc.mallet.topics.ParallelTopicModel
import cc.mallet.types.{Instance, InstanceList}

class TopicModel(args: Args, alpha: Double, beta: Double, instancesIterator: java.util.Iterator[Instance]) {


    def run(stopWordsFileName: String): TopicModelResult = {
        val instances = new InstanceList(PreprocessingPipe.pipe(stopWordsFileName))
        instances.addThruPipe(instancesIterator)

        val numTopics = args.numTopics
        val model = new ParallelTopicModel(numTopics, numTopics * alpha, beta)
        model.alpha = calculateAlphaWithBackgroundTopic(numTopics, alpha)
        model.setOptimizeInterval(10)
        model.setBurninPeriod(100)
        model.addInstances(instances)
        model.setNumThreads(args.numThreads)
        model.setNumIterations(args.numIterations)
        model.showTopicsInterval = 0
        model.estimate()

        new TopicModelResult(model)
    }

    def calculateAlphaWithBackgroundTopic(numTopics: Int, alpha: Double): Array[Double] = {
        val a = new Array[Double](numTopics)
        java.util.Arrays.fill(a, alpha)
        a(0) = alpha * 5
        a
    }
}
