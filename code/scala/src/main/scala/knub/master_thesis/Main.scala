package knub.master_thesis

import java.io.{File, FileInputStream, InputStreamReader}
import java.util.{Formatter, Locale}
import java.util.regex.Pattern

import cc.mallet.pipe._
import cc.mallet.pipe.iterator.CsvIterator
import cc.mallet.topics.ParallelTopicModel
import cc.mallet.types.{FeatureSequence, Instance, InstanceList, TokenSequence}

import scala.collection.mutable
import scala.collection.JavaConverters._

object Main {
    def main(args: Array[String]): Unit = {

        val pipeList = mutable.ArrayBuffer[Pipe]()
        pipeList += new CharSequenceLowercase()
        // Regex explanation: lowercase [lowercase punctuation]+ lowercase  --- at least three characters,
        // no punctuation at the end
        pipeList += new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}"))
        pipeList += new UseOnlyFirstNWordsOfDocument(1000)
        pipeList += new TokenSequenceRemoveStopwords(
            new File("/home/knub/Repositories/master-thesis/code/scala/resources/stopwords.txt"), "UTF-8", false, false, false)
        pipeList += new TokenSequence2FeatureSequence()

        val instances = new InstanceList(new SerialPipes(pipeList.asJava))

        val fileReader = new InputStreamReader(new FileInputStream(new File(args(0))), "UTF-8")

        instances.addThruPipe(new CsvIterator(fileReader, Pattern.compile("^(\\S*)[\\s,]*(\\S*)[\\s,]*(.*)$"),
            3, 2, 1)); // data, label, name fields

        val numTopics = 100
        val model = new ParallelTopicModel(numTopics, 1.0, 0.01)

        model.addInstances(instances)

        model.setNumThreads(2)
        model.setNumIterations(50)
        println("Starting inference")
        model.estimate()
        println("Finished inference")

        // Show the words and topics in the first instance

        // The data alphabet maps word IDs to strings
        val dataAlphabet = instances.getDataAlphabet

        val tokens = model.getData.get(0).instance.getData.asInstanceOf[FeatureSequence]
        val topics = model.getData.get(0).topicSequence

        for (i <- 0 until tokens.getLength) {
            println(s"${dataAlphabet.lookupObject(tokens.getIndexAtPosition(i))}-${topics.getIndexAtPosition(i)}")
        }

        // Estimate the topic distribution of the first instance,
        //  given the current Gibbs state.
        val topicDistribution = model.getTopicProbabilities(0)

        // Get an array of sorted sets of word ID/count pairs
        val topicSortedWords = model.getSortedWords

        // Show top 5 words in topics with proportions for the first document
        for (topic <- 0 until numTopics) {
            val iterator = topicSortedWords.get(topic).iterator()

            println(f"$topic\t${topicDistribution(topic)}%.3f\t")
            var rank = 0
            while (iterator.hasNext && rank < 5) {
                val idCountPair = iterator.next()
                println(s"${dataAlphabet.lookupObject(idCountPair.getID)} (${idCountPair.getWeight}%.0f) ")
                rank += 1
            }
        }

        // Create a new instance with high probability of topic 0
        val topicZeroText = new java.lang.StringBuilder()
        val iterator = topicSortedWords.get(0).iterator()

        var rank = 0
        while (iterator.hasNext && rank < 5) {
            val idCountPair = iterator.next()
            topicZeroText.append(dataAlphabet.lookupObject(idCountPair.getID) + " ")
            rank += 1
        }

        // Create a new instance named "test instance" with empty target and source fields.
        val testing = new InstanceList(instances.getPipe)
        testing.addThruPipe(new Instance(topicZeroText.toString, null, "test instance", null))

        val inferencer = model.getInferencer
        val testProbabilities = inferencer.getSampledDistribution(testing.get(0), 10, 1, 5)
        System.out.println("0\t" + testProbabilities(0))
    }

}
