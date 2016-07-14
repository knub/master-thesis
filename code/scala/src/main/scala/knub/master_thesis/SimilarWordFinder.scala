package knub.master_thesis

import java.util.Comparator

import com.google.common.collect.MinMaxPriorityQueue
import knub.master_thesis.probabilistic.Divergence

import scala.collection.JavaConversions._
import scala.collection.mutable

case class WordPair(word1: String, word2: String, divergence: Double)
class WordPairComparator(ascending: Int = 1) extends Comparator[WordPair] {
    override def compare(wp1: WordPair, wp2: WordPair): Int = {
        if (wp1.divergence == wp2.divergence)
            0
        else if (wp1.divergence < wp2.divergence)
            -1 * ascending
        else
            +1 * ascending
    }
}
case class SimFunction(name: String, sim: (Array[Double], Array[Double]) => Double)

class SimilarWordFinder(res: TopicModelResult, args: Args, frequentWordsRaw: Array[String], topicProbs: Array[Array[Double]]) {
    val topWordsRaw = res.getTopWords(10)

    val topWordsAlphabet = mutable.Map[String, Int]()
    topWordsRaw.foreach { word => topWordsAlphabet += word -> res.dataAlphabet.lookupIndex(word, false) }
    val frequentWordsAlphabet = mutable.Map[String, Int]()
    frequentWordsRaw.foreach { word => frequentWordsAlphabet += word -> res.dataAlphabet.lookupIndex(word, false) }

    val topWords = topWordsRaw.filter { word => topWordsAlphabet(word) >= 0}
    val frequentWords = frequentWordsRaw.filter { word => frequentWordsAlphabet(word) >= 0}.take(150000)

    val topWordsCount = topWords.length
    val frequentWordsCount = frequentWords.length
    println(s"Top-words: $topWordsCount")
    println(s"Frequent-words: $frequentWordsCount")

    def run(): Unit = {
        List(
            SimFunction("max", Divergence.maxDistance),
            SimFunction("sum", Divergence.sumDistance),
            SimFunction("bhattacharyya", Divergence.bhattacharyyaDistance),
            SimFunction("hellinger", Divergence.hellingerDistance),
            SimFunction("jensen-shannon", Divergence.jensenShannonDivergence)
        ).foreach { sim =>
            println(s"Find word pairs ${sim.name}")
            findMostSimilarWordPairs(sim)
        }
    }

    def findMostSimilarWordPairs(sim: SimFunction): Unit = {
        val progress = new util.Progress(topWordsCount.toLong * frequentWordsCount.toLong, -1)
        val pw = args.getPrintWriterFor(s".similars-${sim.name}")

        for (i <- 0 until topWordsCount) {
            val wordI = topWords(i)
            val idxI = topWordsAlphabet(wordI)
            val probsI = topicProbs(idxI)
            val topQueue = MinMaxPriorityQueue.orderedBy(new WordPairComparator)
                .maximumSize(10)
                .create[WordPair]()

            for (j <- 0 until frequentWordsCount) {
                progress.report_progress()

                val wordJ = frequentWords(j)
                if (wordI != wordJ) {
                    val idxJ = frequentWordsAlphabet(wordJ)
                    val probsJ = topicProbs(idxJ)
                    val divergence: Double = sim.sim(probsI, probsJ)
                    topQueue.add(WordPair(wordI, wordJ, divergence))
                }
            }
            topQueue.toList.sortBy(_.divergence).foreach { wordPair =>
                val word1 = wordPair.word1
                val word2 = wordPair.word2
                pw.println(f"${wordPair.divergence}%.9f\tSIM\t$word1\t$word2")
            }
        }
        pw.close()
    }
}
