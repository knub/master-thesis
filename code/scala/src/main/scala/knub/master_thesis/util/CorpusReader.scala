package knub.master_thesis.util

import java.io.{BufferedReader, FileReader}
import java.util

import com.carrotsearch.hppc.IntArrayList

case class Corpus(documents: java.util.List[IntArrayList], topics: java.util.List[IntArrayList])

object CorpusReader {

    def readCorpus(pathToCorpus: String): Corpus = {
        val corpus: java.util.List[IntArrayList] = new util.ArrayList[IntArrayList]()
        val topicAssignments: java.util.List[IntArrayList] = new util.ArrayList[IntArrayList]()

        var docNr = 0
        var lineNr = 0

        var document = new IntArrayList()
        var topics = new IntArrayList()

        val brDocument = new BufferedReader(new FileReader(pathToCorpus))
        var line: String = brDocument.readLine()
        while (line != null) {
            if (line == "##") {
                if (document.size > 0) {
                    corpus.add(document)
                    topicAssignments.add(topics)
                    document = new IntArrayList()
                    topics = new IntArrayList()
                    docNr += 1
                    if (docNr % 100000 == 0) {
                        println(docNr)
                    }
                } else {
                    println("Empty document at line " + lineNr)
                }
            } else {
                try {
                    val wordId = java.lang.Integer.parseInt(line.substring(0, 6))
                    val topicId = java.lang.Integer.parseInt(line.substring(7, 13))
                    document.add(wordId)
                    topics.add(topicId)
                } catch {
                    case e: Exception =>
                        println(line)
                        println("lineNr = " + lineNr)
                        println("line = <" + line + ">")
                        throw e
                }
                lineNr += 1
            }
            line = brDocument.readLine()
        }
        println("Finished reading corpus with " + corpus.size + " documents.")
        Corpus(corpus, topicAssignments)
    }
}
