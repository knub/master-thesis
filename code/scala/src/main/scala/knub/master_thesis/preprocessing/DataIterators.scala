package knub.master_thesis.preprocessing

import java.io.File
import java.nio.file.{Files, Path, Paths}
import java.util

import cc.mallet.types.Instance
import org.apache.commons.io.FileUtils

import scala.collection.JavaConverters._
import scala.io.Source
import scala.util.matching.Regex

object DataIterators {

    def getIteratorForDataFolderName(dataFolderName: String): (java.util.Iterator[Instance], String) = {
        if (dataFolderName.contains("plain-text")) {
            println("Detected Wikipedia")
            (wikipedia(dataFolderName), "../resources/stopwords.txt")
        } else if (dataFolderName.contains("nips")) {
            (nips(dataFolderName), "../resources/stopwords.txt")
        } else if (dataFolderName.contains("20news-bydate-train-with-classes/sentences.txt")) {
            println("Detected 20 news sentences corpus")
            (twentyNewsSentences(dataFolderName), "../resources/stopwords.20news.txt")
        } else if (dataFolderName.contains("20newsgroups")) {
            println("Detected 20 news corpus")
            (twentyNews(dataFolderName), "../resources/stopwords.20news.txt")
        } else {
            throw new Exception("No data iterator found for given data folder name.")
        }
    }

    def nips(dataFolderName: String): java.util.Iterator[Instance] = {
        val regex = new Regex("nips[0-9]{2}")
        val files = new File(dataFolderName).list().filter { f =>
            regex.findFirstIn(f).isDefined
        }.map { f =>
            new File(dataFolderName + "/" + f)
        }

        files.flatMap { folder =>
            folder.list().map { document =>
                val content = FileUtils.readFileToString(new File(s"$folder/$document"))
                new Instance(content, folder, document, null)
            }
        }.iterator.asJava
    }

    def wikipedia(dataFolderName: String) = {
        OnlyNormalPagesIterator.normalPagesIterator(new WikiPlainTextIterator(dataFolderName))
    }

    def twentyNews(dataFolderName: String) = {
        new TwentyNewsIterator(dataFolderName).iterator()
    }
    def twentyNewsSentences(dataFolderName: String): java.util.Iterator[Instance] = {
        Source.fromFile(dataFolderName).getLines().zipWithIndex.map { case (line, idx) =>
            try {
                val split = line.split('\t')
                val doc = split(0).toInt
                val clazz = split(1).toInt
                val text = split(2)
                new Instance(text, clazz, doc, null)
            } catch {
                case e: Exception =>
                    println(s"$idx>>>$line<<<")
                    throw e
            }
        }.asJava
    }
}

object OnlyNormalPagesIterator {
    def normalPagesIterator(wikiPlainTextIterator: WikiPlainTextIterator): java.util.Iterator[Instance] = {
        wikiPlainTextIterator.asScala.filter { inst =>
            val title = inst.getName.asInstanceOf[String]
            !(title.contains("Lists of") ||
                title.contains("List of") ||
                title.contains("isambiguation"))
        }.asJava
    }
}
class TwentyNewsIterator(dataFolderName: String) {

    def iterator(): java.util.Iterator[Instance] = {
        val articlesPath = Paths.get(dataFolderName + "/articles.txt").toFile
        val articles = scala.io.Source.fromFile(articlesPath, "UTF-8").getLines().toBuffer
//        val articlesShuffled = scala.util.Random.shuffle(articles)
        articles.map { text =>
            new Instance(text, null, null, null)
        }.iterator.asJava
    }

}
class WikiPlainTextIterator(dataFolderName: String) extends java.util.Iterator[Instance] {

    val p = Paths.get(dataFolderName)
    val pathIterator = java.nio.file.Files.walk(p).iterator().asScala.filter(Files.isRegularFile(_))
    var lineIterator: Iterator[String] = null
    var currentPath: Path = null
    val sb = new StringBuilder

    override def hasNext: Boolean = {
        pathIterator.hasNext || lineIterator.hasNext
    }

    override def next(): Instance = {
        if (lineIterator == null || !lineIterator.hasNext) {
            currentPath = pathIterator.next()
            lineIterator = scala.io.Source.fromFile(currentPath.toFile).getLines()
        }
        val firstLine = lineIterator.next
        assert(firstLine.startsWith("<doc"), s"$currentPath: '$firstLine' does not start with <doc")

        sb.clear()
        var currentLine = ""
        while (!currentLine.startsWith("</doc>")) {
            sb.append(currentLine + " ")
            currentLine = lineIterator.next
        }
        new Instance(sb.toString(), null, firstLine, null)
    }
}
