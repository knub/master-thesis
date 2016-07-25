package knub.master_thesis

import java.nio.file.{Files, Path, Paths}

import cc.mallet.types.Instance

import scala.collection.JavaConverters._

object DataIterators {
    def getIteratorForDataFolderName(dataFolderName: String): java.util.Iterator[Instance] = {
        if (dataFolderName.contains("plain-text")) {
            wikipedia(dataFolderName)
        } else if (dataFolderName.contains("20newsgroups")) {
            twentyNews(dataFolderName)
        } else {
            throw new Exception("No data iterator found for given data folder name.")
        }
    }

    def wikipedia(dataFolderName: String) = {
        OnlyNormalPagesIterator.normalPagesIterator(new WikiPlainTextIterator(dataFolderName))
    }

    def twentyNews(dataFolderName: String) = {
        new TwentyNewsIterator(dataFolderName).iterator()
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
        val articles = scala.io.Source.fromFile(articlesPath, "ISO-8859-1").getLines().toBuffer
        val articlesShuffled = scala.util.Random.shuffle(articles)
        articlesShuffled.map { text =>
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
