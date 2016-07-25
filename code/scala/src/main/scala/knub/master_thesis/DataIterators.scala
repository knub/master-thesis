package knub.master_thesis

import java.nio.file.{Files, Path, Paths}

import cc.mallet.types.Instance

import scala.collection.JavaConverters._

object DataIterators {
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

    val p = Paths.get(dataFolderName)
    val pathIterator = java.nio.file.Files.walk(p).iterator().asScala.filter(Files.isRegularFile(_))

    val documentBodies = pathIterator.map {  p =>
        val source = scala.io.Source.fromFile(p.toFile, "ISO-8859-1")
        val fileContent = source.getLines()
        val body  = fileContent.dropWhile { l => l.nonEmpty }

        val pathCount = p.getNameCount
        val instance = new Instance(body.mkString("\n"), null, p.getName(pathCount - 1).toString, null)
        source.close()
        instance
    }.toBuffer

    val shuffled = scala.util.Random.shuffle(documentBodies)

    def iterator(): java.util.Iterator[Instance] = {
        shuffled.iterator.asJava
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
