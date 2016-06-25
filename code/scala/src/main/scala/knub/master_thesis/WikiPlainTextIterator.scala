package knub.master_thesis

import java.nio.file.{Files, Paths}

import cc.mallet.types.Instance
import scala.collection.JavaConverters._

class WikiPlainTextIterator(dataFolderName: String) extends java.util.Iterator[Instance] {

    val p = Paths.get(dataFolderName)
    val fileIterator = java.nio.file.Files.walk(p).iterator().asScala.filter(Files.isRegularFile(_))
    var lineIterator: Iterator[String] = null
    val sb = new StringBuilder

    override def hasNext: Boolean = {
        fileIterator.hasNext || lineIterator.hasNext
    }

    override def next(): Instance = {
        if (lineIterator == null || !lineIterator.hasNext)
            lineIterator = scala.io.Source.fromFile(fileIterator.next.toFile).getLines()
        val firstLine = lineIterator.next
        assert(firstLine.startsWith("<doc"))

        sb.clear()
        var currentLine = ""
        while (!currentLine.startsWith("</doc")) {
            currentLine = lineIterator.next
            sb.append(currentLine + " ")
        }
        new Instance(sb.toString(), null, firstLine, null)
    }
}
