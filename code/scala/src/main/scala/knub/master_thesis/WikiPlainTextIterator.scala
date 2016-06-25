package knub.master_thesis

import java.nio.file.{Files, Path, Paths}

import cc.mallet.types.Instance

import scala.collection.JavaConverters._

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
