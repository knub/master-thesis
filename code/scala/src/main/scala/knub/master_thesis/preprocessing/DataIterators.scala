package knub.master_thesis.preprocessing

import java.io.File
import java.nio.file.{Files, Path, Paths}
import java.util

import cc.mallet.types.Instance
import knub.master_thesis.util.CorpusReader
import org.apache.commons.io.FileUtils

import scala.collection.JavaConverters._
import scala.io.Source
import scala.util.matching.Regex

object DataIterators {


    def getIteratorForDataFolderName(dataFolderName: String): (java.util.Iterator[Instance], String) = {
        if (dataFolderName.contains("plain-text")) {
            println("Detected Wikipedia")
            (wikipedia(dataFolderName), "../resources/stopwords.txt")
        } else if (dataFolderName.contains(".restricted")) {
            println("Detected restricted corpus")
            (restrictedCorpus(dataFolderName), "../resources/stopwords.txt")
        } else if (dataFolderName.contains("nips")) {
            (nips(dataFolderName), "../resources/stopwords.txt")
        } else if (dataFolderName.contains("20news-bydate-train-with-classes/sentences.txt")) {
            println("Detected 20 news sentences corpus")
            (twentyNewsSentences(dataFolderName), "../resources/stopwords.20news.txt")
        } else if (dataFolderName.contains("20newsgroups")) {
            println("Detected 20 news corpus")
            (twentyNews(dataFolderName), "../resources/stopwords.txt")
        } else {
            throw new Exception("No data iterator found for given data folder name.")
        }
    }

    def restrictedCorpus(dataFolderName: String): java.util.Iterator[cc.mallet.types.Instance] = {
        if (new File(dataFolderName + ".classes").exists()) {
            val vocabulary = Source.fromFile(dataFolderName + ".vocab").getLines.toArray
            val classes = Source.fromFile(dataFolderName + ".classes").getLines.toList
            val corpus = CorpusReader.readCorpus(dataFolderName)

            val documents = corpus.documents.asScala.map { doc =>
                val sb = new StringBuilder
                doc.iterator().asScala.foreach { i =>
                    sb.append(s" ${vocabulary(i.value)}")
                }
                sb.toString()
            }
            assert(documents.length == classes.length)
            documents.zip(classes).map { case (document, clazz) =>
                new Instance(document, clazz.toInt, null, null)
            }.iterator.asJava
        } else {
            val vocabulary = Source.fromFile(dataFolderName + ".vocab").getLines.toArray
            val corpus = CorpusReader.readCorpus(dataFolderName)

            val documents = corpus.documents.asScala.map { doc =>
                val sb = new StringBuilder
                doc.iterator().asScala.foreach { i =>
                    sb.append(s" ${vocabulary(i.value)}")
                }
                sb.toString()
            }
            documents.map { case document =>
                new Instance(document, null, null, null)
            }.iterator.asJava

        }
    }

    def nips(dataFolderName: String): java.util.Iterator[Instance] = {
        val regex = new Regex("nips[0-9]{2}")
        val files = new File(dataFolderName).list().filter { f =>
            regex.findFirstIn(f).isDefined
        }.map { f =>
            new File(dataFolderName + "/" + f)
        }

        val instances = files.flatMap { folder =>
            folder.list().map { document =>
                val origLines = FileUtils.readLines(new File(s"$folder/$document")).asScala.toList
                val lines = origLines.dropWhile(_.length < 5)

                val title = lines.head
                val content = lines.dropWhile(!_.toLowerCase.contains("abstract")).drop(1)

                val text = title + content.mkString("\n")
//                val content = FileUtils.readFileToString(new File())
                new Instance(text, folder.getName, folder + "/" + document, text)
            }
        }
        scala.util.Random.shuffle(instances.toList).iterator.asJava
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
        val articlesClassesPath = Paths.get(dataFolderName + "/articles.class.txt").toFile

        val articles = scala.io.Source.fromFile(articlesPath, "UTF-8").getLines().toBuffer
        val classes = scala.io.Source.fromFile(articlesClassesPath, "UTF-8").getLines().toBuffer
        assert(articles.length == classes.length)
        var i = 0
        articles.zip(classes).map { case (articlesLine, clazz) =>
            val split = articlesLine.split('\t')
            assert(split.length == 2, s"split length should be 2 but is ${split.length} at line $i")
            i += 1
            val fileName = split(0)
            val text = split(1)
            new Instance(text, clazz.toInt, fileName, articlesLine)
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
