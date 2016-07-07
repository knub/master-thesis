package knub.master_thesis

import java.io.File
import java.util.regex.Pattern

import cc.mallet.pipe.{Pipe, SerialPipes}
import cc.mallet.pipe._
import knub.master_thesis.preprocessing.{UseFixedVocabulary, UseOnlyFirstNWordsOfDocument}

import scala.collection.mutable
import scala.collection.JavaConverters._
import scala.io.Source

object PreprocessingPipe {
    def pipeList(stopWordsFileName: String): mutable.ArrayBuffer[Pipe] = {
        val vocabulary = Source.fromFile("/san2/data/wikipedia/2016-06-21/vocab.txt").getLines().toSet

        mutable.ArrayBuffer[Pipe](
            new CharSequenceLowercase(),
            // Regex explanation: lowercase [lowercase punctuation]+ lowercase  --- at least three characters,
            // no punctuation at the end
            new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")),
            new UseOnlyFirstNWordsOfDocument(2000),
            new TokenSequenceRemoveStopwords(
                new File(stopWordsFileName), "UTF-8", false, false, false),
//            new UseFixedVocabulary(vocabulary),
            new TokenSequence2FeatureSequence()
        )
    }
    def pipe(stopWordsFileName: String): Pipe = {
        new SerialPipes(pipeList(stopWordsFileName).asJava)

    }
}
