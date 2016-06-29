package knub.master_thesis

import java.io.File
import java.util.regex.Pattern

import cc.mallet.pipe.{Pipe, SerialPipes}
import cc.mallet.pipe._

import scala.collection.mutable
import scala.collection.JavaConverters._

object PreprocessingPipe {
    def pipeList(stopWordsFileName: String): mutable.ArrayBuffer[Pipe] = {

        mutable.ArrayBuffer[Pipe](
            new CharSequenceLowercase(),
            // Regex explanation: lowercase [lowercase punctuation]+ lowercase  --- at least three characters,
            // no punctuation at the end
            new CharSequence2TokenSequence(Pattern.compile("\\p{L}[\\p{L}\\p{P}]+\\p{L}")),
            new UseOnlyFirstNWordsOfDocument(1000),
            new TokenSequenceRemoveStopwords(
                new File(stopWordsFileName), "UTF-8", false, false, false),
            new TokenSequence2FeatureSequence()
        )
    }
    def pipe(stopWordsFileName: String): Pipe = {
        new SerialPipes(pipeList(stopWordsFileName).asJava)

    }
}
