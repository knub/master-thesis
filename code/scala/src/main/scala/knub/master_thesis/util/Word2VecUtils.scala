package knub.master_thesis.util

import org.apache.commons.lang3.text.WordUtils
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors

object Word2VecUtils {
    def findActualWord(word2Vec: WordVectors, word: String): String = {
        findActualWordOption(word2Vec, word) match {
            case Some(w) =>
                w
            case None =>
                throw new RuntimeException(word)
        }
    }

    def findActualWordOption(word2Vec: WordVectors, word: String): Option[String] = {
        if (word2Vec.hasWord(word))
            Some(word)
        else if (word2Vec.hasWord(WordUtils.capitalize(word)))
            Some(WordUtils.capitalize(word))
        else if (word2Vec.hasWord(word.toUpperCase))
            Some(word.toUpperCase())
        else
            None
    }

    def findActualVector(word2Vec: WordVectors, word: String): Option[Array[Double]] = {
        findActualWordOption(word2Vec, word).map(word2Vec.getWordVector)
    }

}
