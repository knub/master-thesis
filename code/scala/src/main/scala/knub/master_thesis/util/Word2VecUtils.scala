package knub.master_thesis.util

import org.apache.commons.lang3.text.WordUtils
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors

object Word2VecUtils {
    def findActualWord(word2Vec: WordVectors, word: String): String = {
        val actualWord =
            if (word2Vec.hasWord(word))
                word
            else if (word2Vec.hasWord(WordUtils.capitalize(word)))
                WordUtils.capitalize(word)
            else if (word2Vec.hasWord(word.toUpperCase))
                word.toUpperCase()
            else
                throw new RuntimeException(word)
        actualWord
    }

}
