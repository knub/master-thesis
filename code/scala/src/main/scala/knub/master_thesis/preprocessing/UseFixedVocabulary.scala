package knub.master_thesis.preprocessing

import java.util
import java.util.HashSet;
import java.util.ArrayList;
import java.io._;

import cc.mallet.types.FeatureSequenceWithBigrams;
import cc.mallet.types.Instance;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;

import cc.mallet.pipe.Pipe

object UseFixedVocabulary {

    private val CURRENT_SERIAL_VERSION = 1
}

@SerialVersionUID(1)
class UseFixedVocabulary(var vocabulary: Set[String]) extends Pipe with Serializable {
    import UseFixedVocabulary._

    override def pipe(carrier: Instance): Instance = {
        val ts = carrier.getData.asInstanceOf[TokenSequence]
        val ret = new TokenSequence()
        var prevToken: Token = null
        for (i <- 0 until ts.size) {
            val t = ts.get(i)
            if (vocabulary.contains(t.getText)) {
                ret.add(t)
                prevToken = t
            }
        }
        carrier.setData(ret)
        carrier
    }

    private def writeObject(out: ObjectOutputStream) {
        out.writeInt(CURRENT_SERIAL_VERSION)
        out.writeObject(vocabulary)
    }

    private def readObject(in: ObjectInputStream) {
        val version = in.readInt()
        vocabulary = in.readObject().asInstanceOf[Set[String]]
    }
}

