package knub.master_thesis.preprocessing

import cc.mallet.pipe.Pipe
import cc.mallet.types.{Instance, TokenSequence}

class UseOnlyFirstNWordsOfDocument(n: Int) extends Pipe {
    override def pipe(carrier: Instance): Instance = {
        val tokens = carrier.getData.asInstanceOf[TokenSequence]
        carrier.setData(new TokenSequence(tokens.subList(0, Math.min(tokens.size, n))))
        carrier
    }
}
