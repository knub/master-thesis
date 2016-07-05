package knub.master_thesis.preprocessing

import cc.mallet.pipe.Pipe
import cc.mallet.types.{Instance, TokenSequence}

/**
  * Created by knub on 24/06/16.
  */
class UseOnlyFirstNWordsOfDocument(n: Int) extends Pipe {
    override def pipe(carrier: Instance): Instance = {
        val tokens = carrier.getData.asInstanceOf[TokenSequence]
        carrier.setData(new TokenSequence(tokens.subList(0, Math.min(tokens.size, n))))
        carrier
    }
}
