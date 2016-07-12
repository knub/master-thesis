package knub.master_thesis.probabilistic

import cc.mallet.util.Maths

object Divergence {

    def jensenShannonDivergence(p: Array[Double], q: Array[Double]): Double = {
        Maths.jensenShannonDivergence(p, q)
    }


    def kullbackLeibler(p: Array[Double], q: Array[Double]): Double = {
        var sum = 0.0
        for (i <- p.indices) {
            if (p(i) != 0.0 && q(i) != 0.0)
                sum += p(i) * Math.log(p(i) / q(i))
        }
        sum
    }

}
