package knub.master_thesis.probabilistic

import cc.mallet.util.Maths

object Divergence {

    def jensenShannonDivergence(p: Array[Double], q: Array[Double]): Double = {
        val m = new Array[Double](p.length)
        for (k <- p.indices)
            m(k) = 0.5 * (p(k) + q(k))

//        val divergence = kullbackLeibler(p, m) + kullbackLeibler(q, m)
        val divergence = Maths.jensenShannonDivergence(p, q)
        divergence
//        0.5 * divergence
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
