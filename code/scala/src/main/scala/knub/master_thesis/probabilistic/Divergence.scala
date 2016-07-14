package knub.master_thesis.probabilistic

import cc.mallet.util.Maths

object Divergence {

    /**
      * The smaller, the more similar. Zero means exact
      */
    def jensenShannonDivergence(p: Array[Double], q: Array[Double]): Double = {
        Maths.jensenShannonDivergence(p, q)
    }

    /**
      * The smaller, the more similar. Zero means exact
      */
    def maxDistance(p: Array[Double], q: Array[Double]): Double = {
        p.indices.map { i => Math.abs(p(i) - q(i)) }.max
    }
    /**
      * The smaller, the more similar. Zero means exact
      */
    def sumDistance(p: Array[Double], q: Array[Double]): Double = {
        0.5 * p.indices.map { i => Math.abs(p(i) - q(i)) }.sum
    }

    /**
      * The smaller, the more similar. Zero means exact
      */
    def bhattacharyyaDistance(p: Array[Double], q: Array[Double]): Double = {
        val BC = p.indices.map { i => Math.sqrt(p(i) * q(i)) }.sum
        - Math.log(BC)
    }

    val sqrt2Rez = 1.0 / Math.sqrt(2)
    /**
      * The smaller, the more similar. Zero means exact
      */
    def hellingerDistance(p: Array[Double], q: Array[Double]): Double = {
        val innerSum = p.indices.map { i =>
            val sqrtDiff = Math.sqrt(p(i)) - Math.sqrt(q(i))
            sqrtDiff * sqrtDiff
        }.sum
        sqrt2Rez * Math.sqrt(innerSum)
    }


//    def kullbackLeibler(p: Array[Double], q: Array[Double]): Double = {
//        var sum = 0.0
//        for (i <- p.indices) {
//            if (p(i) != 0.0 && q(i) != 0.0)
//                sum += p(i) * Math.log(p(i) / q(i))
//        }
//        sum
//    }

}
