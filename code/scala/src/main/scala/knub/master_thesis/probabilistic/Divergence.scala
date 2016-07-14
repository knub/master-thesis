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
        var max = 0.0
        var i = 0
        while (i < p.length) {
            val diff = Math.abs(p(i) - q(i))
            if (diff > max)
                max = diff
            i += 1
        }
        max
    }
    /**
      * The smaller, the more similar. Zero means exact
      */
    def sumDistance(p: Array[Double], q: Array[Double]): Double = {
        var sum = 0.0
        var i = 0
        while (i < p.length) {
            sum += Math.abs(p(i) - q(i))
            i += 1
        }
        0.5 * sum
    }

    /**
      * The smaller, the more similar. Zero means exact
      */
    def bhattacharyyaDistance(p: Array[Double], q: Array[Double]): Double = {
        var sum = 0.0
        var i = 0
        while (i < p.length) {
            sum += Math.sqrt(p(i) * q(i))
            i += 1
        }
        - Math.log(sum)
    }

    val sqrt2Rez = 1.0 / Math.sqrt(2)
    /**
      * The smaller, the more similar. Zero means exact
      */
    def hellingerDistance(p: Array[Double], q: Array[Double]): Double = {
        var sum = 0.0
        var i = 0
        while (i < p.length) {
            val sqrtDiff = Math.sqrt(p(i)) - Math.sqrt(q(i))
            sum += sqrtDiff * sqrtDiff
            i += 1
        }
        sqrt2Rez * Math.sqrt(sum)
    }

}
