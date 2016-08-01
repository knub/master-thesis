package knub.master_thesis.util

import org.apache.commons.math3.random.MersenneTwister

object Sampler {
    private val rand: MersenneTwister = new MersenneTwister()

    def nextDiscrete(probs: Array[Double]): Int = {
        var sum = 0.0
        var i = 0
        while (i < probs.length) {
            sum += probs(i)
            i += 1
        }
        val r = rand.nextDouble() * sum
        sum = 0.0
        i = 0
        while (i < probs.length) {
            sum += probs(i)
            if (sum > r)
                return i
            i += 1
        }
        probs.length - 1
    }

    def nextCoinFlip(successProb: Double): Boolean = {
        rand.nextDouble() < successProb
    }
}

