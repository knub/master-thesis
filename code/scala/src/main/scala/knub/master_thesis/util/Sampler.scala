package knub.master_thesis.util

import org.apache.commons.math3.random.MersenneTwister

object Sampler {

    def nextDiscrete(probs: Array[Double]): Int = {
        var sum = 0.0
        var i = 0
        while (i < probs.length) {
            sum += probs(i)
            i += 1
        }
        val r = MTRandom.nextDouble() * sum
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
}

object MTRandom {

    private var rand: MersenneTwister = new MersenneTwister()

    def setSeed(seed: Long) {
        rand.setSeed(seed)
    }

    def nextDouble(): Double = rand.nextDouble()

    def nextInt(n: Int): Int = rand.nextInt(n)

    def nextBoolean(): Boolean = rand.nextBoolean()
}

