package knub.master_thesis.math

object Bessel {

    val ACC = 40.0
    val BIGNO = 1.0e10
    val BIGNI = 1.0e-10

    def bessi0(x: Double): Double = {
        val ax = Math.abs(x)
        var ans: Double = 0.0
        if (ax < 3.75) {
            var y = x / 3.75
            y = y * y
            ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
                +y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))))
        } else {
            val y=3.75 / ax
            ans=(Math.exp(ax)/Math.sqrt(ax))*(0.39894228+y*(0.1328592e-1
                +y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
                +y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
                +y*0.392377e-2))))))))
        }
        ans
    }

    def bessi1(x: Double): Double = {
        val ax = Math.abs(x)
        var ans: Double = 0.0
        if (ax < 3.75) {
            var y = x/3.75
            y = y * y
            ans = ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934
                +y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3))))))
        } else {
            val y = 3.75 / ax
            ans=0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1
                -y*0.420059e-2))
            ans=0.39894228+y*(-0.3988024e-1+y*(-0.362018e-2
                +y*(0.163801e-2+y*(-0.1031555e-1+y*ans))))
            ans *= (Math.exp(ax)/Math.sqrt(ax))
        }
        if (x < 0.0) -ans else ans
    }

    def bessi(n: Int, x: Double): Double = {
        if (n == 0) {
            bessi0(x)
        } else if (n == 1) {
            bessi1(x)
        } else {
            if (x == 0.0) {
                0.0
            } else {
                val tox = 2.0 / Math.abs(x)
                var bip = 0.0
                var ans = 0.0
                var bi = 1.0
                var j = 2 * (n + Math.sqrt(ACC * n).toInt)
                while (j > 0) {
                    val bim = bip + j * tox * bi
                    bip = bi
                    bi = bim

                    if (Math.abs(bi) > BIGNO) {
                        ans *= BIGNI
                        bi *= BIGNI
                        bip *= BIGNI
                    }
                    if (j == n) {
                        ans = bip
                    }
                    j -= 1
                }
                ans *= bessi0(x) / bi
                if (x < 0.0 && n % 2 == 1)
                    -ans
                else
                    ans
            }
        }
    }
}

