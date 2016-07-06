package knub.master_thesis.util

class Progress(totalNr: Long, granularity: Int = 0) {
    var c = 0L
    val outputEvery = (totalNr / 100 * Math.pow(10, granularity)).toLong

    var time = System.currentTimeMillis()
    def report_progress(): Unit = {
        if (c % outputEvery == 0) {
            val secondsSinceLast = Math.round((System.currentTimeMillis() - time) / 1000.0)
            time = System.currentTimeMillis()
            report(100.0 * c / totalNr, secondsSinceLast)
        }
        c += 1
    }

    private def report(percentage: Double, secondsSinceLast: Long): Unit = {
        val s = s"%.${Math.max(-granularity, 0)}f %% (%d secs)".format(percentage, secondsSinceLast)
        println(s)
    }
}
