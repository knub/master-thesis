package knub.master_thesis.util

import java.text.SimpleDateFormat
import java.util.Calendar

object Date {

    def date(): String = {
        val cal = Calendar.getInstance()
        val sdf = new SimpleDateFormat("HH:mm:ss")
        val date = sdf.format(cal.getTime)
        date
    }

}
