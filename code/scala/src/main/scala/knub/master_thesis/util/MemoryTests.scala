package knub.master_thesis.util

import com.carrotsearch.hppc.IntArrayList

/**
  * Created by knub on 26/07/16.
  */
private object MemoryTests {

    val N = 100000000
    println(s"${FreeMemory.get(false, 0)} RAM")

    // Simple array
    //    val a = new Array[Int](N)

    // ArrayBuffer
    //    val ab = new mutable.ArrayBuffer[Int](N)
    //    for (i <- 0 until N) {
    //        if (i % 1000000 == 0)
    //            println(i)
    //        ab.append(i)
    //    }

    // HPPC
    val a = new IntArrayList(N)
    for (i <- 0 until N) {
        if (i % 1000000 == 0)
            println(i)
        a.add(i)
    }
    println(s"${FreeMemory.get(true, 15)} RAM")


    Thread.sleep(10000)
}
