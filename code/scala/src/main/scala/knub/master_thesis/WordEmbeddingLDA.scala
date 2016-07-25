package knub.master_thesis

import com.carrotsearch.hppc.IntArrayList
import knub.master_thesis.util.FreeMemory

import scala.collection.mutable

class WordEmbeddingLDA {

}

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
