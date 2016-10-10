package knub.master_thesis.welda

import java.util

import cc.mallet.cluster.KMeans
import cc.mallet.types.{Metric, SparseVector}
import knub.master_thesis.Args
import weka.clusterers.SimpleKMeans
import weka.core._

class GaussianKMeansWELDA(p: Args) extends GaussianWELDA(p) {

    override def getFolderName(): String = {
        s"${p.modelFileName}.$embeddingName.welda.gaussian-kmeans." +
            s"distance-$DISTANCE_FUNCTION." +
            s"lambda-${LAMBDA.toString.replace('.', '-')}"
    }

    override def getTopTopicVectors(): Array[Seq[Array[Double]]] = {
//        val kmeans = new KMeans(null, 2)
//        weka.clusterers.XMeans
        val topTopicVectors = super.getTopTopicVectors()
        topTopicVectors.map { topicVectors =>
            val attributes = new util.ArrayList[Attribute]()
            for (i <- 0 until p.pcaDimensions) {
                attributes.add(new Attribute(i.toString))
            }
            val instances = new Instances("vectors", attributes, p.distributionEstimationSamples)
            topicVectors.foreach { vector =>
                val inst = new DenseInstance(1.0, vector)
                instances.add(inst)
            }
            val NUM_CLUSTERS = 2
            val kmeans = new SimpleKMeans()
            kmeans.setNumClusters(NUM_CLUSTERS)
            kmeans.setInitializationMethod(new SelectedTag(SimpleKMeans.KMEANS_PLUS_PLUS, SimpleKMeans.TAGS_SELECTION))
            kmeans.setPreserveInstancesOrder(true)
            kmeans.buildClusterer(instances)

            val counts = new Array[Int](NUM_CLUSTERS)
            val assignments = kmeans.getAssignments
            assignments.foreach { a => counts(a) += 1 }
            println(s"${counts(0)}-${counts(1)}")
            val maxIndex = counts.zipWithIndex.maxBy(_._1)._2
            val maxValue = counts(maxIndex)
            if (maxValue >= 15)
                topicVectors.zip(assignments).filter { case (_, a) => a == maxIndex }.map(_._1)
            else
                topicVectors
        }
    }
}
