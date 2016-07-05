import subprocess

def concept_categorization(num_topics, num_iterations):
    p = subprocess.Popen(
        ["java",
         "-jar",
         "/home/knub/Repositories/master-thesis/code/scala/out/artifacts/master_thesis_jar/master-thesis.jar",
         "topic-model",
         "--create-new-model",
         "--num-topics",
         str(num_topics),
         "--num-iterations",
         str(num_iterations),
         "--stop-words",
         "/home/knub/Repositories/master-thesis/code/resources/stopwords.txt",
         "--concept-categorization",
         "/home/knub/Repositories/master-thesis/data/concept-categorization/battig_concept-categorization.tsv"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = p.communicate()
    error = p.returncode
    if error:
        return RuntimeError("Could not compute topics")
    else:
        try:
            res = stdout.splitlines()[-1]
            return -float(res) / 100
        except:
            print stdout
            print stderr


def main(job_id, params):
    print "params: ", params, " job_id: ", job_id
    return concept_categorization(params["topics"][0], params["iterations"][0])
