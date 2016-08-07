import subprocess
import os


def run_process(command, cwd, grab_output=True):
    print " ".join(command)

    if grab_output:
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
        stdout, stderr = p.communicate()
        error = p.returncode
        if error:
            print "-" * 100
            print stdout
            print "-" * 100
            print stderr
            print "-" * 100
            raise Exception("Process failed")

        return stdout, stderr, error
    else:
        p = subprocess.Popen(command, cwd=cwd)
        p.wait()
        error = p.returncode
        if error:
            raise Exception("Process failed")

        return "", "", ""


def train_topic_model(model_name, alpha, beta):
    print "Training topic model"
    run_process(
        ["bin/topic-model",
         "topic-model-create",
         "--num-topics",
         "50",
         "--num-threads",
         "13",
         "--model-file-name",
         model_name,
         "--data-folder-name",
         "/home/stefan.bunk/master-thesis/data/20newsgroups/20news-bydate-train",
         "--num-iterations",
         "1500",
         "--alpha",
         str(alpha),
         "--beta",
         str(beta)],
        cwd="/home/stefan.bunk/master-thesis/code/scala",
        grab_output=False)


def preprocess_lflda(model_name):
    print "Preprocessing LFLDA"
    run_process(
        ["bin/lftm",
         "-model",
         "preprocess-LFLDA",
         "-topicmodel",
         model_name,
         "-vectors",
         "/data/wikipedia/2016-06-21/embedding-models/skip-gram.model",
         ],
        cwd="/home/stefan.bunk/LFTM",
        grab_output=False)


def train_lflda(model_name, alpha, beta, _lambda):
    print "Count documents"
    stdout, _, _ = run_process(
        ["grep",
         "##",
         model_name + ".skip-gram.model.restricted"], ".")
    num_documents = stdout.count("##")
    print str(num_documents) + " documents"
    print "Training LFLDA"
    run_process(
        ["bin/lftm",
         "-model",
         "LFLDA",
         "-topicmodel",
         model_name,
         "-vectors",
         "/data/wikipedia/2016-06-21/embedding-models/skip-gram.model",
         "-alpha",
         str(alpha),
         "-beta",
         str(beta),
         "-lambda",
         str(_lambda),
         "-sstep",
         "50",
         "-twords",
         "10",
         "-ntopics",
         "50",
         "-ndocs",
         str(num_documents),
         "-niters",
         "500"
         ],
        cwd="/home/stefan.bunk/LFTM",
        grab_output=False)


def run_palmetto_evaluation(model_name):
    stdout, _, _ = run_process(
        ["python",
         "run_palmetto.py",
         model_name + ".lflda-500.topics"],
        cwd="/home/stefan.bunk/master-thesis/code/python/knub/thesis/palmetto"
    )
    return stdout


def lftm_topic_coherence(alpha, beta, _lambda):
    model_folder = "/data/wikipedia/2016-06-21/topic-models/topic.20news.50-1500.alpha-%s.beta-%s" % (
        str(alpha).replace(".", "-"),
        str(beta).replace(".", "-"))
    try:
        os.mkdir(model_folder)
    except OSError:
        pass
    model_name = model_folder + "/model"

    train_topic_model(model_name, alpha, beta)
    preprocess_lflda(model_name)
    train_lflda(model_name, alpha, beta, _lambda)
    stdout = run_palmetto_evaluation(model_name)

    last_line = stdout.splitlines()[-1]
    score = float(last_line.split("\t")[1])
    return -score


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def main(job_id, params):
    print "params: ", params, " job_id: ", job_id
    alpha = params["alpha"][0]
    beta = params["beta"][0]
    _lambda = params["lambda"][0]

    existing_results = [
        (0.0001, 0.0001, 0.1, -0.423),
        (0.02505, 0.02505, 0.55, -0.438),
        (0.05, 0.05, 1.0, -0.427),
        (0.04915117, 0.01676279, 1.0, -0.428),
        (0.01941704, 0.0001, 0.5333867, -0.42)

    ]
    for tmp_alpha, tmp_beta, tmp_lambda, result in existing_results:
        print "Comparing " + str(alpha) + " with " + str(tmp_alpha)
        print "Comparing " + str(beta) + " with " + str(tmp_beta)
        print "Comparing " + str(_lambda) + " with " + str(tmp_lambda)
        if isclose(tmp_alpha, alpha) and isclose(tmp_beta, beta) and isclose(tmp_lambda, _lambda):
            print "Found result " + str(result)
            return result
        print "###"

    return lftm_topic_coherence(alpha, beta, _lambda)
