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
         "11314",
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


def main(job_id, params):
    print "params: ", params, " job_id: ", job_id
    alpha = params["alpha"][0]
    beta = params["beta"][0]
    _lambda = params["lambda"][0]

    existing_results = [
        (0.0001, 0.0001, 0.1, -0.423)
    ]
    for tmp_alpha, tmp_beta, tmp_lambda, result in existing_results:
        if tmp_alpha == alpha and tmp_beta == beta and tmp_lambda == _lambda:
            print "Found result " + str(result)
            return result

    return lftm_topic_coherence(alpha, beta, _lambda)
