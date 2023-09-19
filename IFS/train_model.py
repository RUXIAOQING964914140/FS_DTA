from subprocess import Popen, PIPE
import pandas as pd
import numpy as np
# sourceFile = open('out.txt', 'w')
# cmd = "java -jar RankLib-2.16.jar -train light_shap_train.txt -ranker 0 -metric2t NDCG@50 -tree 500 -leaf 300 -shrinkage 0.03 -mls 5 -save davis_model.txt "
# process = Popen(cmd.split(), stdout=PIPE, stderr=PIPE)
# stdout, stderr = process.communicate()
# print(stdout.decode('utf-8'),file=sourceFile)
# print(stderr)





def readdata(file="light_shap_train.txt",i=1):
    df = pd.read_csv(file, delimiter=r"\s+", header=None, engine="python")
    a = [0, 1]
    b = df.columns[2:][:i].to_list()
    data = df[df.columns[a + b]]

    outfile = f"{file}_{i}"
    data.to_csv(f"result/{outfile}", sep=" ", header=False, index=False)
    return outfile

file = "light_shap_train.txt"
df = pd.read_csv(file, delimiter=r"\s+", header=None, engine="python")
for x in range(0,(len(df.columns)-2),5):
    x = x

    outfile = readdata(file, x)
    data = f"result/{outfile}"


    sourceFile = open(f'result2/out_{x}.txt', 'w')
    print(".",end=" ")

    cmd = f"java -jar RankLib-2.16.jar -train result/{outfile} -ranker 0 -metric2t NDCG@50 -tree 500 -leaf 300 -shrinkage 0.03 -mls 5 -save result_model/davis_model_{x}.txt "

    process = Popen(cmd.split(), stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8"), file=sourceFile)
    #print(stdout.decode('utf-8'),file=sourceFile)
    if stderr.decode("utf-8") != "":
         print(str(stderr))




