from subprocess import Popen, PIPE
import pandas as pd
import numpy as np
# sourceFile = open('out.txt', 'w')
# cmd = "java -jar RankLib-2.16.jar -train light_shap_train.txt -ranker 0 -metric2t NDCG@50 -tree 500 -leaf 300 -shrinkage 0.03 -mls 5 -save davis_model.txt "
# process = Popen(cmd.split(), stdout=PIPE, stderr=PIPE)
# stdout, stderr = process.communicate()
# print(stdout.decode('utf-8'),file=sourceFile)
# print(stderr)

import csv



def readdata(file="light_shap_train.txt",i=1):
    df = pd.read_csv(file, delimiter=r"\s+", header=None, engine="python")
    a = [0, 1]
    b = df.columns[2:][:i].to_list()
    data = df[df.columns[a + b]]

    last2_datalist = []
    last1_datalist = []
    for index, x in enumerate(data[data.columns[-1]]):
        last2_datalist.append(x + "# ")
        last1_datalist.append(str(index + 1))

    last1col = pd.Series(last1_datalist)
    last2col = pd.Series(last2_datalist)
    data.iloc[:,-2] = last2col
    data.iloc[:, -1] = last1col
    outfile = f"{file}_{i}"
    data.to_csv(f"result/{outfile}", sep=" ", header=False, index=False)
    with open(f"result/{outfile}") as f:
        text = f.readlines()
    with open(f"result/{outfile}","w") as f:
        for line in text:
            line = line.replace('"','')
            f.write(line)
    return outfile

file = "light_shap_test_#R.txt"
df = pd.read_csv(file, delimiter=r"\s+", header=None, engine="python")


for x in range(0,(len(df.columns)-2),5):
    x = x

    outfile = readdata(file, x)
    data = f"result/{outfile}"


    sourceFile = open(f'test_result2_out/out_{x}.txt', 'w')
    print(".",end=" ")

   # cmd = f"java -jar RankLib-2.16.jar -train result/{outfile} -ranker 0 -metric2t NDCG@50 -tree 500 -leaf 300 -shrinkage 0.03 -mls 5 -save result_model/davis_model_{x}.txt "
    cmd2 =  f"java -jar RankLib-2.16.jar -load result_model/davis_model_{x}.txt  -rank result/light_shap_test_#R.txt_{x} -indri test_result/davis_test_sam_rank_{x}.txt"
    print(cmd2)
    process = Popen(cmd2.split(), stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(stdout.decode("utf-8"),file=sourceFile)
    #print(stdout.decode('utf-8'),file=sourceFile)
    if stderr.decode("utf-8") != "":
         print(str(stderr))
