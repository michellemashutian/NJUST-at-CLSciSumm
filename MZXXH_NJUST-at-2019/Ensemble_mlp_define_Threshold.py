import numpy as np
def define_Threshold(x,f,Threshold):
    y=[]
    for line in f:
        a=line.replace("[","").replace("]","").split(",")
        if len(line)==0:
            continue
        y.append(float(a[0].strip()))
    number_of_true=0
    all_1_pred=0
    all_1=0
    for i in range(len(y)):
        if y[i]>Threshold:
            y[i]=1
        else:
            y[i]=0
        if x[i]==1:
            all_1+=1
        if y[i]==1:
            all_1_pred+=1
        if y[i]==x[i] and y[i]==1:
            number_of_true+=1
    if all_1_pred==0:
        p=0
    else:
        p=number_of_true/all_1_pred
    r=number_of_true/all_1
    if p==r==0:
        f=0
    else:
        f=p*r*2/(p+r)
    prf=[p,r,f]
    # print("准确："+str(p))
    # print("召回:"+str(r))
    # print("f:"+str(f))
    return prf
import matplotlib.pyplot as plt
def tj(target_path):
    x1=range(1,1001)
    f_list=[]
    max_p=0
    max_r=0
    max_f1=0
    last_Threshold=0
    output=open(r"E:\data\cv\d2v_tj.txt","a+",encoding="utf8")
    standrand_path=r"E:\data\cv\test_label.txt"
    #target_path=r"E:\data\cv\pred_probility\d2v-adam-mean_squared_error.txt"
    x = np.loadtxt(standrand_path).tolist()
    f = open(target_path, encoding="utf8").read().split("\n")
    print(target_path)
    for i in range(1000):
        prf=define_Threshold(x,f,i/1000)
        p=prf[0]
        r=prf[1]
        f1=prf[2]
        f_list.append(f1)
        if f1>max_f1:
            max_f1=f1
            max_p=p
            max_r=r
            last_Threshold=i/1000
    print("max_f1:" + str(max_f1))
    print("p:" + str(max_p))
    print("r:" + str(max_r))
    print("last_Threshold:" + str(last_Threshold))
    output.write(target_path+"============="+"\n")
    output.write("max_f1:" + str(max_f1)+"\n")
    output.write("p:" + str(max_p)+"\n")
    output.write("r:" + str(max_r)+"\n")
    output.write("last_Threshold:" + str(last_Threshold)+"\n")
    plt.plot(x1, f_list)
    plt.show()
import glob
def main():
    paths=glob.glob(r"E:\data\cv\d2v\*.txt")
    for path in paths:
        tj(path)
main()