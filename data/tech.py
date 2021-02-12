from os import listdir
import torch
import IPython

class matrix:
    def __init__(self,name0,dirName,myDic):
        self.name = name0
        self.readData(dirName,myDic)
        self.getMatrix()

    def readData(self,dirName,myDic):
        with open(dirName+self.name+'/features.idx') as f:
            tmp=f.readlines()
            self.n=len(tmp)-9
            self.Map=[]
            for i in range(9,len(tmp)):
                sp=tmp[i].split(' ')
                self.Map.append(myDic[sp[1].replace('\n','')])
        with open(dirName+self.name+'/vectors.dat') as f:
            tmp=f.readlines()
            if (len(tmp)-2)%2==1:
                print("Format error, lines not even")
            self.d=(len(tmp)-2)//2
            self.L=[]
            for i in tmp:
                if i[0]!='#':
                    curCol=[]
                    sp=i.split(' ')
                    for j in range(1,len(sp)):
                        sp2=sp[j].split(':')
                        curCol.append((int(sp2[0])-1,float(sp2[1])))
                    self.L.append(curCol)
    def getMatrix(self):
        self.M=torch.zeros(self.n,self.d)
        colInd=0
        for curCol in self.L:
            for i in curCol:
                self.M[i[0]][colInd]=i[1]
            colInd+=1
        return self.M

def processRaw(rawdir,scale):
    dirName=rawdir+'raw/tech300/'
    folders = [f for f in listdir(dirName)]
    N_train=200
    N_test=len(folders)-N_train
    print("N_train: 200, N_test: %d" % N_test)
    ID=0
    myDic={}
    for curFolder in folders:
        with open(dirName+curFolder+'/features.idx') as f:
            a=f.readlines()
            if (a[9][:2]!='1 '):
                print("error!",curFolder)
            for i in range(9,len(a)):
                sp=a[i].split(' ')
                key=sp[1].replace('\n','')
                if not (key in myDic):
                    myDic[key]=ID
                    ID+=1

    n=len(myDic)
    A_train=[]
    A_test=[]
    for i in range(N_train):
        curM=matrix(folders[i],dirName,myDic)
        U, S, V = curM.M.svd()
        div=S[0].item()/scale
        A_train.append({'M':curM.M/div, 'Map':curM.Map, 'n':curM.n, 'd':curM.d})
        print(i)
    for i in range(N_test):
        curM=matrix(folders[i+N_train],dirName,myDic)
        U, S, V = curM.M.svd()
        div=S[0].item()/100
        A_test.append({'M':curM.M/div, 'Map':curM.Map, 'n':curM.n, 'd':curM.d})
        print(i+N_train)
    torch.save([A_train,A_test,n],dirName[:-1]+"_"+str(scale)+".dat")
    print("Done processing!")
def getTech(raw,rawdir,scale):
    if raw:
        processRaw(rawdir,scale)
    A_train,A_test,n =torch.load(rawdir+'raw/tech300_'+str(scale)+'.dat')
    # best_train = 50.75971984863281
    # best_test = 22.81448745727539
    return A_train,A_test,n,-1
