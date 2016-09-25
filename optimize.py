import math
import datetime
starttime = datetime.datetime.now() # record the start time of the code
allwords = list([])
lines = list()
unique_words = list()
OHmatrix = [[]]
TFmatrix = [[]]
TFIDFmatrix = [[]]
idf=[]
unique_num = 0
LenNum = 0
non_zero = 0


def ReadFile():
    with open("semeval") as file:
        global allwords, unique_words,lines,unique_num
        for line in file.readlines():
            words = line.split('\t', 2)[2].replace('\n', '').split()  # split the words and keep the latter part
            print words
            allwords = allwords + words  # all words appear
            lines = lines + [words]  # seperate words in seperate lines
        unique_words = list(set(allwords))
        unique_words.sort(key=allwords.index)  # total unique words
        unique_num = len(unique_words)
    return


def one_hot():
    with open("one_hot.txt",'w') as OneHot:
        global OHmatrix,LenNum,non_zero
        LenNum = len(lines)
        OHmatrix = [([0] * unique_num) for i in range(LenNum)]  # create 2-dimension list to store the information
        for i in range(LenNum):
            for j in range(unique_num):
                if unique_words[j] in lines[i]:      # find if the unique_word exist in the j training sample
                    OneHot.write("1 ")
                    OHmatrix[i][j] = 1
                    non_zero+=1
                else:
                    OneHot.write("0 ")
                    OHmatrix[i][j] = 0
            OneHot.write("\n")
    return


def TF():
    with open("TF.txt", 'w') as TF:
        global TFmatrix, LenNum, idf
        LenNum = len(lines)
        idf = [0] * unique_num
        TFmatrix = [([0] * unique_num) for i in range(LenNum)] # create 2-dimension list to store the information
        for i in range(LenNum):
            num_of_words = len(lines[i])
            k = 0
            for word in unique_words:
                cnt = 0
                for j in range(num_of_words):
                    if(word==lines[i][j]):       #count the frequency
                        cnt+=1
                if cnt != 0:
                    TFmatrix[i][k] = float(cnt)/float(num_of_words)
                else:
                    TFmatrix[i][k] = 0
                TF.write(str(TFmatrix[i][k])+" ")
                k += 1
            TF.write("\n")
    return


def TFIDF():
    with open("TFIDF.txt", 'w') as TFIDF:
        global TFIDFmatrix, LenNum, TFmatrix
        LenNum = len(lines)
        TFIDFmatrix = [([0] * unique_num) for i in range(LenNum)]
        for i in range(unique_num):
            for l in lines:
                if unique_words[i] in l:  # record the appearance time
                    idf[i]+=1
        for i in range(LenNum):
            for j in range(unique_num):
                if TFmatrix[i][j] != 0:
                    TFIDFmatrix[i][j] = (TFmatrix[i][j] * math.log(float(LenNum)/(1+float(idf[j])))) # output the result in float type
                else:
                    TFIDFmatrix[i][j] = 0
                TFIDF.write(str(TFIDFmatrix[i][j])+" ")
            TFIDF.write("\n")
    return


def smatrix():
    with open("smatrix.txt", 'w') as smatrix:
        global OHmatrix, LenNum, non_zero
        smatrix.write(str(LenNum)+"\n")
        smatrix.write(str(unique_num) + "\n")
        smatrix.write(str(non_zero) + "\n")
        for i in range(LenNum):
            for j in range(unique_num):
                if OHmatrix[i][j] == 1:
                    smatrix.write("%d %d 1\n" %(i,j))
    return


def AplusB():
    with open("AplusB.txt", 'w') as AplusB:
        global OHmatrix, LenNum, non_zero
        half = LenNum/2
        AplusB.write(str(half)+"\n")
        AplusB.write(str(unique_num) + "\n")
        td = 0
        plusMatrix = [([0] * unique_num) for i in range(LenNum)]
        for i in range(half):
            for j in range(unique_num):
                plusMatrix[i][j] = OHmatrix[i][j] + OHmatrix[i+half][j]  # plus the 2 matrix
                if plusMatrix[i][j] > 0 :
                    td+=1
        AplusB.write(str(td) + "\n")
        for i in range(LenNum):
            for j in range(unique_num):
                if plusMatrix[i][j] > 0:
                    AplusB.write("%d %d %d\n" %(i,j,plusMatrix[i][j]))
    return


ReadFile()
one_hot()
TF()
TFIDF()
smatrix()
AplusB()
endtime = datetime.datetime.now()
print "Total cost of time : ", (endtime - starttime).seconds , "s\n"