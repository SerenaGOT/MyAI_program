import math
import datetime
starttime = datetime.datetime.now() # record the start time of the code
# -*- coding: utf-8 -*
# basic train test
allwords = list([])
lines = list()
unique_words = list()
unique_num = 0
LenNum = 246
OHmatrix=list()
TFmatrix = []
emotion = list()
emotion_type = ['anger','digust','fear','joy','sad','surprise']
train_set = []
test_set = []

#test part
test_line_num = 0
test_line=list([])
test_matrix=[[]]
test_emotion = list()


train_sets = list([])
test_distance = []
sort_distance = []
ans = []
correct_rate = 0
all_test_line = []
emotion_cal_ans = []

# read file get train samples and emotions
def ReadFile():
    with open("Regression/Dataset_train.csv") as file:
        global allwords, unique_words,lines,unique_num,test_line_num,test_line,emotion,test_emotion,LenNum
        i=0
        emotion = []*LenNum
        test_emotion = []
        for line in file.readlines():
            if i == 0:
                i+=1
                continue
            else:
                i+=1
            words = line.split(',', 2)
            emotion.append(words[2].replace('\n','').replace('\r','').split(','))
            words = words[1].split()  # split the words and keep the latter part
            #print emotion
            allwords = allwords + words  # all words appear
            lines = lines + [words]  # seperate words in seperate lines
        LenNum = i-1
    with open("Regression/Dataset_validation.csv") as file2:      # include all the unique_words to form the vector
        i = 0
        for line in file2.readlines():
            if i == 0:
                i += 1
                continue
            else:
                i += 1
                test_line_num += 1
            words = line.split(',', 2)
            test_emotion.append(words[2].replace('\r','').replace('\n','').split(','))    # store the  test emotion
            words = words[1].split()  # split the words and keep the latter part
            test_line = test_line + [words]
            allwords = allwords + words  # all words appear
        unique_words = list(set(allwords))
        unique_words.sort(key=allwords.index)  # total unique words
        unique_num = len(unique_words)
    return

#  create one hot matrix
def one_hot():
    global OHmatrix, LenNum, non_zero, test_line, test_line_num, test_distance, test_matrix
    LenNum = len(lines)
    OHmatrix = [([0] * unique_num) for i in range(LenNum)]  # create 2-dimension list to store the information
    for i in range(LenNum):
        temp = set()
        for j in range(unique_num):
            if unique_words[j] in lines[i]:  # find if the unique_word exist in the j training sample
                OHmatrix[i][j] = 1
                temp.add(j)
            else:
                OHmatrix[i][j] = 0
        train_set.append(temp)
    #test
    test_matrix = [([0] * unique_num) for i in range(test_line_num)]  # create 2-dimension list to store the information
    for i in range(test_line_num):
        temp = set()
        for j in range(len(test_line[i])):
            for k in range(unique_num):
                if unique_words[k] == test_line[i][j]:
                    test_matrix[i][k] = 1
                    temp.add(k)
                    break
        test_set.append(temp)
    return


def TF():
    global TFmatrix, LenNum, idf, test_matrix
    LenNum = len(lines)
    idf = [0] * unique_num
    TFmatrix = [([0] * unique_num) for i in range(LenNum)]  # create 2-dimension list to store the information
    for i in range(LenNum):
        num_of_words = len(lines[i])
        k = 0
        temp = set()
        for word in unique_words:
            cnt = 0
            for j in range(num_of_words):
                if (word == lines[i][j]):  # count the frequency
                    cnt += 1
            if cnt != 0:
                TFmatrix[i][k] = float(cnt) / float(num_of_words)
                temp.add(k)
            else:
                TFmatrix[i][k] = 0
            k += 1
        train_set.append(temp)

    test_matrix = [([0] * unique_num) for i in range(test_line_num)]  # create 2-dimension list to store the information
    for i in range(test_line_num):
        num_of_words = len(test_line[i])
        k = 0
        temp = set()
        for word in unique_words:
            cnt = 0
            for j in range(num_of_words):
                if word == test_line[i][j]:
                    cnt += 1
            if cnt > 0:
                test_matrix[i][k] = float(cnt) / float(num_of_words)
                temp.add(k)
            else:
                test_matrix[i][k] = 0
            k += 1
        test_set.append(temp)
    return

#-----  get distance from different method ------ #

def get_block_dis(i,j,flag):
    dis = 0
    if flag == "TF":
        interset = train_set[j] & test_set[i]
        union = train_set[j] | test_set[i]
        for k in range(len(interset)):
            dis += math.pow(TFmatrix[j][list(interset)[k]] - test_matrix[i][list(interset)[k]], 2)
        train_subset = list(train_set[j] - interset)
        test_subset = list(test_set[i] - interset)
        for k in range(len(train_subset)):
            dis += math.fabs(TFmatrix[j][train_subset[k]])
        for k in range(len(test_subset)):
            dis += math.fabs(test_matrix[i][test_subset[k]])
    elif flag == "OH":
        dis = len((train_set[j] | test_set[i]) - (train_set[j] & test_set[i]))  # one hot
    return dis

def get_euclidean_dis(i,j,flag):
    dis = 0
    if flag == "TF":
        interset = train_set[j] & test_set[i]
        union =  train_set[j] | test_set[i]
        for k in range(len(interset)):
            dis += math.pow(TFmatrix[j][list(interset)[k]]-test_matrix[i][list(interset)[k]],2)
        train_subset = list(train_set[j] - interset)
        test_subset = list(test_set[i] - interset)
        for k in range(len(train_subset)):
            dis += math.pow(TFmatrix[j][train_subset[k]],2)
        for k in range(len(test_subset)):
            dis += math.pow(test_matrix[i][test_subset[k]],2)
    elif flag == "OH":
        dis = len((train_set[j] | test_set[i]) - (train_set[j] & test_set[i])) #one hot
    return math.sqrt(dis)

def get_cosine_dis(i,j,flag):
    dis = a = b =  ab = 0
    if flag == "TF":
        interset = train_set[j] & test_set[i]
        sub = list(interset)
        for k in range(len(interset)):
            ab += TFmatrix[j][sub[k]] * test_matrix[i][sub[k]]
        for k in range(len(train_set[j])):
            a += math.pow(TFmatrix[j][list(train_set[j])[k]],2)
        for k in range(len(test_set[i])):
            b += math.pow(test_matrix[i][list(test_set[i])[k]],2)
        dis = float(ab)/(math.sqrt(a)*math.sqrt(b))
    elif flag == "OH":
        interset = train_set[j] & test_set[i]
        ab = len(interset)
        a = len(train_set)
        b = len(test_set)
        dis =  float(ab)/(a*b)
    return dis
#-----  get distance from different method ------ #

# create the one-hot matrix for test file
def test(flag):
    global allwords, unique_words, lines, unique_num,test_line,test_line_num,test_distance,sort_distance,ans,test_matrix,OHmatrix,TFmatrix
    global all_test_line
    test_distance = []
    for i in range(test_line_num):
        for j in range(LenNum):
            dis = get_euclidean_dis(i,j,flag)
            test_distance.append([j,dis])
        test_distance.sort(lambda x,y:cmp(x[1],y[1]))
        all_test_line.append(test_distance)
        test_distance = []
    return


# calculate the average and variance
def get_avg_var(k_num,line):
    ans_max_min = []
    for k in range(k_num):
        ans_max_min.append(1.0/all_test_line[line][k][1])
    avg = sum(ans_max_min)/k_num
    var = 0
    for k in range(k_num):
        var += math.pow((ans_max_min[k]-avg),2)
    var = var / k_num
    return [avg,var]

#calculate the maximun and minimum distance in each test line
def get_max_min(k_num,line):
    ans_max_min = []
    for k in range(k_num):
        ans_max_min.append(all_test_line[line][k][1])
       # print ans_max_min
    return [1.0/min(ans_max_min),1.0/max(ans_max_min)]


def print_answer(k_num):
    with open("outputk.txt",'w') as file:
        global  emotion_cal_ans
        emotion_cal_ans = [([0] * len(emotion_type)) for i in range(test_line_num)]
        if k_num == 1:
            for i in range(test_line_num):
                    for k in range(6):
                        po = int(all_test_line[i][0][0])
                        if all_test_line[i][0][1] == 0:
                            continue
                        temp = float(emotion[po][k]) / all_test_line[i][0][1]
                        file.write(str(temp) + ",")
                        emotion_cal_ans[i][k] = temp
                    file.write("\n")
        else:
            for i in range(test_line_num):
                sum = 0
                six = []
                file.write("text"+str(i+1)+" ")
                for j in range(6):
                    temp = 0
                    normalized = get_max_min(k_num, i)
                    for k in range(k_num):
                        po = int(all_test_line[i][k][0])
                        if all_test_line[i][k][1]==0:
                            continue
                        #temp += float(emotion[po][j])/all_test_line[i][k][1]
                        temp += float(emotion[po][j])*((1.0/all_test_line[i][k][1])-normalized[1])/(normalized[0]-normalized[1])  ##normalize
                    sum += temp
                    six.append(temp)
                for j in range(6):
                    file.write(str(six[j]/sum) + " ")
                    emotion_cal_ans[i][j] = six[j]/sum
                file.write("\n")

def get_relative_score():
    res = 0
    for z in range(6):
        x = 0
        y = 0
        for i in range(test_line_num):
                x += float(test_emotion[i][z])
                y += float(emotion_cal_ans[i][z])
        test_size = test_line_num
        avgX = float(x) / test_size
        avgY = float(y) / test_size
        varx = vary = cov = 0
        for i in range(test_line_num):
                varx += math.pow(float(test_emotion[i][z]) - avgX, 2)
                vary += math.pow(float(emotion_cal_ans[i][z]) - avgY, 2)
                cov += (float(test_emotion[i][z]) - avgX) * (float(emotion_cal_ans[i][z]) - avgY)
        res += cov / (math.sqrt(varx * vary))
    return res/6.0


ReadFile()
#one_hot()   # delete the # when use one hot
TF()
flag = "TF"   #OH for one-hot ; TF for TFmatrix
print "k ","  relative score"
for k in range(10,65):
    test(flag)
    print_answer(k)
    print k," ",get_relative_score()
endtime = datetime.datetime.now()
print "Total cost of time : ", (endtime - starttime).seconds , "s\n"
