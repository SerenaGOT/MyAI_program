import math
import datetime
starttime = datetime.datetime.now() # record the start time of the code

# basic train test
allwords = list([])
lines = list()
unique_words = list()
unique_num = 0
LenNum = 246
OHmatrix=list()
emotion = list()
emotion_type = ['anger','digust','fear','joy','sad','surprise']
smatrix =[]
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
train_set = []
test_set = []



def ReadFile():
    with open("classification/train.txt") as file:
        global allwords, unique_words,lines,unique_num,test_line_num,test_line,emotion,test_emotion,smatrix
        i=0
        smatrix = [[]]
        emotion = []*LenNum
        test_emotion = []
        for line in file.readlines():
            if i == 0:
                i+=1
                continue
            else:
                i+=1
            words = line.split(' ', 3)
            emotion.append([words[1],words[2]])
            words = words[3].replace('\n', '').split()  # split the words and keep the latter part
            #print words
            allwords = allwords + words  # all words appear
            lines = lines + [words]  # seperate words in seperate lines
    with open("classification/test.txt") as file2:      # include all the unique_words to form the vector
        i = 0
        for line in file2.readlines():
            if i == 0:
                i += 1
                continue
            else:
                i += 1
                test_line_num += 1
            words = line.split(' ', 3)
            test_emotion.append(words[2])    # store the emotion

            words = words[3].replace('\n', '').split()  # split the words and keep the latter part
            test_line = test_line + [words]
            #print test_line[test_line_num-1]  #checked
            allwords = allwords + words  # all words appear
        unique_words = list(set(allwords))
        unique_words.sort(key=allwords.index)  # total unique words
        unique_num = len(unique_words)
    return


def one_hot():
    global OHmatrix, LenNum, non_zero, test_line, test_line_num, test_distance, test_matrix
    global train_set, test_set
    LenNum = len(lines)
    OHmatrix = [([0] * unique_num) for i in range(LenNum)]  # create 2-dimension list to store the information
    for i in range(LenNum):
        temp = set()
        for j in range(unique_num):
            if unique_words[j] in lines[i]:  # find if the unique_word exist in the j training sample
                temp.add(j)                  # create smatrix
                OHmatrix[i][j] = 1
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


#faster 30 times

def get_block_dis(i,j):
    dis = len((train_set[j] | test_set[i]) - (train_set[j] & test_set[i]))  # one hot
    return dis

def get_euclidean_dis(i,j):
    dis = len((train_set[j] | test_set[i]) - (train_set[j] & test_set[i])) #one hot
    return math.sqrt(dis)

def get_cosine_dis(i,j):
    dis = a = b =  ab = 0
    interset = train_set[j] & test_set[i]
    ab = len(interset)
    a = math.sqrt(len(train_set))
    b = math.sqrt(len(test_set))
    dis =  float(ab)/(a*b)
    return dis

def test():
    global train_set,test_set,test_distance,all_test_line
    test_distance = []
    for i in range(test_line_num):
        for j in range(LenNum):
            dis = get_block_dis(i,j)  #get the different part that has value
            test_distance.append([j, dis])
        test_distance.sort(lambda x, y: cmp(x[1], y[1]))  #sort the list by the second position of value
        all_test_line.append(test_distance)
        test_distance = []

# create the one-hot matrix for test file  too  slow
def test2():
    global allwords, unique_words, lines, unique_num,test_line,test_line_num,test_distance,sort_distance,ans,test_matrix
    global all_test_line
    test_distance = []
    for i in range(test_line_num):
        for j in range(LenNum):
            dis = 0
            for k in range(unique_num):
                if test_matrix[i][k] != OHmatrix[j][k]:
                    dis += 1
            test_distance.append([j,math.sqrt(dis)])
        test_distance.sort(lambda x,y:cmp(x[1],y[1]))
        all_test_line.append(test_distance)
        test_distance = []
    return


#get the final answer for each test
def knn(k_num):
    for i in range(test_line_num):
        if k_num == 1:
            ans.append(emotion[all_test_line[i][0][0]][1])
        else:
            cnt = [0] * len(emotion_type)
            for j in range(k_num):
                po = int(emotion[all_test_line[i][j][0]][0]) - 1
                cnt[po] += 1
            maxx = 0
            po = 0
            for j in range(len(emotion_type)):
                if cnt[j] > maxx:
                    po = j
                    maxx = cnt[j]
            ans.append(emotion_type[po])

# calculate the corect rate
def check_answer():
    global  correct_rate
    num = 0
    for i in range(test_line_num):
        if test_emotion[i] == ans[i]:
            num += 1
    correct_rate = float(num)/test_line_num


print "Start!"
ReadFile()
print "Finish reading files..."
one_hot()
test()
print "Finish geting distances..."
print "Output the result:"
print "k   correct_rate"
for i in range(1,65):
    knn(i)
    check_answer()
    print i,"   ",correct_rate
    ans = []
endtime = datetime.datetime.now()
print "Total cost of time : ", (endtime - starttime).seconds , "s\n"