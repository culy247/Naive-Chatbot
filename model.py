from subprocess import call
call(['sh', 'word-represent.sh'])
from sklearn.neighbors import NearestNeighbors
import numpy as np

def buildVectorQuestion(args):
    with open('data/message.txt', 'w+') as f:
        f.write(args)
        f.close()
    call(['sh', 'message-vector.sh'])
    
def getVectorQuestion(path):
    with open(path, 'r') as f:
        vec_list_question = []
        for line in f:
            line = line.split()
            vec_list_question.append(line)
    return vec_list_question
            
def getSimilarMessage(args):
    buildVectorQuestion(args)
    #get vector question and message
    Question = getVectorQuestion('data/question_vector.txt')
    Message = getVectorQuestion('data/message_vector.txt')
    Qu = np.array(Question)
    Me = np.array(Message)
    #fit Nearest Neighbor
    Out = NearestNeighbors(n_neighbors=1)
    Out.fit(Qu)
    #find out Nearest Neighbor
    flagMessage = Out.kneighbors(X=Me, n_neighbors=1, return_distance=False)
    return flagMessage
import random
def getAnswerMessage(question):
    path = 'data/respond'
    with open(path + '/' + str(question), 'r') as f:
        vec = []
        for line in f:
            vec.append(line)
        f.close()
    resNum = random.random() * len(vec)
    return vec[int(resNum)]
 