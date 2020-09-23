import numpy as np
import pandas as pd
from sklearn import preprocessing

class PreProcessing():
    def hangouts(self, hangouts):
        ho_feat = []
        for _, each in enumerate(hangouts):
            ho_feat.append([each["agon"],each["alea"],each["mimicry"],each["ilinx"]])
        return np.array(ho_feat)
                           
    def lt_trand(self, lt_trand):
        return np.array([lt_trand["agon"],lt_trand["alea"],lt_trand["mimicry"],lt_trand["ilinx"]])
    
    def answers(self, answers):
        return np.array([answers["q1"],answers["q2"]])


class ShortTerm():
    def __init__(self, hangouts, user_lt, answers):
        self.hangouts = hangouts
        self.user_lt = user_lt
        self.answers = self.preprocessing(answers)    
        self.user_st = np.zeros(4)
                           
    def preprocessing(self, anss):
        q1 = ((anss[0] - 1) * 25) / 100
        q2 = ((anss[1] - 1) * 25) / 100
        return np.array([q1,q2])
        
    def calc_st_trand(self, q1, q2, alpha):
        agon    = alpha * self.user_lt[0] 
                + (1 - alpha) * 0.5 * ((1-self.answers[0]) + self.answers[1])
        alea    = alpha * self.user_lt[1] 
                + (1 - alpha) * 0.5 * ((1-self.answers[0]) + (1-self.answers[1]))
        mimicry = alpha * self.user_lt[2]
                + (1 - alpha) * 0.5 * (self.answers[0] + (1-self.answers[1]))
        ilinx   = alpha * self.user_lt[3]
                + (1 - alpha) * 0.5 * (self.answers[0] + self.answers[1])
        self.update([agon,alea,mimicry,ilinx])
    
    def update(self, elements):
        for i, element in enumerate(elements)
            self.user_st[i] = element
    
    def run(self, anss):
        q1, q2 = self.preprocessing(anss)
        self.calc_st_trand(q1, q2, alpha=0.75)
        return self.user_st
        

class HangoutsRecommender():
    def __init__(self, hangouts, lt_trand, answers):
        self.ppc = PreProcessing()
        self.hangouts = self.ppc.hangouts(hangouts)
        self.lt_trand = self.ppc.lt_trand(lt_trand)
        self.answers = self.ppc.answers(answers)
        
    def calc_rank(self, user_st):
        results =  np.linalg.norm(self.hangouts - user_st, axis=1)
        return np.argsort(results) + 1
        
    def run(self):
        shortterm = ShortTerm(self.hangouts, self.lt_trand, self.answers)
        user_st = shortterm.run(answers)
        rank = calc_rank(user_st)
        return dict(r1=rank[0], r2=rank[1], r3=rank[2], r4=rank[3], r5=rank[4])
                           
"""                           
class FriendsRecommender():
    def __init__(self, ):
"""