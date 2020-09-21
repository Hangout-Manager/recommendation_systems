import numpy as np
import pandas as pd
from sklearn import preprocessing

class LongTerm():        
    def calc_lt_trand(self, user):
        return user.mean(axis=0)
    
    def run(self, user):
        return calc_lt_trand(user)
        
        
class ShortTerm():
    def __init__(self, user_lt, hangouts):
        self.hangouts = hangouts
        self.user_lt = user_lt
        self.user_st = user_lt.copy()
        
    def preprocessing(self, anss):
        q1 = (3 * 25) / 100
        q2 = (3 * 25) / 100
        return q1, q2
        
    def calc_st_trand(self, q1, q2, alpha):
        agon    = alpha * self.user_lt[0] + (1 - alpha) * 0.5 * ((1-q1) + q2)
        alea    = alpha * self.user_lt[1] + (1 - alpha) * 0.5 * ((1-q1) + (1-q2))
        mimicry = alpha * self.user_lt[2] + (1 - alpha) * 0.5 * (q1 + (1-q2))
        ilinx   = alpha * self.user_lt[3] + (1 - alpha) * 0.5 * (q1 + q2)
        self.update([agon,alea,mimicry,ilinx])
    
    def update(self, elements):
        self.user_st.loc["アゴン(競争を伴う遊び)"] = elements[0]
        self.user_st.loc["アレア(運やかけを伴う遊び)"] = elements[1]
        self.user_st.loc["ミミクリ(真似・模倣を伴う遊び)"] = elements[2]
        self.user_st.loc["イリンクス(目眩やスリルを伴う遊び)"] = elements[3]
    
    def run(self, anss):
        q1, q2 = self.preprocessing(anss)
        self.calc_st_trand(q1, q2, alpha=0.75)
        

class RecommenderSystem():
    def __init__(self, hangouts, user, answers):
        self.user = user
        self.answers = answers
        self.hangouts = hangouts
        
    def calc_rank(self, user):
        results =  np.linalg.norm(self.hangouts - user, axis=1)
        return np.argsort(results)
        
    def run(self):
        longterm = LongTerm()
        user_lt = longterm.run(self.user)
        shortterm = ShortTerm(user_lt, self.hangouts)
        user_st = shortterm.run(answers)

