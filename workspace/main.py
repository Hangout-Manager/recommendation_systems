import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity

class PreProcessing():
    def get_all_features(self, dataset):
        features = []
        for _, data in enumerate(dataset):
            features.append([data["agon"],data["alea"],data["mimicry"],data["ilinx"]])
        return np.array(features)
                           
    def get_user_features(self, lt_trand):
        return np.array([lt_trand["agon"],lt_trand["alea"],lt_trand["mimicry"],lt_trand["ilinx"]])
    
    def get_answers(self, answers):
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
    def __init__(self, hangouts, lt_trand, answers, covid_risk):
        self.ppc = PreProcessing()
        self.hangouts = self.ppc.get_all_features(hangouts)
        self.lt_trand = self.ppc.get_user_features(lt_trand)
        self.answers = self.ppc.get_answers(answers)
        self.covid_risk = covid_risk
        
    def get_recommend(self, user_st):
        results =  np.linalg.norm(self.hangouts - user_st, axis=1)
        return np.argsort(results)
    
    def get_ranking(self, rec_index):
        cons_covid = np.array((10,2))
        for i, ho_idx in enumerate(rec_index):
            cons_covid[i,0] = self.covid_risk[ho_idx]
            cons_covid[i,1] = ho_idx
        results = cons_covid[np.argsort(cons_covid[:,0])]
        return cons_covid[0:5,1] + 1
        
        
    def run(self):
        shortterm = ShortTerm(self.hangouts, self.lt_trand, self.answers)
        user_st = shortterm.run(answers)
        recommend = self.get_recommend(user_st)
        ranking = get_ranking(recommend[:10])
        return dict(r1=ranking[0], r2=ranking[1], r3=ranking[2], r4=ranking[3], r5=ranking[4])
                           
                  
class FriendsRecommender():
    def __init__(self, all_users, user):
        self.ppc = PreProcessing()
        self.all_users = self.ppc.get_all_features(all_users)
        self.user = self.ppc.get_user_features(user)
        self.mm = preprocessing.MinMaxScaler()

    def calc_euclid(self):
        results = np.linalg.norm(self.all_users - self.user, axis=1)
        return self.mm.fit_transform(results)
        
    def calc_cos_simi(self):
        return cosine_similarity(self.all_users, self.user)
    
    def calc_eval(self):
        euclid_vals = self.calc_euclid()
        simi_vals = self.calc_cos_simi()
        eval_vals = euclid_vals - simi_vals
        return eval_vals
    
    def get_ranking(self, eval_vals):
        return np.argsort(eval_vals) + 1
    
    def run(self):
        eval_vals = self.calc_eval()
        recom= get_ranking(eval_vals)
        return dict(r1=recom[0], r2=recom[1], r3=recom[2], r4=recom[3], r5=recom[4])