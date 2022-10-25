import numpy as np
import pandas as pd
import pickle
import json

class medical_insurance():
    def __init__(self,age,sex,bim,children,smoker,region):
        self.age=age
        self.sex=sex
        self.bim=bim
        self.children=children
        self.smoker=smoker
        self.region="region_"+ region
    def load_mode(self):
        with open ("model.pkl","rb") as f:
            self.model1=pickle.load(f)

        with open("project_data.json","r")as f:
            self.json_data=json.load(f)
    def prediction(self):
        self.load_mode()
        region_index=self.json_data["colums"].index(self.region)

        array=np.zeros(len(self.json_data["colums"]))
        array[0]=self.age
        array[1]=self.json_data['sex'][self.sex]
        array[2]=self.bim
        array[3]=self.children
        array[4]=self.json_data["smoker"][self.smoker]
        array[region_index]=1
        print("test data",array)
        price=self.model1.predict([array])[0]
        print("price is ",price)
        return price
# if __name__=="__main__":
age=56
sex='male'
bmi=27.9
children=4
smoker='no'
region='northeast'
med_ins=medical_insurance(age, sex, bmi, children, smoker,region)

med_ins.prediction()




