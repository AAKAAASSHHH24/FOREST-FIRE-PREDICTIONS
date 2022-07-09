import pymongo
import pandas as pd
import pickle
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

models = ['classification_model.pkl', 'regression_model.pkl']
classification_model = pickle.load(open(models[0], 'rb')) # loading the model file from the storage
regression_model = pickle.load(open(models[1], 'rb')) # loading the model file from the storage

class Bulk_Predictor:
    def __init__(self, client, db, collection):
        print("inside constractor")
        self.client = str(client)
        print(client) 
        self.db = str(db)
        self.collection = str(collection)
        self.client = pymongo.MongoClient(self.client)
        print(self.client, "clienttttttt")
        self.db = self.client[self.db]
        self.collection = self.db[self.collection]
        print(self.collection)
        

    def bulk_regression(self):
        #print("getRecords")
        results = []
        df = pd.DataFrame(columns=['RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI','FWI'])
        
        for i in self.collection.find():           
            mydict = {
                        'RH': i['RH'], 'Ws': i['Ws'], 'Rain': i['Rain'],
                        'FFMC': i['FFMC'], 'DMC': i['DMC'], 'DC': i['DC'],   
                        'ISI': i['ISI'], 'FWI': i['FWI']                         
            }
            df.loc[-1] = mydict.values()
            df.index = df.index + 1  # shifting index
            df = df.sort_index()  # sorting by index
            results.append(mydict)
        
        #logger.error('Something went wrong during dataframe extract')
        def f_reg(RH,Ws,Rain,FFMC,DMC,DC,ISI,FWI):
                return regression_model.predict([[RH,Ws, Rain,FFMC,DMC,DC,ISI,FWI]])[0]
        

        df['prediction temp'] = df.apply(lambda x: f_reg(x['RH'], x['Ws'], x['Rain'], x['FFMC'], x['DMC'], x['DC'], x['ISI'], x['FWI']), axis = 1)
        return df
    
    def bulk_classification(self):
        results = []
        df = pd.DataFrame(columns=['Temperature','RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI','FWI'])
        
        for i in self.collection.find():           

            mydict = {'Temperature': i['Temperature'],
                        'RH': i['RH'], 'Ws': i['Ws'], 'Rain': i['Rain'],
                        'FFMC': i['FFMC'], 'DMC': i['DMC'], 'DC': i['DC'],   
                        'ISI': i['ISI'], 'FWI': i['FWI']                         
            }
            df.loc[-1] = mydict.values()
            df.index = df.index + 1  # shifting index
            df = df.sort_index()  # sorting by index
            results.append(mydict)
            
            
        def f_class(Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,FWI):
            if classification_model.predict([[Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,FWI]])[0] == 0:
                return "Not Fire"
            else:
                return "Fire"
            
        df['prediction classes'] = df.apply(lambda x: f_class(x['Temperature'],x['RH'], x['Ws'], x['Rain'], x['FFMC'], x['DMC'], x['DC'], x['ISI'],x['FWI']), axis = 1)
        return df