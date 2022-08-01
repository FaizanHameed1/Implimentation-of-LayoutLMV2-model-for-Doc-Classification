
import torch
import parser
from dataframe_split import *
from dataloader import Preprocess
from torch.utils.data import DataLoader
from test import Test
from results import Results

#reading config file
train_config=parser.train_config
test_config=parser.test_config

#test data path
test_data = test_config["test_data"]
test_ocr_path = test_config["test_ocr_path"]
#batch size
batch_size = train_config["batch_size"]

#creating dataframe
df_test=create_df(test_data,test_ocr_path)
print(df_test.head())
print("test dataframe created successfully")
#df_test=df_test[:50]



#preprocessing test data
test_data = Preprocess(df_test)
print("preprocessing process completed")




#creating dataloader for testset
test_dataloader = DataLoader(test_data, batch_size = batch_size,shuffle=False)
#print("length of test_dataloader".format(len(test_dataloader)))

#testing the model
print("testing the model on test data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tst=Test(test_dataloader,device,df_test)
y_pred, y_label = tst.test()

#getting the results
res =  Results(y_pred,y_label)

#classification report
res.classification_report()
#accuracy score
res.accuracy_score()

#confusion metrix
res.confusion_matrix()
