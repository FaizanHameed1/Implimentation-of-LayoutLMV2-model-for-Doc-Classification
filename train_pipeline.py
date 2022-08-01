# import parser
# import torch
# import torch.nn as nn
# train_config=parser.train_config

# ##classes
# class_path=train_config["class_path"]

# with open(class_path,'r') as file:
#     classes=file.read().split('\n')
#     #print(classes)

# from transformers import LayoutLMv2Model
# #

# model=LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# # print(model)
# # print("**********")
# # print(model.pooler)

# #changing the classifier layer
# model.pooler = nn.Sequential(nn.Flatten(),
#                            nn.Linear(768,128),
#                            nn.ReLU(),
#                            nn.Dropout(p=0.1),
#                            nn.Linear(128,13))#totel classes are 13

# print(model)
# # making Classifier layer Trainable                           
# # for parameter in model.classifier.parameters():
# #   print(parameter.requires_grad)
# #print(model)
# #print(type(model.config.hidden_dropout_prob))








##################################

from transformers import LayoutLMv2Processor

from torch.utils.data import DataLoader
import torch
from dataloader import Preprocess
from train import Train
from dataframe_split import *
import parser



train_config=parser.train_config

#data paths
train_data = train_config["train_data"]
train_ocr_path = train_config["train_ocr_path"]


#fraction of trainig data in train and validation split
val_split = train_config["validation_split"]

batch_size = train_config["batch_size"]

total_epochs=train_config["total_epochs"]


#creating dataframe
df_tr=create_df(train_data,train_ocr_path)
#splitting dataframe into train and test
df_train,df_val=split(df_tr,val_split)
# print("len of train df: {}".format(len(df_train)))
# print("len of validation df: {}".format(len(df_val)))

# df_train = df_train[:50]
# df_val = df_val[:10]


#dataloader and data preprocessing
train_data = Preprocess(df_train)
val_data=Preprocess(df_val)


#dataloader
train_dataloader = DataLoader(train_data, batch_size = batch_size,shuffle=True)
val_dataloader=DataLoader(val_data,batch_size=batch_size,shuffle=True)

# print("len of train dataloader: {}".format(len(train_dataloader)))
# print("len of validation dataloader: {}".format(len(val_dataloader)))

print("dataloader is created successfully")
print("*************")
sample = next(iter(train_dataloader))
# print(sample)
# for key,value in sample.items():
#   print(key, value.shape)

#traing the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trn=Train(total_epochs,train_dataloader,val_dataloader,device,df_train,df_val)#total epoch>>>>>>
# #
restore_training = train_config["resume_training"]
model = trn.run(restore_training)#True if we want to resume our training else False



