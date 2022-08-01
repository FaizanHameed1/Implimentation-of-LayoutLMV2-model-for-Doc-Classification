import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def create_df(dest_path,ocr_path):
  print("creating datframe ...")
  #making dataframe for xml,image and ocr_path
  dframe= pd.DataFrame()
  image_pth,ocr_pth,label=[],[],[]
  #making dataframe for xml,image and ocr_path
  dframe= pd.DataFrame()
  for folder in os.listdir(dest_path):
    folder_path=os.path.join(dest_path,folder)
    for file in os.listdir(folder_path):
      img_p=os.path.join(folder_path,file)
      ocr_p=os.path.join(ocr_path,file.replace(".png",".entities.json"))
      if os.path.exists(ocr_p):
        image_pth.append(img_p)
        label.append(folder)
        ocr_pth.append(ocr_p)
      else:
        pass
  dframe["images"]=image_pth
  dframe["ocrs"]=ocr_pth
  dframe["labels"]=label
  print("dataframe is created successfully")
  print("***********")
  return dframe



# #splitting into train and validation
# def split(df,train_fraction):
#   print("splitting dataframe into train and validation ...")
#   msk = np.random.rand(len(df)) < train_fraction
#   df_train = df[msk].reset_index()
#   df_val = df[~msk].reset_index()
#   print("dataframe splitting process completed") 
#   print("**********")
#   return df_train,df_val




#splitting into train and validation
def split(df,val_split):
  print("splitting dataframe into train and validation ...")
  feat=df[["images","ocrs"]]
  lab=df[["labels"]]
  x_train, x_val, y_train, y_val = train_test_split(feat, lab, test_size=val_split,random_state=2,stratify=lab)
  df_train= pd.concat([x_train, y_train], axis=1).reset_index()
  df_val=pd.concat([x_val,y_val],axis=1).reset_index()
  print("dataframe splitting process completed") 
  print("**********")
  return df_train,df_val
    
