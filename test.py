import torch
import os
import parser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



test_config=parser.test_config

class Test():
    def __init__(self,test_dataloader,device,df_test):
        self.test_dataloader = test_dataloader
        self.device = device
        self.df_test=df_test
        self.checkpoint_path= test_config["test_model_path"]

    #loading model and its parameters
    def load_checkpoint(self,save_path):
        dict_parameters = torch.load(save_path, map_location=self.device)
        model = dict_parameters["model"]
        val_acc = dict_parameters["val_acc"]
        optimizer = dict_parameters["optimizer"]
        globel_step = dict_parameters["step"]
        epoch = dict_parameters["epoch"]
        return model, epoch,optimizer,val_acc, globel_step

    def test(self):
        #loading the model
        model,_,_,_,_ = self.load_checkpoint(self.checkpoint_path)
        print("model is successfully loaded")
        #Checking on test dataset
        model.to(self.device)
        model.eval()

        ###
        y_pred=[]
        y_label=[]
        correct = 0
        
        for batch in tqdm(self.test_dataloader):
            
            input_ids = batch["input_ids"].squeeze(1).to(self.device)
            bbox = batch["bbox"].squeeze(1).to(self.device)
            attention_mask = batch["attention_mask"].squeeze(1).to(self.device)
            token_type_ids = batch["token_type_ids"].squeeze(1).to(self.device)
            labels = batch["labels"].squeeze(1).to(self.device)
            image=batch['image'].squeeze(1).to(self.device)
            
        
            with torch.no_grad():
                outputs = model(image=image,input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,labels=labels)


            predictions = outputs.logits.argmax(-1)
            # print("#########")
            # print(predictions)
            # print("#########")
            y_pred.append(predictions)
            ###
            y_label.append(labels)
            # print("###########")
            # print(labels)
            # print(predictions)
            # print("#########")
            correct += (predictions == labels).float().sum()
            #print(correct)

        accuracy = 100 * correct / len(self.df_test)
        print("Testing accuracy:", accuracy.item())
        return y_pred, y_label