import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ExponentialLR
# from transformers import LayoutLMv2Model
from transformers import LayoutLMv2ForSequenceClassification
from transformers import AdamW
#from tqdm.notebook import tqdm\
from tqdm import tqdm
import os
import sys
import parser
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")

train_config=parser.train_config

tens_board_dir=train_config["tens_board"]
writer = SummaryWriter(tens_board_dir)#for tensor board

##classes
class_path=train_config["class_path"]

with open(class_path,'r') as file:
    classes=file.read().split('\n')
    #print(classes)

class Train():

  def __init__(self,total_epochs,train_dataloader,val_dataloader,device,df_train,df_val):
    self.device=device
    self.checkpoint_path=train_config["checkpoint_path"]
    self.loss_list_path=train_config["loss_list_path"]
    self.model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutlmv2-base-uncased",num_labels=len(classes))
    #we will change classifier layer for this model for our 13 labels
    # self.model=LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased")
    self.num_labels = len(classes)
#######################################

#######################################



    #changing the classifier layer
    # self.model.pooler = nn.Sequential(nn.Flatten(),
    #                        nn.Linear(768,128),
    #                        nn.ReLU(),
    #                        nn.Dropout(p=self.model.config.hidden_dropout_prob),
    #                        nn.Linear(128,self.num_labels))#totel classes are 13
    self.model.to(self.device)
    print(self.model)
    #print(self.model.classifier)
    self.optimizer = AdamW(self.model.parameters(), lr = 0.00005)
    #self.scheduler = ExponentialLR(self.optimizer, gamma=0.9) #learning rate scheduler(if scheduler is used than start learning rate from big value like 0.1)
    self.total_epochs =total_epochs
    self.train_dataloader=train_dataloader
    self.val_dataloader=val_dataloader
    self.df_train=df_train
    self.df_val=df_val
    self.loss_list=[]
        

  #creating model checkpoint
  def checkpoint(self,path,model,val_acc, epoch,optimizer, global_step):
    torch.save({"step": global_step,"epoch": epoch,"model": model,"val_acc": val_acc,"optimizer": optimizer},path,)

  #loading model and its parameters
  def load_checkpoint(self,save_path):
    dict_parameters = torch.load(save_path)
    model = dict_parameters["model"]
    val_acc = dict_parameters["val_acc"]
    optimizer = dict_parameters["optimizer"]
    globel_step = dict_parameters["step"]
    epoch = dict_parameters["epoch"]
    return model, epoch,optimizer,val_acc, globel_step

    #validating model
  def validate(self,model,global_step,epoch,optimizer):
    model.eval()
    valid_loss = 0.0
    correct = 0
  
    for batch in tqdm(self.val_dataloader):
      input_ids = batch["input_ids"].squeeze(1).to(self.device)
      bbox = batch["bbox"].squeeze(1).to(self.device)
      attention_mask = batch["attention_mask"].squeeze(1).to(self.device)
      token_type_ids = batch["token_type_ids"].squeeze(1).to(self.device)
      labels = batch["labels"].squeeze(1).to(self.device)
      image=batch['image'].squeeze(1).to(self.device)

      with torch.no_grad():
        outputs = model(image=image,input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,labels=labels)
      loss = outputs.loss
      valid_loss += loss.item()
      predictions = outputs.logits.argmax(-1)
      correct += (predictions == labels).float().sum()
    

    print("Validation loss:", valid_loss / len(self.val_dataloader))
    accuracy = 100 * correct / len(self.df_val)
    print("Validation accuracy:", accuracy.item())


    writer.add_scalar("Validation loss",valid_loss / len(self.val_dataloader),epoch)
    writer.add_scalar("Valdation Accuracy",accuracy.item(),epoch)
    print("-----------")

    val_acc = accuracy.item()
    if int(epoch)==0:
      print("saving the checkpoint. . . ")
      self.checkpoint(self.checkpoint_path,model,float(val_acc), int(epoch),optimizer, int(global_step))
      print("checkpoint is saved successfully")
    else:
      _, _,_,last_acc,_ = self.load_checkpoint(self.checkpoint_path)     
      if val_acc > last_acc:
        print("saving the checkpoint. . . ")
        self.checkpoint(self.checkpoint_path,model,float(val_acc), int(epoch),optimizer, int(global_step))
        print("checkpoint is saved successfully")
  

################training model##########

    
  def train(self,model):
    optimizer=self.optimizer
    global_step = 0
    for epoch in range(self.total_epochs):
      model.train()
      print("Epoch:", epoch)
      running_loss = 0.0
      correct = 0

      for batch in tqdm(self.train_dataloader):
        input_ids = batch["input_ids"].squeeze(1).to(self.device)
        bbox = batch["bbox"].squeeze(1).to(self.device)
        attention_mask = batch["attention_mask"].squeeze(1).to(self.device)
        token_type_ids = batch["token_type_ids"].squeeze(1).to(self.device)
        labels = batch["labels"].squeeze(1).to(self.device)
        image=batch['image'].squeeze(1).to(self.device)

        #forward pass
        outputs = model(image=image,input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                  labels=labels)

        # print("***************************************************")
        # print("shape of image:".format(image.shape))
        # print("shape of input_ids:".format(input_ids.shape))
        # print("shape of bbox:".format(bbox.shape))
        # print("shape of attention_mask:".format(attention_mask.shape))
        # print("shape of token_type_ids:".format(token_type_ids.shape))
        # print("****************************************************")

        # outputs = model(image,input_ids, bbox,attention_mask,token_type_ids)
        # print("**************")
        # print(outputs)
        # print("**************")
        loss = outputs.loss

        running_loss += loss.item()
        predictions = outputs.logits.argmax(-1)
        correct += (predictions == labels).float().sum()

        # backward pass to get the gradients 
        loss.backward()

        # update
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        
        writer.add_scalar("Training batch_loss",loss,global_step)
        #learning rate scadueler for each epoch
        #self.scheduler.step()
        #lr = self.scheduler.get_lr() #getting current learning rate
      
      print("Training Loss:", running_loss / len(self.train_dataloader))
      accuracy = 100 * correct / len(self.df_train)
      print("Training accuracy:", accuracy.item())
      training_loss = running_loss / len(self.train_dataloader)
      # print("#######")
      # print("learning rate is: ", lr[0])
      # print("#######")
      #writer.add_scalar("Learning rate vrs epoch",lr[0],epoch)
      writer.add_scalar("Training Loss",training_loss,epoch)
      writer.add_scalar("training Accuracy",accuracy,epoch)
  


      self.validate(model,global_step,epoch,optimizer)#checking performance on validation data
    #writer.flush()
    #writer.close()
    return model


#function to restore training

  def restore_training(self,model, current_epoch,optimizer, global_step):
    print("resuming training from last checkpoint... . .  .")
    model.to(self.device)
    for epoch in range(current_epoch,self.total_epochs):
      model.train()
      print("Epoch:", epoch)  
      running_loss = 0.0
      correct = 0
      for batch in tqdm(self.train_dataloader):
        input_ids = batch["input_ids"].squeeze(1).to(self.device)
        bbox = batch["bbox"].squeeze(1).to(self.device)
        attention_mask = batch["attention_mask"].squeeze(1).to(self.device)
        token_type_ids = batch["token_type_ids"].squeeze(1).to(self.device)
        labels = batch["labels"].squeeze(1).to(self.device)
        image=batch['image'].squeeze(1).to(self.device)

        outputs = model(image=image,input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,labels=labels)
        loss = outputs.loss

        running_loss += loss.item()
        predictions = outputs.logits.argmax(-1)
        correct += (predictions == labels).float().sum()
        #backward pass to get the gradients 
        loss.backward()

        #update
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        writer.add_scalar("Training batch_loss",loss,global_step)
      
      #learning rate scadueler fter each epoch
      #self.scheduler.step()
      #get learning rate
      #lr = self.scheduler.get_lr() #getting learning rate

      print("Training Loss:", running_loss /len(self.train_dataloader))
      accuracy = 100 * correct / len(self.df_train)
      print("Training accuracy:", accuracy.item())
      training_loss = running_loss / len(self.train_dataloader)
      
      #writer.add_scalar("Learning rate vrs epoch",lr[0],epoch)
      writer.add_scalar("Training Loss",training_loss,epoch)
      writer.add_scalar("training Accuracy",accuracy,epoch)


      self.validate(model,global_step,epoch,optimizer)#checking performance on validation data
    #writer.flush()
    #writer.close()
    return model


      
  def run(self,restore_training=False):
    if restore_training == True:
      model, current_epoch,optimizer,_, globel_step = self.load_checkpoint(self.checkpoint_path)#_ is for avg_val_loss
      print("model and training parameters are successfully loaded")
      model = self.restore_training(model, current_epoch,optimizer, globel_step)

    else:
      model=self.model
      model = self.train(model)
    writer.flush()
    writer.close()
    return model

