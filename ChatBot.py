from transformers import T5Tokenizer,AutoModelForSeq2SeqLM,MT5ForConditionalGeneration
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import  ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import pprintpp as pprint
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
from torch.optim import AdamW

class QuestionGeneration():
    def __init__(self,device,data_path,max_length=256,num_output=25):
        self.model_name = 'noah-ai/mt5-base-question-generation-vi'
        self.device = device
        self.num_output = num_output
        self.max_length = max_length
        self.data_path = data_path

    def __CreateQueries__(self,text):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        input_ids = tokenizer.encode(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            beam_outputs = model.generate(
                input_ids=input_ids,
                max_length=self.max_length,
                num_beams=self.num_output,
                num_return_sequences=self.num_output,
                early_stopping=True,
            )

        l=[]
        for i in range(len(beam_outputs)):
            query = tokenizer.decode(beam_outputs[i], skip_special_tokens=True)
            l.append(query)
        return list(set(l))

    def ReadDataByExcel(self):
        self.content = pd.read_excel(self.data_path,sheet_name='Content')
        self.question = pd.read_excel(self.data_path,sheet_name='Question')

    def HandleData(self):
        self.ReadDataByExcel()
        # lay context chua tao cau hoi
        temp = self.content.where(self.content["state"]==0).dropna()
        for i in temp["id"]:
          # fix loi so thuc khi doc pandas
          i = int(i)
          # xoa cau hoi cua context cu da duoc sua neu co (state = 1 -> state = 0)
          self.question = self.question.mask(self.question["id"]==i).dropna()
          # sinh cau hoi
          q =self.__CreateQueries__(temp.iloc[i]["content"])
          d = {"id":temp.iloc[i]["id"],"question":q,"content":temp.iloc[i]["content"]}
          d = pd.DataFrame(data=d)
          self.question= pd.concat([self.question, d])
        self.content["state"]=1
        with pd.ExcelWriter(self.data_path) as writer:
          self.content.to_excel(writer, sheet_name='Content',index=None)
          self.question.to_excel(writer, sheet_name='Question',index=None)


class ChatBotDataset(Dataset):
    def __init__(self,device,data,tokenizer,max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        data_row = self.data.iloc[index]

        source_encoding = self.tokenizer.encode_plus(
            data_row["question"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_encoding = self.tokenizer.encode_plus(
            data_row["id"],
            max_length=10,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return dict(
            question=data_row["question"],
            context=data_row["content"],
            input_ids=source_encoding["input_ids"].flatten(),
            attention_mask=source_encoding["attention_mask"].flatten(),
            decoder_attention_mask = target_encoding["attention_mask"].flatten(),
            labels=labels.flatten()
        )

class ChatBotDataModel(pl.LightningDataModule):
    def __init__(self,device,model_name,train,test,batch_size=2,max_length=256):
        super().__init__()
        self.train = train
        self.test = test
        self.tokenizer  = T5Tokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device

    def setup(self,stage=None):
        self.train_dataset = ChatBotDataset(self.device ,self.train,self.tokenizer,self.max_length)
        self.test_dataset = ChatBotDataset(self.device ,self.test,self.tokenizer,self.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size)
    
class ChatBotModel(pl.LightningModule):
    def __init__(self,device,model_name):
        super().__init__()
        self.save_hyperparameters(ignore=["device", "model_name"])
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name)

    def forward(self,input_ids,attention_mask,decoder_attention_mask,labels=None):
        output = self.model(input_ids=input_ids,attention_mask=attention_mask,labels=labels,decoder_attention_mask=decoder_attention_mask)
        return output

    def training_step(self,batch,batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels=batch["labels"]
        decoder_attention_mask=batch["decoder_attention_mask"]
        outputs = self(input_ids,attention_mask,decoder_attention_mask,labels)
        self.log("train_loss",outputs.loss, prog_bar=True, logger=True)
        return outputs.loss

    def validation_step(self,batch,batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels=batch["labels"]
        decoder_attention_mask=batch["decoder_attention_mask"]
        outputs = self(input_ids,attention_mask,decoder_attention_mask,labels)
        self.log("val_loss",outputs.loss,prog_bar=True, logger=True,on_epoch=True)
        return outputs.loss

    def test_step(self,batch,batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels=batch["labels"]
        decoder_attention_mask=batch["decoder_attention_mask"]
        outputs = self(input_ids,attention_mask,decoder_attention_mask,labels)
        self.log("test_loss",outputs.loss,prog_bar=True, logger=True)
        return outputs.loss

    def configure_optimizers(self):
        return AdamW(self.parameters(),lr=3e-4)

class ChatBot():
    def __init__(self,data_path,max_length=256,model_name = 'google/mt5-base',path_logger="training-logs", path_checkpoint="Python\checkpoints",name_checkpoit="best_checkpoint"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.max_length = max_length
        self.data_path = data_path
        self.path_checkpoint = path_checkpoint
        self.name_checkpoit = name_checkpoit
        self.path_logger = path_logger
        self.trained_model = None
        self.tokenizer  = None

    def StartQuestionGeneration(self,num_output):
        questionGeneration = QuestionGeneration(self.device,self.data_path,self.max_length,num_output)
        questionGeneration.HandleData()

    def TrainingChatBot(self,loops =1,epochs=10,batch_size=2,is_retrain = False):
        questionGeneration = QuestionGeneration(self.device,self.data_path,self.max_length)
        questionGeneration.ReadDataByExcel()
        data = questionGeneration.question
        # giai phong ram
        questionGeneration = None
        torch.cuda.empty_cache()

        logger = TensorBoardLogger(self.path_logger, name="chatbot_mt5")
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.path_checkpoint,
            filename=self.name_checkpoit,
            save_top_k=1,
            verbose = True,
            monitor="val_loss",
            mode= "min",
            save_weights_only=True
            )
        trainer = pl.Trainer(
                    logger=logger,
                    callbacks=checkpoint_callback,
                    max_epochs=epochs,
                    )
        model = 0
        if is_retrain:
          model = ChatBotModel.load_from_checkpoint(self.path_checkpoint+"/"+self.name_checkpoit+".ckpt",device=self.device,model_name=self.model_name)
        else:
          model = ChatBotModel(self.device,self.model_name)
        for i in range(loops):
          print("loop",i,":")
          #tach data
          if i > 0:
            model = ChatBotModel.load_from_checkpoint(self.path_checkpoint+"/"+self.name_checkpoit+".ckpt",device=self.device,model_name=self.model_name)
          train , test = train_test_split(data,test_size=0.2)
          data_model = ChatBotDataModel(self.device,self.model_name,train,test,batch_size,self.max_length)
          data_model.setup()

          trainer.fit(model=model,datamodule=data_model)
          trainer.test(model=model,datamodule=data_model)
          # giai phong ram
          torch.cuda.empty_cache()


    def GenerateAnswer(self,question,accuracy=0.9,checkpoints="",is_print_loss =False):
        torch.cuda.empty_cache()
        #load check point
        if self.trained_model == None:
          if checkpoints == "":
            self.trained_model = ChatBotModel.load_from_checkpoint(self.path_checkpoint+"/"+self.name_checkpoit+".ckpt",device=self.device,model_name=self.model_name)
          else:
            self.trained_model = ChatBotModel.load_from_checkpoint(self.path_checkpoint+"/"+checkpoints,device=self.device,model_name=self.model_name)
          self.trained_model.eval()
          self.trained_model.freeze()
        if self.tokenizer == None:
          self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        source_encoding = self.tokenizer.encode_plus(
            question,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        questionGeneration = QuestionGeneration(self.device,self.data_path,self.max_length)
        questionGeneration.ReadDataByExcel()
        data = questionGeneration.content

        with torch.no_grad():
          generated = self.trained_model.model.generate(
              input_ids=source_encoding["input_ids"],
              attention_mask = source_encoding["attention_mask"],
              max_length=10,
              return_dict_in_generate=True,
              output_scores = True
                      )
          result = self.tokenizer.decode(generated["sequences"][0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
          target_encoding = self.tokenizer.encode_plus(
            result,
            max_length=10,
            padding="max_length",
            truncation=True,
            return_tensors="pt",).to(self.device)
          labels = target_encoding["input_ids"]
          labels[labels == self.tokenizer.pad_token_id] = -100
          loss = self.trained_model.model(input_ids=source_encoding["input_ids"],
                     attention_mask=source_encoding["attention_mask"],
                     labels=labels,
                     decoder_attention_mask=target_encoding["attention_mask"]).loss
          try:
            result=data[data["id"]==result]["content"].values[0]
          except:
            result = """!!!ERROR!!!
Xin lỗi tôi không thể trả lời câu hỏi này
Bạn có thể tham khảo ChatGPT bằng cách thêm "@#" vào câu hỏi
VD: @:Đại học Duy Tân ở đâu?"""
          if(1-loss.item()<accuracy):
            result = """Xin lỗi tôi không thể trả lời câu hỏi này
Bạn có thể tham khảo ChatGPT bằng cách thêm "@#" vào câu hỏi
VD: @:Đại học Duy Tân ở đâu?"""
          if(is_print_loss):
            result += "\n Loss: "+str(loss.item())
          return result


