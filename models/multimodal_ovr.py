from collections import Counter
import random
import numpy as np
import pandas as pd
import shutil
import zipfile
import json
import gc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from tensorflow import keras
from keras import losses

from madgrad import MADGRAD

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report

import clip
import pickle

from tensorflow import keras
from keras import losses

from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

import copy

import transformers

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    MMBTConfig,
    MMBTModel,
    MMBTForClassification,
    get_linear_schedule_with_warmup,
)

def multimodal_ovr(cv_fold_dir, class_json, class_name):

  # class_json -> like pp_train.json
  # class_name -> string like "PP", capitalized

  class_name = str(class_name).upper()

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # read the JSON files into DataFrames
  data_train = pd.read_json(os.path.join(cv_fold_dir, class_json), lines=True)
  data_val = pd.read_json(os.path.join(cv_fold_dir, 'val.json'), lines=True)
  data_test = pd.read_json(os.path.join(cv_fold_dir, 'test.json'), lines=True)
	 
  # Load CLIP model and needed preprocessing
  clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)

  # Freeze weights of CLIP feature encoder, as we will not finetune it.
  for p in clip_model.parameters():
    p.requires_grad = False

  # nitialize needed variables
  num_image_embeds = 4
  num_labels = 1
  gradient_accumulation_steps = 20
  data_dir = cv_fold_dir
  max_seq_length = 512
  max_grad_norm = 0.5
  train_batch_size = 8
  eval_batch_size = 8
  image_encoder_size = 288
  image_features_size = 640
  num_train_epochs = 3

  #Create a function that will prepare an image for CLIP encoder in a special manner. 
  #This function will split image into three tiles (by height or width, depending on the aspect ratio of the image). 
  #Finally we will get four vectors after encoding (one vector for each tile and one vector for whole image that was padded to square).
  def slice_image(im, desired_size):
        '''
        Resize and slice image
        '''
        old_size = im.size  

        ratio = float(desired_size)/min(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        im = im.resize(new_size, Image.ANTIALIAS)
        
        ar = np.array(im)
        images = []
        if ar.shape[0] < ar.shape[1]:
            middle = ar.shape[1] // 2
            half = desired_size // 2
            
            images.append(Image.fromarray(ar[:, :desired_size]))
            images.append(Image.fromarray(ar[:, middle-half:middle+half]))
            images.append(Image.fromarray(ar[:, ar.shape[1]-desired_size:ar.shape[1]]))
        else:
            middle = ar.shape[0] // 2
            half = desired_size // 2
            
            images.append(Image.fromarray(ar[:desired_size, :]))
            images.append(Image.fromarray(ar[middle-half:middle+half, :]))
            images.append(Image.fromarray(ar[ar.shape[0]-desired_size:ar.shape[0], :]))

        return images

  def resize_pad_image(im, desired_size):
        '''
        Resize and pad image to a desired size
        '''
        old_size = im.size  

        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])

        im = im.resize(new_size, Image.ANTIALIAS)

        # create a new image and paste the resized on it
        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(im, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))

        return new_im

  #Define a function, that will get image features from CLIP.
  class ClipEncoderMulti(nn.Module):
        def __init__(self, num_embeds, num_features=image_features_size):
            super().__init__()        
            self.model = clip_model
            self.num_embeds = num_embeds
            self.num_features = num_features

        def forward(self, x):
            # 4x3x288x288 -> 1x4x640
            out = self.model.encode_image(x.view(-1,3,288,288))
            out = out.view(-1, self.num_embeds, self.num_features).float()
            return out  # Bx4x640
    
  class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, max_seq_length):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = torch.LongTensor(self.tokenizer.encode(self.data[index]["Segment text"], add_special_tokens=True))
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        sentence = sentence[:self.max_seq_length]

        label = torch.FloatTensor([self.data[index][class_name]])

        image = Image.open(os.path.join('/dataset_path/dataset', self.data[index]["img_name"])).convert("RGB")
        sliced_images = slice_image(image, 288)
        sliced_images = [np.array(self.transforms(im)) for im in sliced_images]
        image = resize_pad_image(image, image_encoder_size)
        image = np.array(self.transforms(image))
        
        sliced_images = [image] + sliced_images         
        sliced_images = torch.from_numpy(np.array(sliced_images)).to(device)

        return {
            "image_start_token": start_token,            
            "image_end_token": end_token,
            "sentence": sentence,
            "image": sliced_images,
            "label": label            
        }

    def get_label_frequencies(self):
        label_freqs = Counter()
        for row in self.data:
            label_freqs.update([row["label"]])
        return label_freqs
    
    def get_labels(self):
        labels = []
        for row in self.data:
            labels.append(row["label"])
        return labels
     
  def collate_fn(batch):
    lens = [len(row["sentence"]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = input_row["sentence"]
        mask_tensor[i_batch, :length] = 1
    
    img_tensor = torch.stack([row["image"] for row in batch])
    tgt_tensor = torch.stack([row["label"] for row in batch])
    img_start_token = torch.stack([row["image_start_token"] for row in batch])
    img_end_token = torch.stack([row["image_end_token"] for row in batch])

    return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, tgt_tensor
    
  def load_examples(tokenizer, evaluate=False):
    path = os.path.join(data_dir, "val.json" if evaluate else class_json)
    transforms = preprocess
    dataset = JsonlDataset(path, tokenizer, transforms, max_seq_length - num_image_embeds - 2)
    return dataset
    
  model_name = 'bert-base-uncased'
  transformer_config = AutoConfig.from_pretrained(model_name) 
  transformer = AutoModel.from_pretrained(model_name, config=transformer_config)
  img_encoder = ClipEncoderMulti(num_image_embeds)
  tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
  config = MMBTConfig(transformer_config, num_labels=num_labels, modal_hidden_size=image_features_size)
  model = MMBTForClassification(config, transformer, img_encoder)

  model.to(device);

  train_dataset = load_examples(tokenizer, evaluate=False)
  eval_dataset = load_examples(tokenizer, evaluate=True)  

  train_sampler = RandomSampler(train_dataset)

  eval_sampler = SequentialSampler(eval_dataset)

  train_dataloader = DataLoader(
          train_dataset,
          sampler=train_sampler,
          batch_size=train_batch_size,
          collate_fn=collate_fn
      )

  eval_dataloader = DataLoader(
          eval_dataset, 
          sampler=eval_sampler, 
          batch_size=eval_batch_size, 
          collate_fn=collate_fn
      )
     
  # Prepare optimizer and schedule (linear warmup and decay)
  no_decay = ["bias", 
              "LayerNorm.weight"
            ]
  weight_decay = 0.0005

  optimizer_grouped_parameters = [
          {
              "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
              "weight_decay": weight_decay,
          },
          {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
      ]

  t_total = (len(train_dataloader) // gradient_accumulation_steps) * num_train_epochs
  warmup_steps = t_total // 10

  optimizer = MADGRAD(optimizer_grouped_parameters, lr=2e-4)

  scheduler = get_linear_schedule_with_warmup(
          optimizer, warmup_steps, t_total
      )

  criterion = nn.BCEWithLogitsLoss()
  
  def f1_score_func(preds, labels):
      preds_flat = preds.flatten()
      labels_flat = labels.flatten()
      return f1_score(labels_flat, preds_flat, average='binary')

  def accuracy_function(preds, labels):
      preds_flat = preds.flatten()
      labels_flat = labels.flatten()
      return accuracy_score(labels_flat, preds_flat, average='binary')

  def precision_function(preds, labels):
      preds_flat = preds.flatten()
      labels_flat = labels.flatten()
      return precision_score(labels_flat, preds_flat, average='binary')

  def recall_function(preds, labels):
      preds_flat = preds.flatten()
      labels_flat = labels.flatten()
      return recall_score(labels_flat, preds_flat)

  def evaluate(model, tokenizer, criterion, dataloader, tres = 0.5): 
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    proba = None
    out_label_ids = None
    for batch in dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            labels = batch[5]
            inputs = {
                "input_ids": batch[0],
                "input_modal": batch[2],
                "attention_mask": batch[1],
                "modal_start_tokens": batch[3],
                "modal_end_tokens": batch[4],
                "return_dict": False
            }
            outputs = model(**inputs)
            logits = outputs[0]  # model outputs are always tuple in transformers (see doc)
            tmp_eval_loss = criterion(logits, labels)
            
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = torch.sigmoid(logits).detach().cpu().numpy() > tres
            proba = torch.sigmoid(logits).detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:            
            preds = np.append(preds, torch.sigmoid(logits).detach().cpu().numpy() > tres, axis=0)
            proba = np.append(proba, torch.sigmoid(logits).detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
    
    eval_loss = eval_loss / nb_eval_steps

    result = {
        "loss": eval_loss,
        "accuracy": accuracy_function(out_label_ids, preds),
        "AUC": roc_auc_score(out_label_ids, proba),
        "binary_f1": f1_score_func(out_label_ids, preds),
        "precision": precision_function(out_label_ids, preds),
        "recall": recall_function(out_label_ids, preds),
        "prediction": preds,
        "labels": out_label_ids,
        "proba": proba
    }
    
    return result

  optimizer_step = 0
  global_step = 0
  train_step = 0
  tr_loss, logging_loss = 0.0, 0.0
  best_valid_auc = 0.75
  best_valid_f1 = 0.70
  global_steps_list = []
  train_loss_list = []
  val_loss_list = []
  val_acc_list = []
  val_auc_list = []
  val_f1_list = []
  val_precision_list = []
  val_recall_list = []
  running_loss = 0

  model.zero_grad()

  for i in range(num_train_epochs):
      print("Epoch", i+1, f"from {num_train_epochs}")
      whole_y_pred=np.array([])
      whole_y_t=np.array([])

      for step, batch in enumerate(tqdm(train_dataloader)):
          model.train()
          batch = tuple(t.to(device) for t in batch)
          labels = batch[5]
          inputs = {
              "input_ids": batch[0],
              "input_modal": batch[2],
              "attention_mask": batch[1],
              "modal_start_tokens": batch[3],
              "modal_end_tokens": batch[4],
              "return_dict": False
          }
          outputs = model(**inputs)
          logits = outputs[0]  # model outputs are always tuple in transformers (see doc)
          loss = criterion(logits, labels)        
          
          if gradient_accumulation_steps > 1:
              loss = loss / gradient_accumulation_steps
              
          loss.backward()
          
          tr_loss += loss.item()
          running_loss += loss.item()
          global_step += 1
          
          if (step + 1) % gradient_accumulation_steps == 0:
              torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
              optimizer.step()
              scheduler.step()  # Update learning rate schedule         
              
              optimizer_step += 1
              optimizer.zero_grad()   
                          
      average_train_loss = running_loss / len(train_dataloader)
      train_loss_list.append(average_train_loss)
      global_steps_list.append(global_step)
      running_loss = 0.0  

      val_result = evaluate(model, tokenizer, criterion, eval_dataloader)

      val_loss_list.append(val_result['loss'])
      val_acc_list.append(val_result['accuracy'])
      val_auc_list.append(val_result['AUC'])
      val_f1_list.append(val_result['binary_f1'])
      val_precision_list.append(val_result['precision'])
      val_recall_list.append(val_result['recall'])
             
      print("Train loss:", f"{average_train_loss:.4f}", 
            "Val loss:", f"{val_result['loss']:.4f}",
            "Val acc:", f"{val_result['accuracy']:.4f}",
            "F1 score:", f"{val_result['binary_f1']:.4f}")
      
      print('\n')
    
  # Create a DataFrame from the lists
  df_metrics = pd.DataFrame({
        'epoch': range(1, len(train_loss_list) + 1),
        'train_loss': train_loss_list,
        'val_loss': val_loss_list,
        'val_acc': val_acc_list,
        'val_f1': val_f1_list,
        'val_precision': val_precision_list,
        'val_recall': val_recall_list
    })

  # Export the DataFrame to an Excel file
  file_name_metrics = 'multimodal_ovr_' + class_name + '_metrics.xlsx'
  df_metrics.to_excel(os.path.join(cv_fold_dir, file_name_metrics), index=False)
    
  num_labels = 1
  data_dir = cv_fold_dir
  test_batch_size = 8
     
  class TestJsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, max_seq_length):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = torch.LongTensor(self.tokenizer.encode(self.data[index]["Segment text"], add_special_tokens=True))
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        sentence = sentence[:self.max_seq_length]

        image = Image.open(os.path.join('/dataset_path/dataset', self.data[index]["img_name"])).convert("RGB")
        sliced_images = slice_image(image, 288)
        sliced_images = [np.array(self.transforms(im)) for im in sliced_images]
        image = resize_pad_image(image, image_encoder_size)
        image = np.array(self.transforms(image))        
        sliced_images = [image] + sliced_images        
        sliced_images = torch.from_numpy(np.array(sliced_images)).to(device)

        return {
            "image_start_token": start_token,            
            "image_end_token": end_token,
            "sentence": sentence,
            "image": sliced_images,
            "text": self.data[index]["Segment text"],
        }

  def final_collate_fn(batch):
      lens = [len(row["sentence"]) for row in batch]
      bsz, max_seq_len = len(batch), max(lens)

      mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
      text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

      for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
          text_tensor[i_batch, :length] = input_row["sentence"]
          mask_tensor[i_batch, :length] = 1

      img_tensor = torch.stack([row["image"] for row in batch])
      img_start_token = torch.stack([row["image_start_token"] for row in batch])
      img_end_token = torch.stack([row["image_end_token"] for row in batch])
      texts = [row["text"] for row in batch]

      return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, texts

  def load_test_examples(test_file="test.json"):
      path = os.path.join(data_dir, test_file)
      dataset = TestJsonlDataset(path, tokenizer, preprocess, max_seq_length - num_image_embeds - 2)
      return dataset

  def final_prediction(model, dataloader): 
      preds = None
      proba = None
      all_texts = []  # Initialize a list to store all texts
      for batch in tqdm(dataloader):
          model.eval()
          texts = batch[5]  # Extract the texts from the batch
          batch = tuple(t.to(device) for t in batch[:5])  # Only apply .to(device) to the tensors, not the texts
          with torch.no_grad():
              inputs = {
                  "input_ids": batch[0],
                  "input_modal": batch[2],
                  "attention_mask": batch[1],
                  "modal_start_tokens": batch[3],
                  "modal_end_tokens": batch[4],
                  "return_dict": False
              }
              outputs = model(**inputs)
              logits = outputs[0]
          if preds is None:
              preds = torch.sigmoid(logits).detach().cpu().numpy() > 0.5
              proba = torch.sigmoid(logits).detach().cpu().numpy()
              all_texts = texts  # Store texts for the first batch
          else:  
              preds = np.append(preds, torch.sigmoid(logits).detach().cpu().numpy() > 0.5, axis=0)
              proba = np.append(proba, torch.sigmoid(logits).detach().cpu().numpy(), axis=0)
              all_texts.extend(texts)  # Append texts for subsequent batches

      result = {
          "preds": preds,
          "probs": proba,
          "texts": all_texts,  # Add this line to include texts in the results
      }

      return result
    
  final_test = load_test_examples()
  final_test_sampler = SequentialSampler(final_test)
  final_test_dataloader = DataLoader(
              final_test, 
              sampler=final_test_sampler, 
              batch_size=test_batch_size, 
              collate_fn=final_collate_fn
          )
     
  results = final_prediction(model, final_test_dataloader)

  results['preds'] = results['preds'].reshape(-1)
  results['probs'] = results['probs'].reshape(-1)

  true_labels = data_test['label'].tolist()
  df_true_labels = pd.DataFrame({'true_labels': true_labels})

  df = pd.DataFrame(results)
  df = df[['texts', 'probs', 'preds']]  # Add 'texts' column here
  name_proba_column = 'proba_' + class_name
  df.columns = ['text', name_proba_column, 'label']  # Rename columns

  df.label = df.label.astype(int)

  # Merge df_true_labels with df
  df = pd.concat([df, df_true_labels], axis=1)  # Add this line

  #df['prediction_status'] = np.where(df['true_labels'] == df['label'], 'correct', 'wrong')

  file_name_preds = 'multimodal_ovr_' + class_name + '_predictions.xlsx'
  df.to_excel(os.path.join(cv_fold_dir, file_name_preds), index=False, float_format='%.3f')

  #misclassified_counts = df[df['prediction_status'] == 'wrong'].groupby(['true_labels', 'label']).size().reset_index(name='count')

  #file_name_misclassified = 'multimodal_ovr_' + class_name + '_misclassifications.xlsx'
  #misclassified_counts.to_excel(os.path.join(cv_fold_dir, file_name_misclassified), index=False)

  # Assuming your list of predicted labels is called 'predicted_labels'
  predicted_labels = df['label'].tolist()

  # Extract the true labels from the 'data_test' DataFrame
  true_labels = df['true_labels'].tolist()

  # Generate the classification report
  # report = classification_report(true_labels, predicted_labels, labels=[0, 1, 2], target_names=['Class 0', 'Class 1', 'Class 2'])

  # file_name_report = 'multimodal_ovr_' + class_name + '_classification_report.txt'
  
  #with open(os.path.join(cv_fold_dir, file_name_report), 'w') as file:
      #file.write(report)
      
  # Free up GPU RAM
  del model, transformer, tokenizer, config, img_encoder, clip_model, preprocess
  torch.cuda.empty_cache()
  gc.collect()
 
 def loop_multimodal_ovr():
  
    class_files = ['pp_train.json', 'sp_train.json', 'mc_train.json']

    for i in range(1, 11):
        cv_fold_path = f'/cv_fold_path/cv_fold_{i}'

        for class_file in class_files:
            class_name = class_file[:2].upper()
            multimodal_ovrr(cv_fold_path, f'{cv_fold_path}/{class_file}', class_name)

        # Read the .xlsx files
        pp_preds = pd.read_excel(f'{cv_fold_path}/multimodal_ovr_PP_predictions.xlsx')
        sp_preds = pd.read_excel(f'{cv_fold_path}/multimodal_ovr_SP_predictions.xlsx')
        mc_preds = pd.read_excel(f'{cv_fold_path}/multimodal_ovr_MC_predictions.xlsx')

        # Merge the DataFrames based on the 'text' column
        all_preds = pp_preds.merge(sp_preds[['text', 'proba_SP']], on='text')
        all_preds = all_preds.merge(mc_preds[['text', 'proba_MC']], on='text')

        # Create a new column 'pred' based on the maximum probability value
        all_preds['pred'] = all_preds[['proba_PP', 'proba_SP', 'proba_MC']].idxmax(axis=1)
        all_preds['pred'] = all_preds['pred'].map({'proba_PP': 0, 'proba_SP': 1, 'proba_MC': 2})

        # Reorder the columns
        all_preds = all_preds[['text', 'pred', 'true_labels', 'proba_PP', 'proba_SP', 'proba_MC']]

        true = all_preds['true_labels'].tolist()
        predicted = all_preds['pred'].tolist()
        report = classification_report(true, predicted, labels=[0, 1, 2], target_names=['Class 0', 'Class 1', 'Class 2'])
        file_name_report = 'multimodal_ovr_final_classification_report.txt'

        with open(os.path.join(cv_fold_path, file_name_report), 'w') as file:
          file.write(report)

        # Save the final predictions to an .xlsx file
        all_preds.to_excel(f'{cv_fold_path}/multimodal_ovr_final_predictions.xlsx', index=False, float_format='%.3f')

loop_multimodal_ovr()
