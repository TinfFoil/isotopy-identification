from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (AdamW, AutoModel, AutoTokenizer, BertForSequenceClassification,
                          BertTokenizer, get_linear_schedule_with_warmup)
import gc
import os
import numpy as np
import pandas as pd
import random
import shutil
import tensorflow as tf
import torch
from tqdm.notebook import tqdm

def unimodal_ovr(cv_fold_dir, class_json, class_name):

  model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

  class_name = str(class_name).upper()

  # read the JSON files into DataFrames
  train_data = pd.read_json(os.path.join(cv_fold_dir, class_json), lines=True)
  val_data = pd.read_json(os.path.join(cv_fold_dir, 'val.json'), lines=True)
  test_data = pd.read_json(os.path.join(cv_fold_dir, 'test.json'), lines=True)

  # add the 'data_type' field based on the type of split
  train_data = train_data.assign(data_type='train')
  val_data = val_data.assign(data_type='val')
  test_data = test_data.assign(data_type='test')

  # concatenate the DataFrames into a final DataFrame
  final_data = pd.concat([train_data, val_data, test_data], ignore_index=True)
  data = final_data

  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

  # Encode training dataset using the tokenizer
  encoded_data_train = tokenizer.batch_encode_plus(
          data[data.data_type == 'train']['Segment text'].values.tolist(),
          add_special_tokens=True,
          return_attention_mask=True,  # so we know when a sentence is finished
          pad_to_max_length=True,
          truncation=True,
          max_length=512,
          return_tensors='pt'
      )

  # Encode validation dataset using the tokenizer
  encoded_data_val = tokenizer.batch_encode_plus(
          data[data.data_type == 'val']['Segment text'].values.tolist(),
          add_special_tokens=True,
          return_attention_mask=True,  
          pad_to_max_length=True,
          truncation=True,
          max_length=512,
          return_tensors='pt'
      )

  # Extract IDs, attention masks and labels from training dataset
  input_ids_train = encoded_data_train['input_ids']
  attention_masks_train = encoded_data_train['attention_mask']
  labels_train = torch.tensor(data[data.data_type == 'train'][class_name].values)
  labels_train
          
  # Extract IDs, attention masks and labels from validation dataset
  input_ids_val = encoded_data_val['input_ids']
  attention_masks_val = encoded_data_val['attention_mask']
  labels_val = torch.tensor(data[data.data_type == 'val'][class_name].values)

  # Create train and validation dataset from extracted features
  dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
  dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
  print("Train dataset length: {}\nValidation dataset length: {}".format(len(dataset_train), len(dataset_val)))

  # Define the size of each batch
  batch_size = 16

  # Load training dataset
  dataloader_train= DataLoader(
          dataset_train,
          sampler=RandomSampler(dataset_train),
          batch_size=batch_size)

  # Load valuation dataset
  dataloader_val= DataLoader(
          dataset_val,
          sampler=RandomSampler(dataset_val),
          batch_size=batch_size)

  # Define model optimizer -> Adam
  optimizer = AdamW(
          model.parameters(),
          lr = 1e-5, 
          eps=1e-8
      )
  
  # Define model scheduler
  epochs = 1
  scheduler = get_linear_schedule_with_warmup(optimizer,
                                                  num_warmup_steps=0,
                                                  num_training_steps=len(dataloader_train)*epochs)

  # Define random seeds
  seed_val = 17
  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)

  # Define processor type for torch
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  device

  # Metrics
  def f1_score_func(preds, labels):
      preds_flat = np.argmax(preds, axis=1).flatten()
      labels_flat = labels.flatten()
      return f1_score(labels_flat, preds_flat, average='binary')

  def precision_score_func(preds, labels):
      preds_flat = np.argmax(preds, axis=1).flatten()
      labels_flat = labels.flatten()
      return precision_score(labels_flat, preds_flat, average='binary')

  def recall_score_func(preds, labels):
      preds_flat = np.argmax(preds, axis=1).flatten()
      labels_flat = labels.flatten()
      return recall_score(labels_flat, preds_flat, average='binary')

      # Prints the accuracy of the model
  def accuracy_function(preds, labels):
      preds_flat=np.argmax(preds, axis=1).flatten()
      labels_flat=labels.flatten()
      return accuracy_score(labels_flat, preds_flat)# , average='binary')

  # Evaluates the model using the validation set
  def evaluate(dataloader_val):
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)
        inputs = {'input_ids': batch[0],
          'attention_mask': batch[1],
          'labels': batch[2],
            }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

  # Initialize lists for storing metrics
  train_loss_list = []
  val_loss_list = []
  val_f1_list = []
  val_precision_list = []
  val_recall_list = []
  val_acc_list = []

  # Training loop
  for epoch in tqdm(range(1, epochs + 1)):

      model.train()  # model is training

      loss_train_total = 0

      progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
      for batch in progress_bar:
          model.zero_grad()
          batch = tuple(b.to(device) for b in batch)
          inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'labels': batch[2]}
              
          outputs = model(**inputs)

          loss = outputs[0]
          loss_train_total += loss.item()
          loss.backward()  # to backpropagate

          torch.nn.utils.clip_grad_norm_(model.parameters(),
                                            1.0)  # prevents the gradient from being too small or too big

          optimizer.step()
          scheduler.step()
          progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

      tqdm.write(f'\nEpoch {epoch}/{epochs}')

      loss_train_avg = loss_train_total / len(dataloader_train)
      tqdm.write(f'Training loss: {loss_train_avg}')  # make sure that model is still training

      val_loss, predictions, true_vals = evaluate(dataloader_val)  # to check overtraining (or overfitting)
      val_f1 = f1_score_func(predictions, true_vals)
      val_precision = precision_score_func(predictions, true_vals)
      val_recall = recall_score_func(predictions, true_vals)
      val_acc = accuracy_function(predictions, true_vals)
      tqdm.write(f'Validation loss: {val_loss}')
      tqdm.write(f'F1 Score: {val_f1}')
      tqdm.write(f'Precision Score: {val_precision}')
      tqdm.write(f'Recall Score: {val_recall}')
      tqdm.write(f'Accuracy Score: {val_acc}')

      # Append the values to the respective lists
      train_loss_list.append(loss_train_avg)
      val_loss_list.append(val_loss)
      val_f1_list.append(val_f1)
      val_precision_list.append(val_precision)
      val_recall_list.append(val_recall)
      val_acc_list.append(val_acc)

  # Create a DataFrame from the lists
  df_metrics = pd.DataFrame({
          'epoch': range(1, len(train_loss_list) + 1),
          'train_loss': train_loss_list,
          'val_loss': val_loss_list,
          'val_f1': val_f1_list,
          'val_precision': val_precision_list,
          'val_recall': val_recall_list,
          'val_acc':val_acc_list
      })

  # Export the metrics
  file_name_metrics = 'unimodal_ovr_' + 'PP' + '_metrics.xlsx'
  
  # Renaming test_data
  data_test = test_data

  # Encode validation dataset using the tokenizer
  encoded_data_test = tokenizer.batch_encode_plus(
          data_test['Segment text'].values.tolist(),
          add_special_tokens=True,
          return_attention_mask=True,  
          pad_to_max_length=True,
          truncation=True,
          max_length=512,
          return_tensors='pt'
      )

  # Extract IDs, attention masks, and labels from validation dataset
  input_ids_test = encoded_data_test['input_ids']
  attention_masks_test = encoded_data_test['attention_mask']

  dataset_test = TensorDataset(input_ids_test, attention_masks_test)
  print("Test dataset length: {}".format(len(dataset_test)))

  dataloader_test = DataLoader(dataset_test)

  # Evaluates the model using the test set
  def predict(dataset_test):
      predictions = []
      probabilities = []

      for row in dataset_test:
          row = tuple(r.to(device) for r in row)
          inputs = {'input_ids': row[0],
                    'attention_mask': row[1]
                    }

          with torch.no_grad():
              outputs = model(**inputs)

          logits = outputs[0]
          logits = logits.detach().cpu().numpy()
          predictions.append(logits)
          probabilities.append(np.exp(logits) / np.sum(np.exp(logits), axis=1))

      return predictions, probabilities

  # Predict values for test dataset
  predictions, probabilities = predict(dataloader_test)

  results = []
  for i, prediction in enumerate(predictions):
      predicted = np.argmax(prediction, axis=1)[0]
      results.append(predicted)

  # Flatten the probabilities list to remove the extra dimension
  probabilities_flat = [item[0] for item in probabilities]

  # Assuming your list of predicted labels is called 'predicted_labels'
  predicted_labels = results

  # Extract the true labels from the 'data_test' DataFrame
  true_labels = test_data[class_name].tolist()

  # Create DataFrames for texts, true labels, predicted labels, and probabilities
  df_texts = pd.DataFrame({'text': data_test['Segment text'].values.tolist()})
  df_true_labels = pd.DataFrame({'true_labels': true_labels})
  df_preds = pd.DataFrame({'label': predicted_labels})
  df_probs = pd.DataFrame(probabilities_flat, columns=[f'prob_class_{i}' for i in range(probabilities_flat[0].shape[-1])])

  # Combine DataFrames and save to Excel file
  df_final = pd.concat([df_texts, df_true_labels, df_preds, df_probs], axis=1)
  df_final['prediction_status'] = np.where(df_final['true_labels'] == df_final['label'], 'correct', 'wrong')
  df_final = df_final.rename(columns={'prob_class_1': f'proba_{class_name}'})

  # Generate the classification report
  report = classification_report(true_labels, predicted_labels) #, labels=[0, 1, 2], target_names=['Class 0', 'Class 1', 'Class 2'])

  # Save the output files in the cv_fold_dir
  name_metrics = 'unimodal_ovr_' + class_name + '_metrics.json'
  name_final = 'unimodal_ovr_' + class_name + '_predictions.xlsx'
  name_report = 'unimodal_ovr_' + class_name + '_classification_report.txt'
    

  df_metrics.to_json(os.path.join(cv_fold_dir, name_metrics), orient='records')
  df_final.to_excel(os.path.join(cv_fold_dir, name_final), index=False, float_format='%.3f')
  
  with open(os.path.join(cv_fold_dir, name_report), 'w') as file:
      file.write(report)
    
  return model
  
 def load_true_labels(cv_fold_path, file_name):
    file_path = os.path.join(cv_fold_path, file_name)
    data = pd.read_json(file_path, lines=True)
    true_labels = data['label'].tolist()
    return true_labels

def loop_unimodal_ovr(folds_dir):

    class_files = ['pp_train.json', 'sp_train.json', 'mc_train.json']  

    for i in range(1, 11):
    
        cv_fold_path = folds_dir

        for class_file in class_files:
            class_name = class_file[:2].upper()
            unimodal_multiclass(cv_fold_path, f'{cv_fold_path}/{class_file}', class_name)

        # Read the .xlsx files
        pp_preds = pd.read_excel(f'{cv_fold_path}/unimodal_ovr_PP_predictions.xlsx')
        sp_preds = pd.read_excel(f'{cv_fold_path}/unimodal_ovr_SP_predictions.xlsx')
        mc_preds = pd.read_excel(f'{cv_fold_path}/unimodal_ovr_MC_predictions.xlsx')

        # Merge the DataFrames based on the 'text' column
        all_preds = pp_preds.merge(sp_preds[['text', 'proba_SP']], on='text')
        all_preds = all_preds.merge(mc_preds[['text', 'proba_MC']], on='text')

        # Create a new column 'pred' based on the maximum probability value
        all_preds['pred'] = all_preds[['proba_PP', 'proba_SP', 'proba_MC']].idxmax(axis=1)
        all_preds['pred'] = all_preds['pred'].map({'proba_PP': 0, 'proba_SP': 1, 'proba_MC': 2})

        # Reorder the columns
        all_preds = all_preds[['text', 'pred', 'true_labels', 'proba_PP', 'proba_SP', 'proba_MC']]

        true = load_true_labels(cv_fold_path, 'test.json')
        predicted = all_preds['pred'].tolist()
        report = classification_report(true, predicted, labels=[0, 1, 2], target_names=['Class 0', 'Class 1', 'Class 2'])
        file_name_report = 'txt_ovr_final_classification_report.txt'

        with open(os.path.join(cv_fold_path, file_name_report), 'w') as file:
            file.write(report)

        # Save the final predictions to an .xlsx file
        all_preds.to_excel(f'{cv_fold_path}/txt_ovr_final_predictions.xlsx', index=False, float_format='%.3f')
 
loop_unimodal_ovr('cv_fold_path/cv_fold_{i}')
