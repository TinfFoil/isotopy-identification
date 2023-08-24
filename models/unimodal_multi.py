from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
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
import gc
from tqdm.notebook import tqdm

def unimodal_multiclass(cv_fold_dir):

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    label_dict = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}

    # read the JSON files into DataFrames
    train_data = pd.read_json(os.path.join(cv_fold_dir, 'train.json'), lines=True)
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
    labels_train = torch.tensor(data[data.data_type == 'train'].label.values)
    labels_train
        
    # Extract IDs, attention masks and labels from validation dataset
    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(data[data.data_type == 'val'].label.values)

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
    epochs = 3
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

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Metrics
    def f1_score_func(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='macro')

    def precision_score_func(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return precision_score(labels_flat, preds_flat, average='macro')

    def recall_score_func(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return recall_score(labels_flat, preds_flat, average='macro')

    def accuracy_per_class(preds, labels):
        label_dict_inverse = {v: k for k, v in label_dict.items()}
        
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        
        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat==label]
            y_true = labels_flat[labels_flat==label]
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy:{len(y_preds[y_preds==label])}/{len(y_true)}\n')

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

        #torch.save(model, path)
        tqdm.write(f'\nEpoch {epoch}/{epochs}')

        loss_train_avg = loss_train_total / len(dataloader_train)
        tqdm.write(f'Training loss: {loss_train_avg}')  # make sure that model is still training

        val_loss, predictions, true_vals = evaluate(dataloader_val)  # to check overtraining (or overfitting)
        val_f1 = f1_score_func(predictions, true_vals)
        val_precision = precision_score_func(predictions, true_vals)
        val_recall = recall_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1 Score (macro): {val_f1}')
        tqdm.write(f'Precision Score (macro): {val_precision}')
        tqdm.write(f'Recall Score (macro): {val_recall}')

        # Append the values to the respective lists
        train_loss_list.append(loss_train_avg)
        val_loss_list.append(val_loss)
        val_f1_list.append(val_f1)
        val_precision_list.append(val_precision)
        val_recall_list.append(val_recall)

    # Create a DataFrame from the lists
    df_metrics = pd.DataFrame({
        'epoch': range(1, len(train_loss_list) + 1),
        'train_loss': train_loss_list,
        'val_loss': val_loss_list,
        'val_f1': val_f1_list,
        'val_precision': val_precision_list,
        'val_recall': val_recall_list
    })

    # Export the metrics
    df_metrics_json = df_metrics.to_json(orient='records')
    with open(os.path.join(os.getcwd(), 'df_metrics.json'), 'w') as file:
        file.write(df_metrics_json)

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
    true_labels = test_data['label'].tolist()

    # Create DataFrames for texts, true labels, predicted labels, and probabilities
    df_texts = pd.DataFrame({'text': data_test['Segment text'].values.tolist()})
    df_true_labels = pd.DataFrame({'true_labels': true_labels})
    df_preds = pd.DataFrame({'label': predicted_labels})
    df_probs = pd.DataFrame(probabilities_flat, columns=[f'prob_class_{i}' for i in range(probabilities_flat[0].shape[-1])])

    # Combine DataFrames and save to Excel file
    df_final = pd.concat([df_texts, df_true_labels, df_preds, df_probs], axis=1)
    df_final['prediction_status'] = np.where(df_final['true_labels'] == df_final['label'], 'correct', 'wrong')
    #df_final.to_excel('predictions_analysis.xlsx', index=False, float_format='%.3f')

    misclassified_counts = df_final[df_final['prediction_status'] == 'wrong'].groupby(['true_labels', 'label']).size().reset_index(name='count')

    # Generate the classification report
    report = classification_report(true_labels, predicted_labels, labels=[0, 1, 2], target_names=['Class 0', 'Class 1', 'Class 2'])

    # Save the output files in the cv_fold_dir
    df_metrics.to_json(os.path.join(cv_fold_dir, 'unimodal_multiclass_metrics.json'), orient='records')
    df_final.to_excel(os.path.join(cv_fold_dir, 'unimodal_multiclass_predictions.xlsx'), index=False, float_format='%.3f')
    misclassified_counts.to_excel(os.path.join(cv_fold_dir, 'unimodal_multiclass_misclassifications.xlsx'), index=False)
    with open(os.path.join(cv_fold_dir, 'unimodal_multiclass_classification_report.txt'), 'w') as file:
        file.write(report)
    
    return model
 
# Loop through cv_fold_{i} directories
for i in range(1, 11):

    cv_fold_path = f'cv_fold_path/cv_fold_{i}'
    
    # Create the model
    unimodal_multiclass(cv_fold_path)

    # Free up GPU RAM
    torch.cuda.empty_cache()
