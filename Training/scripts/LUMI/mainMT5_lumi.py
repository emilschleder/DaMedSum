print('start')

import transformers
from datasets import Dataset, DatasetDict
from evaluate import load
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoConfig
import torch.cuda
import torch
import random
import nltk
import numpy as np
import os
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "max_split_size_mb:1000"
nltk.download('punkt')

def main():
    """ Fine-tuning a model on a summarization task """

    """ PARAMS """
    max_input_length = 2000
    max_target_length = 548
    batch_size = 1
    learning_rate = 5e-5
    weight_decay = 0.01
    num_epochs = 11
    num_steps = 3
    saves = 10
    model_checkpoint = "google/mt5-xl"
    data_source = "summaries.csv"
    hub_model_id = f'emilstabil/{model_checkpoint.split("/")[1]}_V{random.randint(0, 100000)}'
    gradient_checkpointing = True
    
    print(f"""
          Running main.py with the following parameters: 
          max_input_length: {max_input_length} 
          max_target_length: {max_target_length} 
          learning_rate: {learning_rate}
          gradient_accumulation_steps: {num_steps}
          batch_size: {batch_size} 
          num_epochs: {num_epochs} 
          saves: {saves}
          model_checkpoint: {model_checkpoint}
          data_source: {data_source}
          hub_model_id: {hub_model_id}
          gradient_checkpointing: {gradient_checkpointing}
    """)
    
    print(transformers.__version__)

    """ Loading the dataset """
    metric = load("rouge")
    
    df = pd.read_csv(data_source, sep=',')

    ds_train_devtest = Dataset.from_pandas(df).train_test_split(test_size=0.2, seed=42)
    ds_devtest = ds_train_devtest['test'].train_test_split(test_size=0.5, seed=42)

    dataset = DatasetDict({
        'train': ds_train_devtest['train'],
        'valid': ds_devtest['train'],
        'test': ds_devtest['test']
    })

    """ Preprocessing the data """

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def preprocess_function(examples):
        inputs = ["summarize: " + str(doc) for doc in examples["text"]]
        
        model_inputs = tokenizer(
            inputs, 
            max_length=max_input_length, 
            truncation=True
        )

        # Setup the tokenizer for targets
        labels = tokenizer(
            text_target=examples["summary"], 
            max_length=max_target_length, 
            truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    print('tokenized_datasets')
    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    
    """ Fine-tuning the model """
    print('model_checkpoint')
    mt5_config = AutoConfig.from_pretrained(
        "google/mt5-large",
        max_length=max_target_length,
        length_penalty=5.0,
        no_repeat_ngram_size=3,
        num_beams=11
    )
    model = (AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, config=mt5_config))
            
    print('Seq2SeqTrainingArguments')
    model_name = model_checkpoint.split("/")[-1]
    
    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-test", #{random.randint(0, 100000)}",
        evaluation_strategy = "steps",
        learning_rate= learning_rate,
        gradient_accumulation_steps=num_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        gradient_checkpointing=gradient_checkpointing,
        save_total_limit=saves,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        push_to_hub=True,
        hub_model_id=hub_model_id
    )

    print('data_collator')
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    """ Computing metrics from predictions """
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(
            predictions, 
            skip_special_tokens=True
        )
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, 
            skip_special_tokens=True
        )

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(
            predictions=decoded_preds, 
            references=decoded_labels, 
            use_stemmer=True, 
            use_aggregator=True
        )
        # Extract a few results
        result = {key: value * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    """ Creating the Trainer """
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    """ Training the model """    
    print('empty cache')
    torch.cuda.empty_cache()

    print('start training')
    trainer.train()

    print('push to hub')
    trainer.push_to_hub()

if __name__ == '__main__':
    main()