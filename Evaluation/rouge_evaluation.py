import pandas as pd
from transformers import pipeline
from evaluate import load

# Define your models
models = ["emilstabil/DaMedSum-small", 
          "emilstabil/DaMedSum-base", 
          "emilstabil/DaMedSum-large", 
          "emilstabil/DaMedSumT5",
          "Danish-summarisation/DanSumT5-small", 
          "Danish-summarisation/DanSumT5-base", 
          "Danish-summarisation/DanSumT5-large"]

# Load the CSV file
df = pd.read_csv('evaluation/data/rouge_summaries_test.csv')

# Initialize the Rouge metric
rouge = load("rouge")

# Function to generate summary using the pipeline
def generate_summary(pipe, text):
    return pipe(text, max_length=548, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)[0]['summary_text']

# Evaluate each model
model_results = {}

for model_name in models:
    # Initialize the pipeline for each model
    pipe = pipeline("summarization", model=model_name)
    
    # Initialize the dictionary for the evaluations
    evaluations = {}

    # Generate summaries for each text in the dataset
    for index, row in df.iterrows():
        print(f"Generating summary for {model_name} - text{index+1}...")
        
        # Generate the summary
        text, true_summary = row['text'], row['summary']
        generated_summary = generate_summary(pipe, text)
        scores = rouge.compute(predictions=[generated_summary], references=[true_summary])
        
        # Print the results and status
        print(f"Generated summary for text{index+1}... = {generated_summary}")
        print(f"Computing ROUGE-scores for {model_name} - text{index+1}")
        
        # Save the scores and the generated summary
        evaluations[f"text{index+1}"] = {
            "rouge1":    scores["rouge1"] * 100,
            "rouge2":    scores["rouge2"] * 100, 
            "rougeL":    scores["rougeL"] * 100, 
            "rougeLsum": scores["rougeLsum"] * 100, 
            "summary":   generated_summary,
            "text_length": len(text),
            "summary_length": len(generated_summary),
        }

    # Save the evaluations for the model
    model_results[model_name] = {
        "evaluations": evaluations
    }

# Convert the results to a format suitable for a DataFrame
data_for_df = []
for model, results in model_results.items():
    for text_id, scores in results['evaluations'].items():
        data_for_df.append({
            'Model': model,
            'Text_ID': text_id,
            'ROUGE-1': scores['rouge1'],
            'ROUGE-2': scores['rouge2'],
            'ROUGE-L': scores['rougeL'],
            'ROUGE-Lsum': scores['rougeLsum'],
            'Summary': scores['summary'],
            'Text_length': scores['text_length'], 
            'Summary_length': scores['summary_length']
        })

# Create a DataFrame from the list of dictionaries
df_results = pd.DataFrame(data_for_df)
df_means = df_results.groupby('Model')[['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-Lsum']].mean()

# Print the results
print(df_means)
print('=========================')
print(df_results)

# # Save to CSV
df_results.to_csv('evaluation/results/all/model_evaluation_results_all_scores.csv', index=False)
df_means.to_csv('evaluation/results/mean/model_evaluation_results_mean_scores.csv', index=False)