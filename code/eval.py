import pandas as pd
import torch
import ast
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

# Load the saved model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./trained_roberta_multilabel_early")
tokenizer = AutoTokenizer.from_pretrained("./trained_roberta_multilabel_early")

# Load the test dataset
test_df = pd.read_csv("data/test.csv")  # Replace with your actual file path

# Convert the 'label_vector' from string representation to an actual list
test_df['label_vector'] = test_df['label_vector'].apply(ast.literal_eval)

# Prepare lists to store results
actual_labels = []
predicted_labels = []
texts = []

# Disable gradient computation for inference
model.eval()
with torch.no_grad():
    for idx, row in test_df.iterrows():
        text = row['text']
        label_vector = row['label_vector']

        # Tokenize the input text
        inputs = tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

        # Get the model's prediction (logit)
        outputs = model(**inputs)
        logits = outputs.logits

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits).squeeze().numpy()

        # Convert probabilities to binary predictions (threshold: 0.5)
        pred_labels = (probs > 0.5).astype(int)

        # Append the actual and predicted labels, along with the text
        actual_labels.append(label_vector)
        predicted_labels.append(pred_labels.tolist())  # Convert numpy array to list
        texts.append(text)

# Save the predicted labels in a new CSV file
results_df = pd.DataFrame({
    'text': texts,
    'label_vector': actual_labels,  # Original label vectors
    'preds': predicted_labels  # Predicted label vectors
})

results_df.to_csv("test_results_with_preds_test_new.csv", index=False)

# Convert lists to arrays for metric calculation
actual_labels = np.array(actual_labels)
predicted_labels = np.array(predicted_labels)

# Calculate precision, recall, and F1-score
precision, recall, f1, _ = precision_recall_fscore_support(actual_labels, predicted_labels, average='micro')
acc = accuracy_score(actual_labels, predicted_labels)

# Print the results
print(f"Test Accuracy on test Data: {acc:.4f}")
print(f"Test Precision on test Data: {precision:.4f}")
print(f"Test Recall on test Data: {recall:.4f}")
print(f"Test F1 Score on test Data:  {f1:.4f}")

# Notify where the CSV has been saved
print("Predicted labels saved in 'test_results_with_preds_test.csv'.")
