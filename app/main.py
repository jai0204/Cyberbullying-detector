# Bert model code
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch
# Load a trained model and vocabulary that you have fine-tuned
output_dir = "./model_save/"
model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)

# Copy the model to the CPU.
device=torch.device("cpu")
model.to(device)

import numpy as np
# Put model in evaluation mode
model.eval()
# Evaluation function
def evaluate(review_text):
    predictions = []
    encoded_review = tokenizer.encode_plus(
      review_text,
      max_length=230,
      add_special_tokens=True,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask = encoded_review['input_ids'].to(device), encoded_review['attention_mask'].to(device)

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask)

    logits = outputs[0]

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()

    # Store predictions and true labels
    predictions.append(logits)
    pred_labels = np.argmax(predictions[0], axis=1).flatten()
    
    return pred_labels[0]



# Web server code
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        review_text = str(request.args.get("input-text"))
        return jsonify(prediction = int(evaluate(review_text)))
    # if request.method == 'POST':
    #     review_text = str(request.form.get("input-text"))
    #     return jsonify(prediction = int(evaluate(review_text)))
