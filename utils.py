import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function to test the model on new input
def test_model(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    return sentiment_map[prediction]

# Function to plot the confusion matrix
def plot_confusion_matrix(model, tokenizer, dataset, device):
    y_true = []
    y_pred = []
    for sample in dataset:
        inputs = tokenizer(sample["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
        outputs = model(**inputs)
        y_pred.append(torch.argmax(outputs.logits, dim=-1).item())
        y_true.append(sample["labels"])

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
