import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import matplotlib.pyplot as plt

# Disable Weights & Biases logging
os.environ["WANDB_DISABLED"] = "true"

# Set device (use GPU if available)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the Yelp Reviews dataset
def load_dataset_and_prepare():
    print("Loading and tokenizing the dataset...")
    dataset = load_dataset("yelp_review_full")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_datasets, tokenizer

# Initialize the BERT model
def initialize_model():
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
    model.to(device)
    return model

# Train and evaluate the model
def train_and_evaluate_model(model, tokenized_datasets, tokenizer):
    metric = load_metric("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        save_steps=500,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Training the model...")
    trainer.train()
    trainer.save_model("./fine_tuned_bert")

    print("Evaluating the model...")
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)

    return trainer

# Plot training and evaluation metrics
def plot_metrics(trainer):
    metrics = trainer.state.log_history
    train_loss = [m["loss"] for m in metrics if "loss" in m]
    eval_accuracy = [m["eval_accuracy"] for m in metrics if "eval_accuracy" in m]
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, eval_accuracy, label="Evaluation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.title("Training and Evaluation Metrics")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    datasets, tokenizer = load_dataset_and_prepare()
    model = initialize_model()
    trainer = train_and_evaluate_model(model, datasets, tokenizer)
    plot_metrics(trainer)
