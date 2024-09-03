from datasets import load_dataset

print("roberta bitfit")

dataset = load_dataset("glue", "sst2")

from transformers import RobertaTokenizer, RobertaForSequenceClassification

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

for name, param in model.named_parameters():
    if "bias" or "classifier" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

train_dataset = tokenized_datasets["train"].select(range(100))
eval_dataset = tokenized_datasets["validation"].select(range(100))

from transformers import Trainer, TrainingArguments, TrainerCallback
from copy import deepcopy

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',           
    logging_strategy="epoch",       
    logging_steps=10,
)

from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.add_callback(CustomCallback(trainer)) 
trainer.train()

trainer.evaluate()

import matplotlib.pyplot as plt

log_history = trainer.state.log_history

train_accuracy = [entry['train_accuracy'] for entry in log_history if 'train_accuracy' in entry]
eval_accuracy = [entry['eval_accuracy'] for entry in log_history if 'eval_accuracy' in entry]

if len(eval_accuracy) > len(train_accuracy):
    eval_accuracy = eval_accuracy[:len(train_accuracy)]
elif len(train_accuracy) > len(eval_accuracy):
    train_accuracy = train_accuracy[:len(eval_accuracy)]

epochs = list(range(1, len(train_accuracy) + 1))

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_accuracy, label="Training Accuracy")
plt.plot(epochs, eval_accuracy, label="Evaluation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Evaluation Accuracy per Epoch")
plt.legend()
plt.grid()
print("roberta_plot")
plt.savefig('roberta_bitfit.png')

