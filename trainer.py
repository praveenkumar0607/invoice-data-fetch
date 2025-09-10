# src/trainer.py
import torch
from transformers import Trainer, TrainingArguments
from .model import InvoiceExtractor as InvoiceModel  # ✅ correct import

class ModelTrainer:
    """Training pipeline for invoice extraction"""
    def __init__(self, config):
        self.config = config
        self.model = InvoiceModel(config)

    def train(self, train_dataset, val_dataset):
        # Use single batch_size from config.json
        batch_size = self.config.batch_size
        learning_rate = self.config.learning_rate
        num_epochs = self.config.num_epochs
        warmup_steps = getattr(self.config, "warmup_steps", 500)

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=50,
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            report_to="none",
            eval_strategy="steps",   # ✅ compatible with Transformers 4.56.1
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # tokenizer=train_dataset.processor,  # ❌ removed to avoid eos_token_id error
        )

        print("Starting training...")
        trainer.train()
        print("Training finished! Saving model...")
        self.model.model.save_pretrained(self.config.output_dir)  # save underlying HF model
        print("Model saved at:", self.config.output_dir)
