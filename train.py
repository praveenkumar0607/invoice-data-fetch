# train.py
import argparse
from transformers import LayoutLMv3Processor
from src.config import ExtractionConfig
from src.dataset import InvoiceDataset
from src.trainer import ModelTrainer
from transformers import TrainingArguments

def main():
    parser = argparse.ArgumentParser(description='Train Invoice Extraction Model')
    parser.add_argument('--config_path', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()

    print("1. Loading configuration...")
    config = ExtractionConfig.from_json_file(args.config_path)

    print("2. Loading processor...")
    processor = LayoutLMv3Processor.from_pretrained(
    config.model_name,
    apply_ocr=False
    )

    print("3. Creating datasets...")
    train_dataset = InvoiceDataset(
        data_path=f"{config.data_dir}/train",
        processor=processor,
        config=config,
        is_training=True
    )
    val_dataset = InvoiceDataset(
        data_path=f"{config.data_dir}/val",
        processor=processor,
        config=config,
        is_training=False
    )
    
    print("4. Starting training...")
    trainer = ModelTrainer(config)
    trainer.train(train_dataset, val_dataset)
    print("Training complete! Model saved to:", config.output_dir)

if __name__ == "__main__":
    main()
