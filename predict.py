# predict.py
import argparse
import json
from src.config import ExtractionConfig
from src.inference import InferenceEngine

def main():
    parser = argparse.ArgumentParser(description='Run Inference on an Invoice')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the invoice image or PDF')
    parser.add_argument('--model_path', type=str, default='./models', help='Path to the trained model directory')
    parser.add_argument('--config_path', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()

    config = ExtractionConfig.from_json_file(args.config_path)
    
    print(f"Loading model from {args.model_path}...")
    engine = InferenceEngine(model_path=args.model_path, config=config)
    
    print(f"Extracting data from {args.image_path}...")
    results = engine.extract_from_image(args.image_path)
    
    print("\n--- Extracted Data ---")
    print(json.dumps(results, indent=4))
    print("--------------------")

if __name__ == "__main__":
    main()