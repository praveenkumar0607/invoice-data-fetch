# check_version.py (Version 2)
import transformers
import dataclasses

print("-" * 50)
print(f"✅ Transformers library version: {transformers.__version__}")
print(f"✅ Library location: {transformers.__file__}")
print("-" * 50)

try:
    from transformers import TrainingArguments
    
    # This is a more direct way to check for the argument
    field_names = {f.name for f in dataclasses.fields(TrainingArguments)}
    
    if 'evaluation_strategy' in field_names:
        print("👍 This version SUPPORTS 'evaluation_strategy'.")
        print("   Your code in src/trainer.py MUST use this argument.")
    else:
        print("❌ This version DOES NOT support 'evaluation_strategy'.")

    if 'evaluate_during_training' in field_names:
        print("👍 This version SUPPORTS the older 'evaluate_during_training'.")
    else:
        print("❌ This version DOES NOT support 'evaluate_during_training'.")

except Exception as e:
    print(f"An error occurred during the check: {e}")

print("-" * 50)