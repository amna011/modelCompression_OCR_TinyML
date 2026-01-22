#!/usr/bin/env python3
"""
Batch convert multiple TensorFlow Lite models to C headers for ESP32 deployment.
Processes all models in the compressed_models directory.
"""

import os
import sys
from pathlib import Path
from convert_model import convert_tflite_to_header

def batch_convert_models(models_dir="../../compressed_models", output_dir="main"):
    """Convert all TFLite models in a directory to C headers."""
    
    models_path = Path(models_dir)
    output_path = Path(output_dir)
    
    if not models_path.exists():
        print(f"❌ Models directory not found: {models_path}")
        return False
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    
    # Find all .tflite files
    tflite_files = list(models_path.glob("*.tflite"))
    
    if not tflite_files:
        print(f"❌ No .tflite files found in {models_path}")
        return False
    
    print(f"🔍 Found {len(tflite_files)} TensorFlow Lite models")
    print(f"📁 Output directory: {output_path}")
    print("=" * 60)
    
    converted_count = 0
    failed_count = 0
    
    for tflite_file in tflite_files:
        print(f"\n📦 Processing: {tflite_file.name}")
        
        # Generate model name from filename
        model_name = tflite_file.stem.replace("-", "_").replace(".", "_")
        output_file = output_path / f"{model_name}_data.h"
        
        # Convert model
        success = convert_tflite_to_header(
            str(tflite_file), 
            str(output_file), 
            model_name
        )
        
        if success:
            converted_count += 1
        else:
            failed_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Conversion Summary:")
    print(f"   ✅ Successfully converted: {converted_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"   📁 Output files in: {output_path}")
    
    if converted_count > 0:
        print(f"\n🚀 Ready to build ESP32 project with {converted_count} models!")
        
        # Generate include statements
        print(f"\n📝 Add these includes to your main.cpp:")
        for tflite_file in tflite_files:
            model_name = tflite_file.stem.replace("-", "_").replace(".", "_")
            print(f'#include "{model_name}_data.h"')
    
    return failed_count == 0

def main():
    if len(sys.argv) > 1:
        models_dir = sys.argv[1]
    else:
        models_dir = "../../compressed_models"
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "main"
    
    print("🔄 Batch Model Conversion for ESP32")
    print(f"📂 Models directory: {models_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    success = batch_convert_models(models_dir, output_dir)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()