#!/usr/bin/env python3
"""
Dataset Validation Script

This script validates that the IC pin detection dataset is properly formatted
and ready for training.
"""

import sys
from pathlib import Path
import yaml


def validate_dataset():
    """Validate the dataset structure and configuration."""
    
    print("=" * 60)
    print("IC Pin Detection Dataset Validation")
    print("=" * 60)
    print()
    
    errors = []
    warnings = []
    
    # Check dataset.yaml exists
    config_path = Path("dataset.yaml")
    if not config_path.exists():
        errors.append("dataset.yaml not found in current directory")
        print("❌ dataset.yaml not found")
        return False
    
    print("✓ dataset.yaml found")
    
    # Load and validate YAML
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✓ dataset.yaml is valid YAML")
    except Exception as e:
        errors.append(f"Failed to parse dataset.yaml: {e}")
        print(f"❌ Failed to parse dataset.yaml: {e}")
        return False
    
    # Check required keys
    required_keys = ['train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in config:
            errors.append(f"Missing required key '{key}' in dataset.yaml")
            print(f"❌ Missing required key: {key}")
        else:
            print(f"✓ Found key: {key}")
    
    if errors:
        return False
    
    # Validate paths
    train_path = Path(config['train'])
    val_path = Path(config['val'])
    
    print("\nChecking dataset paths...")
    
    if not train_path.exists():
        errors.append(f"Training images path does not exist: {train_path}")
        print(f"❌ Training images path not found: {train_path}")
    else:
        train_images = list(train_path.glob("*.[jp][pn]g"))
        print(f"✓ Training images path exists: {train_path}")
        print(f"  Found {len(train_images)} training images")
        
        if len(train_images) == 0:
            warnings.append("No training images found")
            print("⚠️  Warning: No training images found")
    
    if not val_path.exists():
        errors.append(f"Validation images path does not exist: {val_path}")
        print(f"❌ Validation images path not found: {val_path}")
    else:
        val_images = list(val_path.glob("*.[jp][pn]g"))
        print(f"✓ Validation images path exists: {val_path}")
        print(f"  Found {len(val_images)} validation images")
        
        if len(val_images) == 0:
            warnings.append("No validation images found")
            print("⚠️  Warning: No validation images found")
    
    # Check labels
    print("\nChecking label files...")
    
    train_labels_path = train_path.parent.parent / "labels" / "train"
    val_labels_path = val_path.parent.parent / "labels" / "val"
    
    if not train_labels_path.exists():
        errors.append(f"Training labels path does not exist: {train_labels_path}")
        print(f"❌ Training labels path not found: {train_labels_path}")
    else:
        train_labels = list(train_labels_path.glob("*.txt"))
        print(f"✓ Training labels path exists: {train_labels_path}")
        print(f"  Found {len(train_labels)} training label files")
        
        if len(train_labels) == 0:
            warnings.append("No training label files found")
            print("⚠️  Warning: No training label files found")
    
    if not val_labels_path.exists():
        errors.append(f"Validation labels path does not exist: {val_labels_path}")
        print(f"❌ Validation labels path not found: {val_labels_path}")
    else:
        val_labels = list(val_labels_path.glob("*.txt"))
        print(f"✓ Validation labels path exists: {val_labels_path}")
        print(f"  Found {len(val_labels)} validation label files")
        
        if len(val_labels) == 0:
            warnings.append("No validation label files found")
            print("⚠️  Warning: No validation label files found")
    
    # Validate class configuration
    print("\nChecking class configuration...")
    
    nc = config['nc']
    names = config['names']
    
    if len(names) != nc:
        errors.append(f"Number of classes ({nc}) doesn't match number of names ({len(names)})")
        print(f"❌ Class count mismatch: nc={nc}, but {len(names)} names provided")
    else:
        print(f"✓ Class configuration is consistent: {nc} classes")
    
    # Display class names
    print("\nClass names:")
    for idx, name in names.items():
        print(f"  {idx}: {name}")
    
    # Sample a label file to check format
    if train_labels_path.exists():
        label_files = list(train_labels_path.glob("*.txt"))
        if label_files:
            print("\nValidating label file format...")
            sample_label = label_files[0]
            
            try:
                with open(sample_label, 'r') as f:
                    lines = f.readlines()
                    
                if len(lines) == 0:
                    warnings.append(f"Empty label file: {sample_label.name}")
                    print(f"⚠️  Warning: Sample label file is empty")
                else:
                    # Check first line format
                    parts = lines[0].strip().split()
                    if len(parts) != 5:
                        warnings.append(f"Label format issue in {sample_label.name}")
                        print(f"⚠️  Warning: Expected 5 values per line, got {len(parts)}")
                    else:
                        try:
                            class_id = int(parts[0])
                            coords = [float(x) for x in parts[1:]]
                            
                            if class_id >= nc:
                                warnings.append(f"Class ID {class_id} exceeds nc={nc}")
                                print(f"⚠️  Warning: Found class ID {class_id}, but only {nc} classes defined")
                            
                            for coord in coords:
                                if coord < 0 or coord > 1:
                                    warnings.append("Coordinates should be normalized (0-1)")
                                    print(f"⚠️  Warning: Found coordinate {coord} outside [0,1] range")
                                    break
                            
                            print(f"✓ Label format looks correct (sample: {sample_label.name})")
                            
                        except ValueError:
                            warnings.append(f"Invalid number format in {sample_label.name}")
                            print(f"⚠️  Warning: Invalid number format in label file")
                
            except Exception as e:
                warnings.append(f"Error reading label file: {e}")
                print(f"⚠️  Warning: Error reading label file: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    if errors:
        print(f"\n❌ Found {len(errors)} error(s):")
        for error in errors:
            print(f"   - {error}")
    
    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"   - {warning}")
    
    if not errors and not warnings:
        print("\n✓ ✓ ✓ All checks passed! Dataset is ready for training.")
        return True
    elif not errors:
        print("\n✓ No critical errors found. You can proceed with training.")
        print("   (But review the warnings above)")
        return True
    else:
        print("\n❌ Please fix the errors above before training.")
        return False


if __name__ == "__main__":
    try:
        success = validate_dataset()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
