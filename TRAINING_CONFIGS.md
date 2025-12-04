# Training Configuration Examples

This file contains example training configurations for different scenarios.
Copy and paste these into the training_params in the notebook.

## 1. Quick Test Configuration (Fast, Low Accuracy)
# Use this for testing the setup and making sure everything works

training_params = {
    'data': dataset_config,
    'epochs': 10,                      # Very few epochs for quick test
    'imgsz': 416,                      # Smaller image size
    'batch': 8,                        # Small batch
    'device': 0,                       # GPU
    'patience': 5,
    'save': True,
    'cache': False,                    # Don't cache for testing
    'project': 'runs/detect',
    'name': 'test_run',
}

## 2. CPU Training Configuration
# Use when GPU is not available (will be slow)

training_params = {
    'data': dataset_config,
    'epochs': 50,                      # Fewer epochs for CPU
    'imgsz': 640,
    'batch': 4,                        # Small batch for CPU
    'device': 'cpu',                   # Use CPU
    'patience': 25,
    'save': True,
    'cache': False,
    'project': 'runs/detect',
    'name': 'cpu_training',
}

## 3. Balanced Configuration (Recommended)
# Good balance between speed and accuracy

training_params = {
    'data': dataset_config,
    'epochs': 300,
    'imgsz': 800,
    'batch': 16,
    'device': 0,
    'patience': 100,
    'save': True,
    'cache': True,
    'cos_lr': True,
    'close_mosaic': 10,
    'box': 10,
    'cls': 15.0,
    'dfl': 3.0,
    'flipud': 0.5,
    'fliplr': 0.5,
    'project': 'runs/detect',
    'name': 'ic_pin_detection',
}

## 4. High Accuracy Configuration
# For best possible results (requires good GPU)

training_params = {
    'data': dataset_config,
    'epochs': 500,
    'imgsz': 1024,                     # Larger images
    'batch': 8,                        # Smaller batch for large images
    'device': 0,
    'patience': 150,
    'save': True,
    'cache': True,
    'cos_lr': True,
    'close_mosaic': 10,
    'box': 12,                         # Higher box weight
    'cls': 20.0,                       # Higher classification weight
    'dfl': 3.0,
    'flipud': 0.5,
    'fliplr': 0.5,
    'degrees': 10.0,                   # Add rotation augmentation
    'translate': 0.2,
    'scale': 0.6,
    'project': 'runs/detect',
    'name': 'high_accuracy',
}

## 5. Speed-Optimized Configuration
# For faster training with acceptable accuracy

training_params = {
    'data': dataset_config,
    'epochs': 150,
    'imgsz': 640,                      # Smaller images
    'batch': 32,                       # Larger batch
    'device': 0,
    'patience': 50,
    'save': True,
    'cache': True,
    'cos_lr': True,
    'close_mosaic': 5,
    'box': 10,
    'cls': 15.0,
    'dfl': 3.0,
    'flipud': 0.5,
    'fliplr': 0.5,
    'project': 'runs/detect',
    'name': 'speed_optimized',
}
# Also use: model_size = "yolo11s.pt" or "yolo11n.pt"

## 6. Fine-tuning Configuration
# When you already have a trained model and want to improve it

training_params = {
    'data': dataset_config,
    'epochs': 100,
    'imgsz': 800,
    'batch': 16,
    'device': 0,
    'patience': 50,
    'save': True,
    'cache': True,
    'cos_lr': True,
    'lr0': 0.001,                      # Lower learning rate for fine-tuning
    'lrf': 0.001,
    'warmup_epochs': 1.0,
    'box': 10,
    'cls': 15.0,
    'dfl': 3.0,
    'project': 'runs/detect',
    'name': 'fine_tuned',
}
# Use: model = YOLO('runs/detect/ic_pin_detection/weights/best.pt')

## 7. Memory-Constrained Configuration
# For systems with limited GPU memory

training_params = {
    'data': dataset_config,
    'epochs': 200,
    'imgsz': 640,
    'batch': 4,                        # Very small batch
    'device': 0,
    'patience': 75,
    'save': True,
    'cache': False,                    # Don't cache to save memory
    'cos_lr': True,
    'close_mosaic': 10,
    'box': 10,
    'cls': 15.0,
    'dfl': 3.0,
    'project': 'runs/detect',
    'name': 'low_memory',
}
# Also use: model_size = "yolo11n.pt"

## 8. Production-Ready Configuration
# For final production model with best practices

training_params = {
    'data': dataset_config,
    'epochs': 400,
    'imgsz': 800,
    'batch': 16,
    'device': 0,
    'patience': 120,
    'save': True,
    'save_period': 50,                 # Save checkpoint every 50 epochs
    'cache': True,
    'cos_lr': True,
    'close_mosaic': 15,
    'box': 10,
    'cls': 18.0,                       # Higher for defect detection
    'dfl': 3.0,
    'flipud': 0.5,
    'fliplr': 0.5,
    'degrees': 5.0,
    'translate': 0.15,
    'scale': 0.5,
    'hsv_h': 0.015,                    # Color augmentation
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'project': 'runs/detect',
    'name': 'production_model',
}

## 9. Debug Configuration
# For debugging and experimentation

training_params = {
    'data': dataset_config,
    'epochs': 5,                       # Very few epochs
    'imgsz': 416,
    'batch': 2,
    'device': 0,
    'patience': 5,
    'save': True,
    'cache': False,
    'verbose': True,                   # Verbose output
    'plots': True,                     # Generate plots
    'project': 'runs/detect',
    'name': 'debug',
}

## Model Size Selection

# Nano - Fastest, lowest accuracy
model_size = "yolo11n.pt"
# Use when: Speed is critical, limited compute, proof of concept

# Small - Fast, good accuracy
model_size = "yolo11s.pt"
# Use when: Good balance needed, production deployment

# Medium - Balanced (Recommended for most cases)
model_size = "yolo11m.pt"
# Use when: Best balance of speed and accuracy

# Large - Slower, high accuracy
model_size = "yolo11l.pt"
# Use when: Accuracy is more important than speed

# Extra Large - Slowest, highest accuracy
model_size = "yolo11x.pt"
# Use when: Maximum accuracy needed, speed not important

## Common Parameter Explanations

# epochs: Number of complete passes through the dataset
#   - More epochs = longer training, potentially better accuracy
#   - Too many = overfitting
#   - Typical range: 100-500

# imgsz: Input image size (pixels)
#   - Larger = better accuracy, slower training, more memory
#   - Typical values: 416, 640, 800, 1024

# batch: Number of images processed together
#   - Larger = faster training, more memory needed
#   - Smaller = less memory, more stable training
#   - Typical range: 4-32

# patience: Early stopping - stops if no improvement
#   - Higher = more patient, trains longer
#   - Lower = stops earlier if not improving
#   - 0 = no early stopping

# cache: Cache images in RAM
#   - True = faster training, uses more RAM
#   - False = slower, less RAM usage
#   - 'disk' = cache to disk

# cos_lr: Cosine learning rate scheduler
#   - True = smooth learning rate decay
#   - Recommended: True

# box, cls, dfl: Loss weights
#   - Higher = more emphasis on that loss component
#   - Increase 'cls' for classification focus (defect detection)

# Augmentation parameters:
#   - flipud, fliplr: Flip probability (0-1)
#   - degrees: Rotation range (-deg to +deg)
#   - translate: Translation (-frac to +frac)
#   - scale: Scale factor (0-1)
#   - hsv_*: Color augmentation
