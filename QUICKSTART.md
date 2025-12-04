# Quick Start Guide

This guide will help you get started with training your IC pin detection model in under 5 minutes.

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (optional but recommended)
- At least 8GB RAM (16GB recommended)

## Step-by-Step Instructions

### 1. Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

If you get any errors, try installing the core packages individually:
```bash
pip install ultralytics opencv-python matplotlib pillow pyyaml jupyter
```

### 2. Verify Your Dataset

Make sure your dataset is in the correct location:
```
783-Pin-Detection/datasets/aug-ic-dataset/
├── images/
│   ├── train/     (should contain training images)
│   └── val/       (should contain validation images)
└── labels/
    ├── train/     (should contain .txt label files)
    └── val/       (should contain .txt label files)
```

### 3. Open the Notebook

```bash
jupyter notebook train_model.ipynb
```

Your browser should open automatically with the notebook.

### 4. Run the Training

In Jupyter:
1. Click "Cell" → "Run All" to run all cells
2. Or run cells one by one using Shift+Enter

The notebook will:
- ✓ Verify your dataset
- ✓ Show sample images
- ✓ Train the model
- ✓ Evaluate performance
- ✓ Save the trained model

### 5. Monitor Training

Training progress will show:
- Epoch number
- Loss values (box, cls, dfl)
- Metrics (precision, recall, mAP)
- Estimated time remaining

**Training time**: 
- With GPU: ~30-60 minutes (for 300 epochs)
- With CPU: ~4-6 hours (for 300 epochs)

### 6. Use Your Trained Model

After training completes, find your model at:
```
runs/detect/ic_pin_detection/weights/best.pt
```

**Quick inference test:**

```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/detect/ic_pin_detection/weights/best.pt')

# Test on an image
results = model('path/to/test/image.jpg')
results[0].show()
```

## Troubleshooting

### Issue: "No module named 'ultralytics'"
**Solution**: 
```bash
pip install ultralytics
```

### Issue: "CUDA out of memory"
**Solution**: In the training configuration cell, reduce batch size:
```python
'batch': 8,  # or even 4
```

### Issue: "RuntimeError: torch not compiled with CUDA enabled"
**Solution**: Either install CUDA-enabled PyTorch or use CPU:
```python
'device': 'cpu',
```

### Issue: Training is very slow
**Solutions**:
1. Use GPU instead of CPU
2. Reduce image size: `'imgsz': 640`
3. Use smaller model: `model_size = "yolo11s.pt"`
4. Reduce epochs: `'epochs': 100`

## Quick Configuration Changes

### For Faster Training (Lower Accuracy)
```python
model_size = "yolo11n.pt"  # Use nano model
training_params = {
    'epochs': 100,
    'imgsz': 640,
    'batch': 16,
}
```

### For Better Accuracy (Slower Training)
```python
model_size = "yolo11l.pt"  # Use large model
training_params = {
    'epochs': 500,
    'imgsz': 1024,
    'batch': 8,
}
```

### For CPU Training
```python
training_params = {
    'device': 'cpu',
    'batch': 4,
    'epochs': 50,  # Reduce epochs for CPU
    'imgsz': 640,
}
```

## Next Steps

After successful training:
1. Check `runs/detect/ic_pin_detection/results.png` for training curves
2. Review `confusion_matrix.png` to understand model performance
3. Test on new images using the inference code
4. Fine-tune hyperparameters if needed
5. Collect more data to improve accuracy

## Getting Help

- Review the full README.md for detailed documentation
- Check [Ultralytics Documentation](https://docs.ultralytics.com/)
- Open an issue on GitHub if you encounter problems

## Common Tasks

### Retrain with Different Settings
Simply modify the training_params in the notebook and run the training cell again.

### Resume Training
```python
model = YOLO('runs/detect/ic_pin_detection/weights/last.pt')
results = model.train(resume=True)
```

### Export Model for Deployment
```python
model = YOLO('runs/detect/ic_pin_detection/weights/best.pt')
model.export(format='onnx')  # Export to ONNX
```

---

**Need more help?** See the full README.md or check the Ultralytics documentation.
