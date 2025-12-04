# Troubleshooting Guide

Common issues and their solutions when training the IC pin detection model.

## Installation Issues

### Issue: "No module named 'ultralytics'"

**Cause**: The ultralytics package is not installed.

**Solution**:
```bash
pip install ultralytics
```

If that fails, try:
```bash
pip install --upgrade pip
pip install ultralytics
```

### Issue: "ImportError: DLL load failed" or "Cannot import cv2"

**Cause**: OpenCV is not properly installed.

**Solution**:
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

### Issue: Requirements installation fails

**Cause**: Conflicting package versions.

**Solution**: Install packages individually:
```bash
pip install ultralytics
pip install opencv-python
pip install matplotlib
pip install pillow
pip install pyyaml
pip install jupyter
```

## Training Issues

### Issue: "CUDA out of memory"

**Cause**: GPU memory is insufficient for the batch size.

**Solutions** (try in order):
1. Reduce batch size:
   ```python
   'batch': 8,  # or 4, or 2
   ```

2. Reduce image size:
   ```python
   'imgsz': 640,  # instead of 800
   ```

3. Use a smaller model:
   ```python
   model_size = "yolo11s.pt"  # instead of yolo11m.pt
   ```

4. Use CPU (slower but no memory limits):
   ```python
   'device': 'cpu',
   ```

### Issue: "RuntimeError: torch not compiled with CUDA enabled"

**Cause**: PyTorch is not installed with CUDA support, but you're trying to use GPU.

**Solutions**:
1. Use CPU instead:
   ```python
   'device': 'cpu',
   ```

2. Or install CUDA-enabled PyTorch:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### Issue: Training is extremely slow

**Causes & Solutions**:

1. **Using CPU instead of GPU**:
   - Check: `'device': 0` for GPU or `'device': 'cpu'`
   - Solution: Use GPU if available
   
2. **Large batch size on CPU**:
   - Reduce: `'batch': 4`
   
3. **Large image size**:
   - Reduce: `'imgsz': 640`
   
4. **Large model**:
   - Use smaller: `model_size = "yolo11n.pt"`

5. **Too many epochs**:
   - Reduce for testing: `'epochs': 50`

### Issue: "FileNotFoundError: [Errno 2] No such file or directory: 'dataset.yaml'"

**Cause**: The notebook is not finding the dataset configuration file.

**Solution**:
1. Make sure you're running the notebook from the repository root directory
2. Check that `dataset.yaml` exists in the same directory as the notebook
3. Verify the paths in `dataset.yaml` are correct

### Issue: No images or labels found

**Cause**: Dataset paths are incorrect or dataset structure is wrong.

**Solution**:
1. Run the validation script:
   ```bash
   python validate_dataset.py
   ```

2. Check dataset structure:
   ```
   783-Pin-Detection/datasets/aug-ic-dataset/
   ├── images/
   │   ├── train/    # Should have .jpg or .png files
   │   └── val/      # Should have .jpg or .png files
   └── labels/
       ├── train/    # Should have .txt files
       └── val/      # Should have .txt files
   ```

3. Verify paths in `dataset.yaml` match your directory structure

## Notebook Issues

### Issue: Jupyter notebook won't start

**Cause**: Jupyter is not installed or port is in use.

**Solutions**:
1. Install Jupyter:
   ```bash
   pip install jupyter
   ```

2. Try a different port:
   ```bash
   jupyter notebook --port 8889
   ```

### Issue: Kernel keeps dying

**Cause**: Out of memory or system resources.

**Solutions**:
1. Restart the kernel: Kernel → Restart
2. Close other applications
3. Reduce batch size in training configuration
4. Use CPU instead of GPU
5. Restart your computer

### Issue: Cannot run cells / notebook is not responding

**Cause**: Long-running operation or kernel issue.

**Solutions**:
1. Wait for current cell to complete (look for [*] indicator)
2. Interrupt kernel: Kernel → Interrupt
3. Restart kernel: Kernel → Restart
4. Clear output and restart: Kernel → Restart & Clear Output

## Data Issues

### Issue: "Label file has wrong format"

**Cause**: Label files are not in YOLO format.

**Solution**:
Each label file should have format:
```
class_id x_center y_center width height
```
Where:
- `class_id` is an integer (0, 1, 2, or 3)
- All other values are floats between 0 and 1 (normalized coordinates)

Example:
```
1 0.267597 0.891586 0.03034 0.100324
0 0.304915 0.890979 0.031857 0.107201
```

### Issue: Low accuracy / model not learning

**Possible causes and solutions**:

1. **Insufficient training data**:
   - Collect more images
   - Use data augmentation (already enabled in training)

2. **Class imbalance**:
   - Check class distribution
   - Collect more samples of underrepresented classes

3. **Incorrect labels**:
   - Review and fix label annotations
   - Use validation script to check format

4. **Learning rate too high/low**:
   - Default should work, but can adjust:
     ```python
     'lr0': 0.001,  # Initial learning rate
     ```

5. **Not enough training**:
   - Increase epochs: `'epochs': 500`
   - Remove early stopping: `'patience': 0`

6. **Model too small**:
   - Use larger model: `model_size = "yolo11l.pt"`

## Inference Issues

### Issue: "Model file not found"

**Cause**: Training hasn't completed or model path is wrong.

**Solution**:
1. Complete training first
2. Check model exists at: `runs/detect/ic_pin_detection/weights/best.pt`
3. Update path in inference code

### Issue: No detections on test images

**Causes & Solutions**:

1. **Confidence threshold too high**:
   ```python
   results = model('image.jpg', conf=0.1)  # Lower threshold
   ```

2. **Model not trained properly**:
   - Check training metrics
   - Retrain with more data or different parameters

3. **Test images very different from training images**:
   - Add similar images to training set
   - Check image resolution and quality

### Issue: Too many false positives

**Cause**: Confidence threshold too low.

**Solution**:
```python
results = model('image.jpg', conf=0.5)  # Higher threshold
```

## Performance Issues

### Issue: Training takes too long

**For quick testing**:
```python
model_size = "yolo11n.pt"
training_params = {
    'epochs': 50,
    'imgsz': 640,
    'batch': 16,
    'device': 0,
}
```

**For production training** (if you have good GPU):
```python
model_size = "yolo11m.pt"
training_params = {
    'epochs': 300,
    'imgsz': 800,
    'batch': 16,
    'device': 0,
    'cache': True,
}
```

### Issue: Inference is slow

**Solutions**:
1. Use smaller model for deployment:
   - Export: `model.export(format='onnx')`
   - Use optimized runtime

2. Reduce image size:
   ```python
   results = model('image.jpg', imgsz=640)
   ```

3. Use GPU for inference:
   ```python
   model = YOLO('best.pt')
   model.to('cuda')
   ```

## Environment Issues

### Issue: Different results on different machines

**Cause**: Different package versions or CUDA versions.

**Solution**:
1. Use the same Python version (3.8+)
2. Use the same package versions (see requirements.txt)
3. Set random seed (already done in notebook)

### Issue: Cannot find GPU / CUDA device

**Check GPU availability**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

**Solutions**:
1. Install CUDA toolkit from NVIDIA
2. Install CUDA-enabled PyTorch
3. Update GPU drivers
4. Use CPU if GPU is not available

## Getting More Help

If you're still having issues:

1. **Check the error message carefully** - it often tells you exactly what's wrong

2. **Run the validation script**:
   ```bash
   python validate_dataset.py
   ```

3. **Check Ultralytics documentation**:
   - [Training Guide](https://docs.ultralytics.com/modes/train/)
   - [Common Issues](https://docs.ultralytics.com/help/FAQ/)

4. **Search for the error**:
   - [Ultralytics GitHub Issues](https://github.com/ultralytics/ultralytics/issues)
   - [Stack Overflow](https://stackoverflow.com)

5. **Create an issue**:
   - Include the full error message
   - Include your system info (OS, Python version, GPU)
   - Include relevant code snippets

## System Requirements

**Minimum**:
- Python 3.8+
- 8GB RAM
- 10GB disk space

**Recommended**:
- Python 3.8+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- 20GB+ disk space
- CUDA 11.8 or 12.1

**For CPU-only training**:
- 16GB+ RAM recommended
- Be prepared for slow training (hours instead of minutes)
- Consider using smaller model (yolo11n.pt)
- Reduce epochs and batch size
