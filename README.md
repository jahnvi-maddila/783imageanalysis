# IC Pin Detection Model Training

This repository contains a comprehensive Jupyter notebook for training a YOLOv11 model to detect and classify IC (Integrated Circuit) pins and identify defects.

## Overview

The model can detect and classify:
- **Bent pins**: Defective pins that are bent
- **Okay pins**: Normal, properly aligned pins
- **Package**: IC package detection
- **Text**: Text regions on the IC

## Dataset Structure

The training data follows the YOLO format with the following structure:

```
783-Pin-Detection/
└── datasets/
    └── aug-ic-dataset/
        ├── images/
        │   ├── train/     # Training images
        │   └── val/       # Validation images
        └── labels/
            ├── train/     # Training labels (YOLO format .txt files)
            └── val/       # Validation labels (YOLO format .txt files)
```

### Label Format

Each label file (`.txt`) contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: 0=bent, 1=okay, 2=package, 3=text
- All coordinates are normalized (0-1)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd 783imageanalysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install ultralytics opencv-python matplotlib pillow pyyaml jupyter
```

### 3. Verify Dataset

Ensure your dataset is properly structured in the `783-Pin-Detection/datasets/aug-ic-dataset/` directory.

## Usage

### Training the Model

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook train_model.ipynb
   ```

2. **Run the cells sequentially**:
   - The notebook will guide you through:
     - Dataset verification
     - Model initialization
     - Training configuration
     - Model training
     - Evaluation and testing
     - Model export

3. **Customize Training Parameters** (Optional):
   
   In the training configuration cell, you can adjust:
   ```python
   training_params = {
       'epochs': 300,        # Number of training epochs
       'imgsz': 800,         # Image size
       'batch': 16,          # Batch size
       'device': 0,          # GPU device (0) or 'cpu'
       'patience': 100,      # Early stopping patience
       # ... more parameters
   }
   ```

### Using the Trained Model

After training, the model is saved in `runs/detect/ic_pin_detection/weights/`:
- `best.pt`: Best performing model
- `last.pt`: Last epoch model

**Inference Example**:

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/ic_pin_detection/weights/best.pt')

# Run inference on an image
results = model('path/to/image.jpg')

# Display results
results[0].show()

# Get detection details
for box in results[0].boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = model.names[class_id]
    print(f"{class_name}: {confidence:.2%}")
```

## Configuration

### Dataset Configuration (dataset.yaml)

```yaml
train: 783-Pin-Detection/datasets/aug-ic-dataset/images/train
val: 783-Pin-Detection/datasets/aug-ic-dataset/images/val
nc: 4
names:
  0: bent
  1: okay
  2: package
  3: text
```

### Model Sizes

Choose from different YOLOv11 model sizes based on your needs:

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolo11n.pt | Nano | Fastest | Good |
| yolo11s.pt | Small | Fast | Better |
| yolo11m.pt | Medium | Moderate | Best balance ⭐ |
| yolo11l.pt | Large | Slow | High |
| yolo11x.pt | Extra Large | Slowest | Highest |

## Training Tips

1. **GPU Recommended**: Training is much faster on GPU. Adjust `device` parameter accordingly.

2. **Batch Size**: If you get out-of-memory errors, reduce the batch size:
   ```python
   'batch': 8,  # or 4
   ```

3. **Image Size**: Larger images improve accuracy but increase training time:
   ```python
   'imgsz': 640,  # or 800, 1024
   ```

4. **Early Stopping**: Training stops automatically if no improvement:
   ```python
   'patience': 100,  # Stop if no improvement for 100 epochs
   ```

5. **Data Augmentation**: Already configured with reasonable defaults, but can be adjusted in training_params.

## Results and Metrics

After training, you'll find in `runs/detect/ic_pin_detection/`:
- `weights/best.pt`: Best model weights
- `weights/last.pt`: Last epoch weights
- `results.png`: Training curves (loss, metrics)
- `confusion_matrix.png`: Confusion matrix
- `val_batch*_pred.jpg`: Validation predictions

### Key Metrics

- **mAP@50**: Mean Average Precision at IoU threshold 0.5
- **mAP@50-95**: Mean Average Precision averaged over IoU thresholds 0.5-0.95
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'ultralytics'**
   ```bash
   pip install ultralytics
   ```

2. **CUDA out of memory**
   - Reduce batch size
   - Reduce image size
   - Use a smaller model (e.g., yolo11n.pt)

3. **No training/validation images found**
   - Verify dataset paths in `dataset.yaml`
   - Check image file extensions (.jpg, .png)

4. **Training too slow**
   - Enable GPU: `'device': 0`
   - Reduce image size: `'imgsz': 640`
   - Use smaller model: `yolo11s.pt`

## Next Steps

1. **Improve the Model**:
   - Collect more training data
   - Balance the dataset classes
   - Experiment with hyperparameters
   - Try different model sizes

2. **Deploy the Model**:
   - Export to ONNX format for production
   - Integrate into quality control system
   - Set up automated testing pipeline

3. **Monitor Performance**:
   - Track model metrics over time
   - Collect edge cases
   - Retrain periodically

## References

- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)
- [YOLO Paper](https://arxiv.org/abs/2304.00501)
- Original project: [Counterfeit IC Detection](https://github.com/rahulbhattachan/Counterfeit_IC_Detection)

## License

This project is based on the Counterfeit IC Detection project and follows similar licensing terms.
