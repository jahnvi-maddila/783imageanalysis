#!/usr/bin/env python3
"""
Simple Inference Script for IC Pin Detection

Usage:
    python inference.py <image_path> [--model MODEL_PATH] [--conf CONFIDENCE]

Example:
    python inference.py test_image.jpg
    python inference.py test_image.jpg --model runs/detect/ic_pin_detection/weights/best.pt
    python inference.py test_image.jpg --conf 0.5
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2


def main():
    parser = argparse.ArgumentParser(description='Run IC pin detection on an image')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--model', type=str, 
                       default='runs/detect/ic_pin_detection/weights/best.pt',
                       help='Path to trained model (default: runs/detect/ic_pin_detection/weights/best.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--output', type=str, default='output.jpg',
                       help='Path to save output image (default: output.jpg)')
    parser.add_argument('--show', action='store_true',
                       help='Display the result image')
    
    args = parser.parse_args()
    
    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image not found: {args.image}")
        return 1
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model not found: {args.model}")
        print("\nDid you train the model yet?")
        print("Run the train_model.ipynb notebook first to train a model.")
        return 1
    
    print(f"Loading model from: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    print(f"Running inference on: {args.image}")
    print(f"Confidence threshold: {args.conf}")
    
    # Run inference
    results = model(args.image, conf=args.conf, verbose=False)
    
    # Get detection results
    boxes = results[0].boxes
    
    print(f"\n{'='*60}")
    print(f"Detection Results")
    print(f"{'='*60}")
    
    if len(boxes) > 0:
        print(f"Found {len(boxes)} object(s):\n")
        
        # Count detections by class
        class_counts = {}
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            print(f"  {class_name:12s} | Confidence: {conf:.2%} | Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
        
        print(f"\n{'='*60}")
        print("Summary by Class:")
        print(f"{'='*60}")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name:12s}: {count}")
        
        # Check for defects
        if 'bent' in class_counts:
            print(f"\n⚠️  WARNING: Detected {class_counts['bent']} bent pin(s)!")
        else:
            print(f"\n✓ No bent pins detected")
    else:
        print("No objects detected")
        print(f"\nTry lowering the confidence threshold with --conf")
    
    # Save annotated image
    annotated = results[0].plot()
    cv2.imwrite(args.output, annotated)
    print(f"\nAnnotated image saved to: {args.output}")
    
    # Show image if requested
    if args.show:
        print("\nDisplaying image... (press any key to close)")
        cv2.imshow('IC Pin Detection', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return 0


if __name__ == "__main__":
    import sys
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
