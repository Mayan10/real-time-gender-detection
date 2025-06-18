#!/usr/bin/env python
# A script to train our Vision Transformer model for gender classification

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import sys
import platform
import gc
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import get_project_root

class GenderDataset(Dataset):
    """Our custom dataset for gender classification"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load the image from disk
        image = Image.open(img_path).convert('RGB')
        
        # Apply any transformations (like resizing, normalization, etc.)
        if self.transform:
            image = self.transform(image)
            
        return image, label

def train_model(
    data_dir, 
    batch_size=4,  # Reduced from 8 to 4 to save memory
    num_epochs=20, 
    learning_rate=1e-4,
    output_dir=None,
    use_metal=True,
    use_cuda=True,
    image_size=160,  # Reduced from 224 to 160 to save memory
    grad_accum_steps=8,  # Increased from 4 to 8 to simulate larger batches
    model_size="base",  # Default to base since tiny might not be available
    eval_every=2,  # Increased from 1 to 2 to save time
    save_every=5,
    early_stopping=3
):
    """
    Train the Vision Transformer model for gender classification with memory optimizations
    """
    # Clean up memory at the start to give us a fresh slate
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # For MPS devices (Apple Silicon), let's manage memory more carefully
    if platform.system() == 'Darwin':
        # Set conservative memory limits for MPS to avoid crashes
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.5'  # More conservative memory limit
        os.environ['PYTORCH_MPS_ALLOCATOR_POLICY'] = 'garbage_collection_conservative'
    
    # Figure out what device to use (GPU, Apple Silicon, or CPU)
    if platform.system() == 'Darwin' and use_metal and hasattr(torch, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Metal (MPS) device for training")
        print(f"MPS memory settings: PYTORCH_MPS_HIGH_WATERMARK_RATIO={os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'not set')}")
    elif torch.cuda.is_available() and use_cuda:
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU for training")
    
    # Pick which model size to use
    if model_size == "tiny":
        # ViT-tiny isn't a standard Hugging Face model, so let's use a known one instead
        model_name = "google/vit-base-patch16-224"
        print("ViT-tiny not available, using base model with reduced image size instead")
    elif model_size == "small":
        model_name = "google/vit-base-patch16-224"
        print("ViT-small not available, using base model instead")
    else:
        model_name = "google/vit-base-patch16-224"
    
    print(f"Using model: {model_name}")
    
    # Load up the Vision Transformer model and processor
    try:
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        print("Successfully loaded model")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Trying alternative approach...")
        try:
            # Try a different model as fallback
            fallback_model = "google/vit-base-patch16-224"
            processor = ViTImageProcessor.from_pretrained(fallback_model)
            model = ViTForImageClassification.from_pretrained(
                fallback_model,
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            print(f"Successfully loaded fallback model: {fallback_model}")
        except Exception as e2:
            print(f"Fatal error loading any model: {e2}")
            raise
    
    # Set up our class names (male and female)
    id2label = {0: "Female", 1: "Male"}
    label2id = {"Female": 0, "Male": 1}
    model.config.id2label = id2label
    model.config.label2id = label2id
    
    # Move the model to our chosen device
    model.to(device)
    print(f"Model moved to {device}")
    
    # Set up our data directories
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    # Set up data augmentation and preprocessing with reduced image size to save memory
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.2)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Prepare our datasets
    print("Preparing datasets...")
    
    # Training dataset - we can limit the number of images if memory is tight
    train_images = []
    train_labels = []
    
    for gender in ['female', 'male']:
        gender_dir = os.path.join(train_dir, gender)
        gender_label = 0 if gender == 'female' else 1
        
        if os.path.exists(gender_dir):
            for img_name in os.listdir(gender_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(gender_dir, img_name)
                    train_images.append(img_path)
                    train_labels.append(gender_label)
    
    # For memory constraints, we can limit the training dataset size if needed
    max_train_samples = len(train_images)  # Use full dataset by default
    if max_train_samples < len(train_images):
        indices = np.random.choice(len(train_images), max_train_samples, replace=False)
        train_images = [train_images[i] for i in indices]
        train_labels = [train_labels[i] for i in indices]
        print(f"Limited training set to {max_train_samples} samples")
    
    # Validation dataset
    val_images = []
    val_labels = []
    
    for gender in ['female', 'male']:
        gender_dir = os.path.join(val_dir, gender)
        gender_label = 0 if gender == 'female' else 1
        
        if os.path.exists(gender_dir):
            for img_name in os.listdir(gender_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(gender_dir, img_name)
                    val_images.append(img_path)
                    val_labels.append(gender_label)
    
    # For memory constraints, we can limit the validation dataset size if needed
    max_val_samples = len(val_images)  # Use full dataset by default
    if max_val_samples < len(val_images):
        indices = np.random.choice(len(val_images), max_val_samples, replace=False)
        val_images = [val_images[i] for i in indices]
        val_labels = [val_labels[i] for i in indices]
        print(f"Limited validation set to {max_val_samples} samples")
    
    print(f"Found {len(train_images)} training images and {len(val_images)} validation images")
    
    # Create our datasets
    train_dataset = GenderDataset(train_images, train_labels, train_transform)
    val_dataset = GenderDataset(val_images, val_labels, val_transform)
    
    # Reduce workers for memory constraints
    num_workers = 0 if device.type == 'mps' else 1  # 0 workers = main process does loading
    
    # Create data loaders with memory optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # Disable pin_memory to save RAM
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,  # Keep same as training to avoid memory spikes
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    # Set up our optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Set up learning rate scheduler with warmup
    total_steps = len(train_loader) * num_epochs // grad_accum_steps
    warmup_steps = min(1000, total_steps // 10)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print("Starting training...")
    
    # Track our metrics
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = os.path.join(get_project_root(), "models", "gender_vit_memory_optimized")
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to resume from checkpoint if we have one
    start_epoch = 0
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    
    # Training loop with memory optimizations
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # Clean up memory before each epoch
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        # Zero gradients at the beginning
        optimizer.zero_grad()
        
        for i, (images, labels) in enumerate(progress_bar):
            try:
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(pixel_values=images)
                loss = criterion(outputs.logits, labels)
                loss = loss / grad_accum_steps  # Scale loss for accumulation
                
                # Backward pass
                loss.backward()
                
                # Step every grad_accum_steps or at the end of the epoch
                if (i + 1) % grad_accum_steps == 0 or (i + 1 == len(train_loader)):
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                
                # Track metrics
                running_loss += (loss.item() * grad_accum_steps) * images.size(0)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item() * grad_accum_steps,
                    'acc': 100 * correct / total,
                    'lr': optimizer.param_groups[0]['lr']
                })
                
                # Clear some memory
                del images, labels, outputs, loss, predicted
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: Out of memory in batch {i}. Skipping...")
                    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                        print("MPS out of memory. Collecting garbage and continuing...")
                        # Try to recover
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        optimizer.zero_grad()
                    else:
                        print("CUDA out of memory. Collecting garbage and continuing...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        optimizer.zero_grad()
                else:
                    raise e
        
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = 100 * correct / total if total > 0 else 0
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # Only evaluate on validation set periodically to save time and memory
        if (epoch + 1) % eval_every == 0:
            # Clean up memory before validation
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Validation phase
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
                for images, labels in progress_bar:
                    try:
                        images = images.to(device)
                        labels = labels.to(device)
                        
                        # Forward pass
                        outputs = model(pixel_values=images)
                        loss = criterion(outputs.logits, labels)
                        
                        # Track metrics
                        running_loss += loss.item() * images.size(0)
                        _, predicted = torch.max(outputs.logits, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        
                        # Update progress bar
                        progress_bar.set_postfix({
                            'loss': loss.item(),
                            'acc': 100 * correct / total
                        })
                        
                        # Clear some memory
                        del images, labels, outputs, loss, predicted
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"WARNING: Out of memory in validation. Skipping batch...")
                            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                                print("MPS out of memory during validation. Collecting garbage...")
                                gc.collect()
                            else:
                                print("CUDA out of memory during validation. Collecting garbage...")
                                torch.cuda.empty_cache()
                                gc.collect()
                            continue
                        else:
                            raise e
                
                epoch_val_loss = running_loss / len(val_dataset)
                epoch_val_acc = 100 * correct / total if total > 0 else 0
                val_losses.append(epoch_val_loss)
                val_accs.append(epoch_val_acc)
                
                print(f"Validation - Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%")
                
                # Check if this is the best model so far
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    patience_counter = 0
                    
                    # Save the best model
                    model.save_pretrained(output_dir)
                    processor.save_pretrained(output_dir)
                    print(f"New best model saved! Validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    print(f"No improvement for {patience_counter} evaluations")
        
        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accs': train_accs,
                'val_accs': val_accs
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")
        
        # Early stopping
        if patience_counter >= early_stopping:
            print(f"Early stopping triggered after {patience_counter} evaluations without improvement")
            break
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")
    
    # Save final model
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Training completed! Final model saved to {output_dir}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(range(0, len(val_losses) * eval_every, eval_every), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    if val_accs:
        plt.plot(range(0, len(val_accs) * eval_every, eval_every), val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.show()
    
    return model, processor

def main():
    parser = argparse.ArgumentParser(description="Train Vision Transformer for gender classification with memory optimizations")
    parser.add_argument(
        "--data", "-d", type=str, required=True,
        help="Path to data directory with train and val folders"
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=4,  # Reduced from 8 to 4
        help="Batch size for training (default: 4)"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--lr", "-l", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output directory for saving model and results (default: models/gender_vit_memory_optimized)"
    )
    parser.add_argument(
        "--metal", action="store_true", default=True,
        help="Use Metal on macOS (default: True if available)"
    )
    parser.add_argument(
        "--no-metal", action="store_false", dest="metal",
        help="Don't use Metal on macOS even if available"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=True,
        help="Use CUDA if available (default: True)"
    )
    parser.add_argument(
        "--no-cuda", action="store_false", dest="cuda",
        help="Don't use CUDA even if available"
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU usage (overrides --metal and --cuda options)"
    )
    parser.add_argument(
        "--image-size", type=int, default=160,  # Reduced from 224 to 160
        help="Image size for training (default: 160)"
    )
    parser.add_argument(
        "--grad-accum", type=int, default=8,  # Increased from 4 to 8
        help="Gradient accumulation steps (default: 8)"
    )
    parser.add_argument(
        "--model-size", type=str, default="base", choices=["tiny", "small", "base"],
        help="Model size to use (default: base)"
    )
    parser.add_argument(
        "--eval-every", type=int, default=2,
        help="Evaluate on validation set every N epochs (default: 2)"
    )
    parser.add_argument(
        "--save-every", type=int, default=5,
        help="Save checkpoint every N epochs (default: 5)"
    )
    parser.add_argument(
        "--early-stopping", type=int, default=3,
        help="Early stopping patience (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Override device selection if CPU is forced
    if args.cpu:
        args.metal = False
        args.cuda = False
    
    print("Starting gender classification training...")
    print(f"Data directory: {args.data}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Image size: {args.image_size}")
    print(f"Gradient accumulation steps: {args.grad_accum}")
    print(f"Model size: {args.model_size}")
    
    # Train the model
    model, processor = train_model(
        data_dir=args.data,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output,
        use_metal=args.metal,
        use_cuda=args.cuda,
        image_size=args.image_size,
        grad_accum_steps=args.grad_accum,
        model_size=args.model_size,
        eval_every=args.eval_every,
        save_every=args.save_every,
        early_stopping=args.early_stopping
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()