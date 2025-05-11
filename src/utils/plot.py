import matplotlib.pyplot as plt
import numpy as np
import torch

def imshow(data):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(data), size=(1,)).item()  # Seleciona índice aleatório
        img_tensor, label = data[sample_idx]
        
        # Denormaliza para exibição
        img = img_tensor * 0.5 + 0.5  # Desfaz a normalização (-0.5 a 0.5 para 0 a 1)
        img = img.squeeze(0)  # Remove o canal único

        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis('off')
        plt.imshow(img, cmap='gray')
    plt.show()

def _plot_predictions(model, dataset, device, epoch, plot_dir, num_samples=6):
    """Plot sample predictions with labels"""
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(indices, 1):
        image, label = dataset[idx]
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            _, pred = torch.max(output, 1)
        
        # Denormalize image
        img = image.squeeze().cpu().numpy()
        img = img * 0.5 + 0.5  # Undo normalization
        
        plt.subplot(2, 3, i)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {CFG_ASLNet.LABELS[label]}\nPred: {CFG_ASLNet.LABELS[pred.item()]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(plot_dir/f"predictions_epoch_{epoch}.png")
    plt.close()

def _plot_metrics(train_losses, val_accuracies, plot_dir):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(plot_dir/"training_metrics.png")
    plt.close()

def _plot_confusion_matrix(preds, labels, plot_dir):
    """Plot confusion matrix at end of training"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=CFG_ASLNet.LABELS, 
                yticklabels=CFG_ASLNet.LABELS)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(plot_dir/"confusion_matrix.png")
    plt.close()
