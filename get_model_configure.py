import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time
import os
import csv
import re
from ptflops import get_model_complexity_info
from typing import Tuple, Any, Callable

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def load_cifar10_data(data_folder, model_folder, batch_size, vit_16_using):
    if vit_16_using:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])

    test_set = datasets.CIFAR10(root=data_folder, train=False, download=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return test_loader

def test_single_image_inference(model, device, data_folder, model_folder, vit_16_using):
    # Load one image from CIFAR10 dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)) if vit_16_using else transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    
    test_set = datasets.CIFAR10(root=data_folder, train=False, download=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    # Warm-up run
    # for _ in range(10):
    #     for images, _ in test_loader:
    #         images = images.to(device)
    #         _ = model(images)
    #         break
    
    # Get a single image
    single_image, _ = next(iter(test_loader))
    single_image = single_image.to(device)

    # Measure single image inference time
    with torch.no_grad():
        torch.cuda.synchronize()  # Synchronize CUDA operations
        start_time = time.time()
        _ = model(single_image)
        torch.cuda.synchronize()  # Synchronize CUDA operations
        end_time = time.time()

    single_image_inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

    return single_image_inference_time

def test_model_accuracy(model, device, data_loader):
    correct = 0
    total = 0
    total_inference_time = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_inference_time_per_image = (total_inference_time / total) * 1000  # Convert to milliseconds
    return accuracy, avg_inference_time_per_image, total_inference_time, total

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb

def get_macs_and_params(model, input_size):
    macs, params = get_model_complexity_info(model, input_size, as_strings=False,
                                             print_per_layer_stat=False, verbose=False)
    return macs, params

def write_result_to_csv(model_folder, variant, single_image_inference_time, accuracy, avg_inference_time_per_image, model_size, macs, total_inference_time, total_labels, models_dir):
    filename = os.path.join(models_dir, model_folder, f"{model_folder}_inference_results.csv")
    file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['Model', 'Variant', 'Pruning Amount', 'Single Image Inference Time (ms)', 'Accuracy (%)', 'Avg Inference Time per Image (ms)', 'Model Size (MB)', 'MACs', 'Total Inference Time (s)', 'Total Labels']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        # Extract pruning amount or check if it's the original model
        pruning_match = re.search(r'pruned_(\d+\.\d+)', variant)
        pruning_amount = pruning_match.group(1) if pruning_match else 'original'
        
        writer.writerow({
            'Model': model_folder, 
            'Variant': variant, 
            'Pruning Amount': pruning_amount, 
            'Single Image Inference Time (ms)': f"{single_image_inference_time:.3f}", 
            'Accuracy (%)': accuracy,
            'Avg Inference Time per Image (ms)': f"{avg_inference_time_per_image:.3f}",
            'Model Size (MB)': f"{model_size:.3f}",
            'MACs': macs,
            'Total Inference Time (s)': f"{total_inference_time:.3f}",
            'Total Labels': total_labels
        })

def scan_models(models_dir):
    model_details = {}
    for folder in os.listdir(models_dir):
        folder_path = os.path.join(models_dir, folder)
        if os.path.isdir(folder_path):
            model_details[folder] = [file for file in os.listdir(folder_path) if file.endswith('.pth')]
    return model_details

def preload_models_and_data(models_dir, data_dir, device, batch_size):
    model_details = scan_models(models_dir)
    
    preloaded_models = {}
    preloaded_datasets = {}

    for folder, files in model_details.items():
        preloaded_models[folder] = {}
        for file_name in files:
            model_path = os.path.join(models_dir, folder, file_name)
            model = load_model(model_path, device)
            preloaded_models[folder][file_name] = model
        
        test_data_loader = load_cifar10_data(data_dir, folder, batch_size, folder == 'vit_b_16')
        preloaded_datasets[folder] = test_data_loader

    return preloaded_models, preloaded_datasets

def main(models_dir, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 100  # For full test set accuracy
    preloaded_models, preloaded_datasets = preload_models_and_data(models_dir, data_dir, device, batch_size)

    # Measure inference time for a single image for all models first
    for folder, models in preloaded_models.items():
        for file_name, model in models.items():
            # Measure inference time for a single image
            single_image_inference_time = test_single_image_inference(model, device, data_dir, folder, folder == 'vit_b_16')
            
            # Measure accuracy and average inference time using the full test set
            full_test_loader = preloaded_datasets[folder]
            accuracy, avg_inference_time_per_image, total_inference_time, total_labels = test_model_accuracy(model, device, full_test_loader)
            
            model_size = get_model_size(model)
            macs, _ = get_macs_and_params(model, (3, 224, 224) if folder == 'vit_b_16' else (3, 32, 32))
            write_result_to_csv(
                folder, file_name, single_image_inference_time, accuracy, avg_inference_time_per_image, model_size, macs, total_inference_time, total_labels, models_dir
            )
            print(f"Single image inference time: {single_image_inference_time:.3f} ms, Accuracy: {accuracy:.2f}%, Avg Inference Time per Image: {avg_inference_time_per_image:.3f} ms, Total Inference Time: {total_inference_time:.3f} s, Total Labels: {total_labels}")

if __name__ == "__main__":
    models_directory = "./models"
    data_directory = "./data"
    main(models_directory, data_directory)
