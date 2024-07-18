import csv
import os
import time
import torch
import heapq
import threading
import pynvml
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from concurrent.futures import ThreadPoolExecutor
import re
import random

class Task:
    def __init__(self, task_id, model_type, dataset, batch_size, start_time, deadline, data_size, pruning_amount):
        self.task_id = task_id
        self.model_type = model_type
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.start_time = int(start_time)
        self.deadline = int(deadline)
        self.data_size = int(data_size)
        self.pruning_amount = pruning_amount
        self.priority = float('inf')
        self.variant = None
        self.missed_deadline = None

    def __lt__(self, other):
        return self.priority < other.priority

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()
    return model

def preload_dataset(data_directory, model_type, data_size):
    if model_type == 'vit_b_16':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])

    full_test_set = datasets.CIFAR10(root=data_directory, train=False, download=False, transform=transform)
    indices = list(range(len(full_test_set)))
    if data_size < len(indices):
        indices = indices[:data_size]
    full_subset = torch.utils.data.Subset(full_test_set, indices)

    random_index = random.randint(0, len(full_test_set) - 1)
    single_image_subset = torch.utils.data.Subset(full_test_set, [random_index])

    return full_subset, single_image_subset

def create_data_loader_from_preloaded(preloaded_data, batch_size):
    return DataLoader(preloaded_data, batch_size=batch_size, shuffle=False, num_workers=0)

def check_gpu_resources(threshold=0.9):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        total_memory = mem_info.total
        used_memory = mem_info.used
        memory_usage = used_memory / total_memory
        gpu_utilization = util_info.gpu / 100.0
        if memory_usage < threshold and gpu_utilization < threshold:
            pynvml.nvmlShutdown()
            return True
    pynvml.nvmlShutdown()
    return False

def monitor_scheduler(start_time, task_waitlist, task_scheduler_queue, interval=20):
    while True:
        current_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        for task in task_waitlist[:]:  # Create a copy of the list for safe iteration
            if task.start_time <= current_time:
                task.priority = float('inf')  # Assign a random priority
                task_scheduler_queue.append(task)
                task_waitlist.remove(task)
        num_tasks_remaining = len(task_scheduler_queue)
        time.sleep(interval)

def read_task_definitions(csv_file_path):
    tasks = []
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            start_time = int(row['start_time_ms'])
            if start_time > 1e12:  # Assuming 1e12 as a threshold for invalid start time
                start_time = 0  # Resetting invalid start time to 0
            pruning_amount = float(row.get('pruning_amount', 0))
            task = Task(row['task_id'], row['model_type'], row['dataset'], row['batch_size'], start_time, row['deadline_ms'], row['data_size'], pruning_amount)
            tasks.append(task)
    return tasks

def create_cuda_streams():
    high_priority_stream = torch.cuda.Stream(priority=-1)
    low_priority_stream = torch.cuda.Stream(priority=0)
    return high_priority_stream, low_priority_stream

def warm_up_model(model, data_loader, device, streams):
    stream = streams[0]
    with torch.cuda.stream(stream):
        with torch.no_grad():
            for _ in range(10):  # Run 10 warm-up iterations
                for images, _ in data_loader:
                    images = images.to(device)
                    outputs = model(images)
                    break  # Only do this for the first batch
    torch.cuda.synchronize(stream)  # Ensure all warm-up tasks on the stream are complete


def execute_task(task, models, preloaded_datasets, results_file, scheduler_start_time, streams, stream_priority):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models[task.model_type][task.pruning_amount]
    data_loader = create_data_loader_from_preloaded(preloaded_datasets[task.model_type]['full_loader'], task.batch_size)
    stream = streams[stream_priority]

    # Warm-up runs
    # for _ in range(10):
    #     with torch.cuda.stream(stream):
    #         with torch.no_grad():
    #             for images, _ in data_loader:
    #                 images = images.to(device)
    #                 outputs = model(images)
    #                 break  # Only do this for the first batch

    torch.cuda.synchronize(stream)  # Ensure warm-up tasks on the stream are complete

    start_time = time.time()
    with torch.cuda.stream(stream):
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(device)
                outputs = model(images)
    torch.cuda.synchronize(stream)  # Ensure all tasks on the stream are complete

    elapsed_time = time.time() - (scheduler_start_time + task.start_time / 1000)
    #elapsed_time = time.time() - start_time
    elapsed_time_ms = elapsed_time * 1000
    actual_start_time = (start_time - scheduler_start_time) * 1000
    task.missed_deadline = elapsed_time_ms > task.deadline

    results = {
        'task_id': task.task_id,
        'model_type': task.model_type,
        'dataset': task.dataset,
        'batch_size': task.batch_size,
        'start_time': task.start_time,
        'actual_start_time': actual_start_time,
        'deadline': task.deadline,
        'elapsed_time_ms': elapsed_time_ms,
        'missed_deadline': task.missed_deadline,
        'model_variant': task.variant,
        'pruning_amount': task.pruning_amount,
        'data_size': task.data_size
    }

    with open(results_file, 'a', newline='') as csvfile:
        fieldnames = ['task_id', 'model_type', 'dataset', 'batch_size', 'start_time', 'actual_start_time', 'deadline', 'elapsed_time_ms', 'missed_deadline', 'model_variant', 'pruning_amount', 'data_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(results)

    return task  # Return the task after processing

def write_to_csv(results, results_file):
    with open(results_file, 'a', newline='') as csvfile:
        fieldnames = ['task_id', 'model_type', 'dataset', 'batch_size', 'start_time', 'actual_start_time', 'deadline', 'elapsed_time_ms', 'missed_deadline', 'model_variant', 'pruning_amount', 'data_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(results)

def scan_models(models_dir):
    model_details = {}
    for folder in os.listdir(models_dir):
        folder_path = os.path.join(models_dir, folder)
        if os.path.isdir(folder_path):
            model_details[folder] = [file for file in os.listdir(folder_path) if file.endswith('.pth')]
    return model_details

def preload_models_and_data(models_dir, data_dir, device, streams):
    model_details = scan_models(models_dir)
    
    preloaded_models = {}
    preloaded_datasets = {}

    for folder, files in model_details.items():
        preloaded_models[folder] = {}
        for file_name in files:
            model_path = os.path.join(models_dir, folder, file_name)
            model = load_model(model_path, device)
            pruning_amount = 0 if "original" in file_name else float(re.search(r'pruned_(\d+\.\d+)', file_name).group(1))
            preloaded_models[folder][pruning_amount] = model
        
        full_dataset, single_image_loader = preload_dataset(data_dir, folder, 1)
        preloaded_datasets[folder] = {
            'full_loader': full_dataset,
            'single_image_loader': single_image_loader
        }

        # Perform warm-up runs
        warm_up_model(model, create_data_loader_from_preloaded(single_image_loader, 1), device, streams)

    return preloaded_models, preloaded_datasets

def main(task_definitions_file, models_dir, results_file):
    task_waitlist = read_task_definitions(task_definitions_file)
    task_scheduler_queue = []
    results = []

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['task_id', 'model_type', 'dataset', 'batch_size', 'start_time', 'actual_start_time', 'deadline', 'elapsed_time_ms', 'missed_deadline', 'model_variant', 'pruning_amount', 'data_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    high_priority_stream, low_priority_stream = create_cuda_streams()
    streams = {-1: high_priority_stream, 0: low_priority_stream}

    # Preload models and data
    models, preloaded_datasets = preload_models_and_data(models_dir, './data', device, streams)

    # Start the scheduler monitor
    scheduler_start_time = time.time()
    monitor_thread = threading.Thread(target=monitor_scheduler, args=(scheduler_start_time, task_waitlist, task_scheduler_queue))
    monitor_thread.daemon = True
    monitor_thread.start()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        current_high_priority_task = None

        while task_waitlist or task_scheduler_queue or futures:
            current_time = (time.time() - scheduler_start_time) * 1000

            for task in task_waitlist[:]:
                if task.start_time <= current_time:
                    task.priority = float('inf')  # Assign random priority
                    task_scheduler_queue.append(task)
                    task_waitlist.remove(task)

            if task_scheduler_queue:
                if current_high_priority_task is None:
                    next_task = heapq.heappop(task_scheduler_queue)
                    if check_gpu_resources():
                        try:
                            current_high_priority_task = next_task
                            futures.append(executor.submit(execute_task, next_task, models, preloaded_datasets, results_file, scheduler_start_time, streams, -1))
                        except RuntimeError:
                            heapq.heappush(task_scheduler_queue, next_task)
                    else:
                        heapq.heappush(task_scheduler_queue, next_task)
                else:
                    for i in range(len(task_scheduler_queue)):
                        task = task_scheduler_queue[i]
                        if task.priority >= current_high_priority_task.priority and check_gpu_resources():
                            try:
                                futures.append(executor.submit(execute_task, task, models, preloaded_datasets, results_file, scheduler_start_time, streams, 0))
                                task_scheduler_queue.pop(i)
                                break
                            except RuntimeError:
                                continue

            for future in futures[:]:
                if future.done():
                    task = future.result()
                    if task == current_high_priority_task:
                        current_high_priority_task = None
                    results.append(task)
                    futures.remove(future)
            # very important metric now i don't know how to change 
            time.sleep(0.008)

        total_tasks = len(results)
        missed_count = sum(1 for result in results if result.missed_deadline)
        deadline_miss_rate = (missed_count / total_tasks) * 100 if total_tasks > 0 else 0

        summary_results = {
            'total_tasks': total_tasks,
            'tasks_met_deadline': total_tasks - missed_count,
            'tasks_missed_deadline': missed_count,
            'deadline_miss_rate': deadline_miss_rate
        }

        with open(results_file, 'a', newline='') as csvfile:
            csvfile.write('\n')
            writer = csv.DictWriter(csvfile, fieldnames=summary_results.keys())
            writer.writeheader()
            writer.writerow(summary_results)

        print("\nFinal Results:")
        print(f"Total tasks: {total_tasks}")
        print(f"Tasks that met the deadline: {total_tasks - missed_count}")
        print(f"Tasks that missed the deadline: {missed_count}")
        print(f"Deadline Miss Rate: {deadline_miss_rate:.2f}%")

if __name__ == "__main__":
    main('./task_set2.csv', './models', './results.csv')
