import pandas as pd

# Define task parameters with smaller deadline increments
task_params = {
    'resnet50': {'pruning_amounts': [0, 0.52, 0.56, 0.6, 0.64, 0.72], 'deadlines': [100,200,300,400,500, 600,700,800,900,1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]},
    'mobilenet_v3_large': {'pruning_amounts': [0, 0.56, 0.6, 0.72, 0.76, 0.8], 'deadlines': [100,200,300,400,500, 600,700,800,900,1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]},
    'vgg16': {'pruning_amounts': [0, 0.04, 0.24, 0.44, 0.72, 0.8], 'deadlines': [100,200,300,400,500, 600,700,800,900,1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]},
    'vit_b_16': {'pruning_amounts': [0, 0.04, 0.36, 0.48, 0.64, 0.76], 'deadlines': [100,200,300,400,500, 600,700,800,900,1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]}
}

# Initialize task list
tasks = []

# Create tasks
task_id = 1
start_time_ms = 500
for model_type, params in task_params.items():
    for pruning_amount in params['pruning_amounts']:
        for deadline in params['deadlines']:
            tasks.append({
                'task_id': task_id,
                'model_type': model_type,
                'dataset': 'cifar10',
                'batch_size': 1000,
                'start_time_ms': start_time_ms,
                'deadline_ms': deadline,
                'data_size': 1,
                'pruning_amount': pruning_amount
            })
            task_id += 1
            start_time_ms += 10000  # Increment start time by 10 seconds (10000ms)

# Convert to DataFrame
tasks_df = pd.DataFrame(tasks)

# Save to CSV
tasks_df.to_csv('task_set.csv', index=False)
print("Task set saved to task_set.csv")
