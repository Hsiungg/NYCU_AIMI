#!/usr/bin/env python3
"""
Batch training script for multiple model configurations
Trains different models with different hyperparameters
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Define training configurations
TRAINING_CONFIGS = [
    # DINOv2 configurations
    {
        'name': 'dinov2_base_lr2e5_bs32',
        'model_type': 'dinov2',
        'model_name': 'facebook/dinov2-base',
        'output_dir': './output/dinov2_base_lr2e5_bs32',
        'learning_rate': 2e-5,
        'batch_size': 32,
        'num_epochs': 100,
        'early_stopping_patience': 15,
        'use_resample': True,
        'use_class_weights': True,
    },
    {
        'name': 'dinov2_base_lr1e5_bs32',
        'model_type': 'dinov2',
        'model_name': 'facebook/dinov2-base',
        'output_dir': './output/dinov2_base_lr1e5_bs32',
        'learning_rate': 1e-5,
        'batch_size': 32,
        'num_epochs': 100,
        'early_stopping_patience': 15,
        'use_resample': True,
        'use_class_weights': True,
    },
    {
        'name': 'dinov2_base_lr5e6_bs32',
        'model_type': 'dinov2',
        'model_name': 'facebook/dinov2-base',
        'output_dir': './output/dinov2_base_lr5e6_bs32',
        'learning_rate': 5e-6,
        'batch_size': 32,
        'num_epochs': 100,
        'early_stopping_patience': 15,
        'use_resample': True,
        'use_class_weights': True,
    },
    
    # DINOv2 with different batch sizes
    {
        'name': 'dinov2_base_lr2e5_bs16',
        'model_type': 'dinov2',
        'model_name': 'facebook/dinov2-base',
        'output_dir': './output/dinov2_base_lr2e5_bs16',
        'learning_rate': 2e-5,
        'batch_size': 16,
        'num_epochs': 100,
        'early_stopping_patience': 15,
        'use_resample': True,
        'use_class_weights': True,
    },
    
    # DINOv2 without resampling
    {
        'name': 'dinov2_base_lr2e5_bs32_no_resample',
        'model_type': 'dinov2',
        'model_name': 'facebook/dinov2-base',
        'output_dir': './output/dinov2_base_lr2e5_bs32_no_resample',
        'learning_rate': 2e-5,
        'batch_size': 32,
        'num_epochs': 100,
        'early_stopping_patience': 15,
        'use_resample': False,
        'use_class_weights': True,
    },
    
    # DINOv3 configurations
    {
        'name': 'dinov3_base_lr2e5_bs32',
        'model_type': 'dinov3',
        'model_name': 'vit_base_patch16_dinov3.lvd1689m',
        'output_dir': './output/dinov3_base_lr2e5_bs32',
        'learning_rate': 2e-5,
        'batch_size': 32,
        'num_epochs': 100,
        'early_stopping_patience': 15,
        'use_resample': True,
        'use_class_weights': True,
    },
    {
        'name': 'dinov3_base_lr1e5_bs32',
        'model_type': 'dinov3',
        'model_name': 'vit_base_patch16_dinov3.lvd1689m',
        'output_dir': './output/dinov3_base_lr1e5_bs32',
        'learning_rate': 1e-5,
        'batch_size': 32,
        'num_epochs': 100,
        'early_stopping_patience': 15,
        'use_resample': True,
        'use_class_weights': True,
    },
]


def run_training(config):
    """Run training with given configuration"""
    print("\n" + "="*80)
    print(f"Starting training: {config['name']}")
    print("="*80)
    print(f"Config: {config}")
    print("="*80 + "\n")
    
    # Build command
    cmd = [
        sys.executable,  # Use current Python interpreter
        'train.py',
        '--model_type', config['model_type'],
        '--model_name', config['model_name'],
        '--output_dir', config['output_dir'],
        '--learning_rate', str(config['learning_rate']),
        '--batch_size', str(config['batch_size']),
        '--num_epochs', str(config['num_epochs']),
        '--early_stopping_patience', str(config['early_stopping_patience']),
        '--use_wandb',
        '--wandb_run_name', config['name'],
    ]
    
    # Add optional flags
    if config['use_resample']:
        cmd.append('--use_resample')
    else:
        cmd.append('--no_resample')
    
    if config['use_class_weights']:
        cmd.append('--use_class_weights')
    else:
        cmd.append('--no_class_weights')
    
    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        status = "SUCCESS"
        print(f"\n✓ Training completed successfully in {elapsed/60:.1f} minutes")
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        status = "FAILED"
        print(f"\n✗ Training failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        status = "INTERRUPTED"
        print(f"\n⚠ Training interrupted after {elapsed/60:.1f} minutes")
        raise
    
    return {
        'name': config['name'],
        'status': status,
        'elapsed_time': elapsed,
    }


def main():
    """Main batch training function"""
    print("\n" + "="*80)
    print("BATCH TRAINING SCRIPT")
    print("="*80)
    print(f"Total configurations to train: {len(TRAINING_CONFIGS)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Print all configurations
    print("Training queue:")
    for i, config in enumerate(TRAINING_CONFIGS, 1):
        print(f"  {i}. {config['name']}")
        print(f"     - Model: {config['model_name']}")
        print(f"     - LR: {config['learning_rate']}, BS: {config['batch_size']}")
        print(f"     - Resample: {config['use_resample']}, Weights: {config['use_class_weights']}")
    
    print("\n" + "="*80)
    input("Press Enter to start training (Ctrl+C to cancel)...")
    
    # Run all trainings
    results = []
    start_time = time.time()
    
    for i, config in enumerate(TRAINING_CONFIGS, 1):
        print(f"\n{'='*80}")
        print(f"Training {i}/{len(TRAINING_CONFIGS)}")
        print(f"{'='*80}\n")
        
        try:
            result = run_training(config)
            results.append(result)
        except KeyboardInterrupt:
            print("\n\n⚠ Batch training interrupted by user")
            break
        except Exception as e:
            print(f"\n\n✗ Unexpected error: {e}")
            results.append({
                'name': config['name'],
                'status': 'ERROR',
                'elapsed_time': 0,
            })
            # Continue with next configuration
            continue
    
    # Print summary
    total_elapsed = time.time() - start_time
    print("\n\n" + "="*80)
    print("BATCH TRAINING SUMMARY")
    print("="*80)
    print(f"Total time: {total_elapsed/3600:.2f} hours")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")
    
    success_count = 0
    for result in results:
        status_symbol = "✓" if result['status'] == "SUCCESS" else "✗"
        print(f"  {status_symbol} {result['name']}: {result['status']} ({result['elapsed_time']/60:.1f} min)")
        if result['status'] == "SUCCESS":
            success_count += 1
    
    print(f"\nSuccess rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    print("="*80 + "\n")
    
    # Save results to file
    results_file = Path('batch_training_results.txt')
    with open(results_file, 'w') as f:
        f.write("BATCH TRAINING RESULTS\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_elapsed/3600:.2f} hours\n")
        f.write(f"Success rate: {success_count}/{len(results)}\n\n")
        for result in results:
            f.write(f"{result['name']}: {result['status']} ({result['elapsed_time']/60:.1f} min)\n")
    
    print(f"Results saved to: {results_file}")


if __name__ == '__main__':
    main()

