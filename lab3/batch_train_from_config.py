#!/usr/bin/env python3
"""
Batch training script that reads configurations from JSON file
Usage: python batch_train_from_config.py [--config training_configs.json]
"""
import subprocess
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path


def extract_best_f1_macro(output_dir):
    """Extract best F1 macro score from training results"""
    output_path = Path(output_dir)
    
    # Try to read from best_metrics.json (TIMM models)
    best_metrics_path = output_path / 'best_metrics.json'
    if best_metrics_path.exists():
        try:
            with open(best_metrics_path, 'r') as f:
                metrics = json.load(f)
            if 'best_f1_macro' in metrics:
                return metrics['best_f1_macro']
        except Exception as e:
            print(f"Warning: Could not read best_metrics.json: {e}")
    
    # Try to read from trainer_state.json (HuggingFace Trainer) - in root directory
    trainer_state_path = output_path / 'trainer_state.json'
    if trainer_state_path.exists():
        try:
            with open(trainer_state_path, 'r') as f:
                trainer_state = json.load(f)
            
            # Find best f1_macro from log_history
            best_f1 = 0.0
            for entry in trainer_state.get('log_history', []):
                if 'eval_f1_macro' in entry:
                    best_f1 = max(best_f1, entry['eval_f1_macro'])
            
            if best_f1 > 0:
                return best_f1
        except Exception as e:
            print(f"Warning: Could not read trainer_state.json: {e}")
    
    # Try to read from checkpoint subdirectories (HuggingFace Trainer)
    if output_path.exists():
        checkpoint_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
        
        # Sort by checkpoint number (descending) to get the latest
        checkpoint_dirs.sort(key=lambda x: int(x.name.split('-')[1]), reverse=True)
        
        for checkpoint_dir in checkpoint_dirs:
            trainer_state_path = checkpoint_dir / 'trainer_state.json'
            if trainer_state_path.exists():
                try:
                    with open(trainer_state_path, 'r') as f:
                        trainer_state = json.load(f)
                    
                    # Find best f1_macro from log_history
                    best_f1 = 0.0
                    for entry in trainer_state.get('log_history', []):
                        if 'eval_f1_macro' in entry:
                            best_f1 = max(best_f1, entry['eval_f1_macro'])
                    
                    if best_f1 > 0:
                        return best_f1
                except Exception as e:
                    continue
    
    return None


def run_training(config):
    """Run training with given configuration"""
    print("\n" + "="*80)
    print(f"Starting training: {config['name']}")
    print("="*80)
    print(f"Model: {config['model_name']}")
    print(f"LR: {config['learning_rate']}, BS: {config['batch_size']}, Epochs: {config['num_epochs']}")
    print(f"Resample: {config['use_resample']}, Weights: {config['use_class_weights']}")
    print(f"Output: {config['output_dir']}")
    print("="*80 + "\n")
    
    # Build command
    cmd = [
        sys.executable,
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
    best_f1_macro = None
    
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        status = "SUCCESS"
        print(f"\n✓ Training completed successfully in {elapsed/60:.1f} minutes")
        
        # Extract best F1 macro score
        best_f1_macro = extract_best_f1_macro(config['output_dir'])
        if best_f1_macro is not None:
            print(f"Best F1 Macro: {best_f1_macro:.4f}")
        
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
        'output_dir': config['output_dir'],
        'best_f1_macro': best_f1_macro,
    }


def main():
    parser = argparse.ArgumentParser(description='Batch training from config file')
    parser.add_argument('--config', type=str, default='training_configs.json',
                        help='Path to training configurations JSON file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show configurations without training')
    args = parser.parse_args()
    
    # Load configurations
    config_file = Path(args.config)
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        sys.exit(1)
    
    with open(config_file, 'r') as f:
        data = json.load(f)
    
    all_configs = data['configs']
    # Filter enabled configs
    configs = [c for c in all_configs if c.get('enabled', True)]
    
    print("\n" + "="*80)
    print("BATCH TRAINING FROM CONFIG")
    print("="*80)
    print(f"Config file: {config_file}")
    print(f"Total configurations: {len(all_configs)}")
    print(f"Enabled configurations: {len(configs)}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    if len(configs) == 0:
        print("No enabled configurations found. Enable configs by setting 'enabled': true in JSON.")
        sys.exit(0)
    
    # Print training queue
    print("Training queue:")
    for i, config in enumerate(configs, 1):
        print(f"\n  {i}. {config['name']}")
        print(f"     Model: {config['model_type']} - {config['model_name']}")
        print(f"     LR: {config['learning_rate']}, BS: {config['batch_size']}, Epochs: {config['num_epochs']}")
        print(f"     Resample: {config['use_resample']}, Weights: {config['use_class_weights']}")
        print(f"     Output: {config['output_dir']}")
    
    print("\n" + "="*80)
    
    if args.dry_run:
        print("DRY RUN - No training will be performed")
        print("="*80 + "\n")
        return
    
    input("Press Enter to start training (Ctrl+C to cancel)...")
    
    # Run all trainings
    results = []
    start_time = time.time()
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"Progress: {i}/{len(configs)}")
        print(f"{'='*80}\n")
        
        try:
            result = run_training(config)
            results.append(result)
        except KeyboardInterrupt:
            print("\n\n⚠ Batch training interrupted by user")
            break
        except Exception as e:
            print(f"\n\n✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': config['name'],
                'status': 'ERROR',
                'elapsed_time': 0,
                'output_dir': config['output_dir'],
            })
            # Continue with next configuration
            continue
    
    # Print summary
    total_elapsed = time.time() - start_time
    print("\n\n" + "="*80)
    print("BATCH TRAINING SUMMARY")
    print("="*80)
    print(f"Total time: {total_elapsed/3600:.2f} hours ({total_elapsed/60:.1f} minutes)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults:")
    
    success_count = 0
    for result in results:
        status_symbol = "✓" if result['status'] == "SUCCESS" else "✗"
        f1_str = f", F1: {result['best_f1_macro']:.4f}" if result.get('best_f1_macro') is not None else ""
        print(f"  {status_symbol} {result['name']}: {result['status']} ({result['elapsed_time']/60:.1f} min{f1_str})")
        if result['status'] == "SUCCESS":
            success_count += 1
    
    if len(results) > 0:
        print(f"\nSuccess rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    # Find best model
    best_result = None
    for result in results:
        if result['status'] == "SUCCESS" and result.get('best_f1_macro') is not None:
            if best_result is None or result['best_f1_macro'] > best_result['best_f1_macro']:
                best_result = result
    
    if best_result:
        print(f"\n🏆 Best Model: {best_result['name']}")
        print(f"   F1 Macro: {best_result['best_f1_macro']:.4f}")
        print(f"   Output: {best_result['output_dir']}")
    
    print("="*80 + "\n")
    
    # Save results to file
    results_file = Path('batch_training_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("BATCH TRAINING RESULTS\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {total_elapsed/3600:.2f} hours ({total_elapsed/60:.1f} minutes)\n")
        f.write(f"Success rate: {success_count}/{len(results)}\n\n")
        
        if best_result:
            f.write(f"Best Model: {best_result['name']}\n")
            f.write(f"  F1 Macro: {best_result['best_f1_macro']:.4f}\n")
            f.write(f"  Output: {best_result['output_dir']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("All Results:\n")
        f.write("="*80 + "\n\n")
        
        for result in results:
            f.write(f"{result['name']}: {result['status']} ({result['elapsed_time']/60:.1f} min)\n")
            f.write(f"  Output: {result['output_dir']}\n")
            if result.get('best_f1_macro') is not None:
                f.write(f"  Best F1 Macro: {result['best_f1_macro']:.4f}\n")
            f.write("\n")
    
    print(f"Results saved to: {results_file}")


if __name__ == '__main__':
    main()

