#!/usr/bin/env python3
"""
Analysis script for Episodic-Centric Training.
Parses logs from standard output (or log file) and visualizes:
1. Loss trends (CE vs PC)
2. Episodic Memory Usage
3. Surprise Levels
4. Sleep Cycles
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import argparse
import sys

def parse_log_file(log_path):
    """
    Parses a log file containing training output.
    Returns dictionaries of extracted metrics.
    """
    data = {
        'steps': [],
        'ce_loss': [],
        'pc_loss': [],
        'memory_count': [],
        'memory_percent': [],
        'avg_surprise': [],
        'sleep_cycles': []
    }
    
    # Patterns
    # Pre-training: Step 100/500000: Loss=2.7543, CE=2.7340, PC=0.2033 (184.24s)
    pretrain_pattern = re.compile(r"Step (\d+)/\d+: Loss=([\d.]+), CE=([\d.]+), PC=([\d.]+)")
    
    # Personalization:
    # Step 100/10000:
    #   Memories: 50/50000 (0.1%)
    #   Added: 15, Rejected: 17
    #   Avg Surprise: 0.4532, Threshold: 0.8234
    #   Sleep Cycles: 0
    
    current_step = None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Check Pre-training
        match_pre = pretrain_pattern.search(line)
        if match_pre:
            step = int(match_pre.group(1))
            ce = float(match_pre.group(3))
            pc = float(match_pre.group(4))
            
            data['steps'].append(step)
            data['ce_loss'].append(ce)
            data['pc_loss'].append(pc)
            continue
            
        # Check Personalization Step Header
        if line.startswith("Step") and "/" in line and ":" in line:
            try:
                parts = line.split()[1].split('/')
                current_step = int(parts[0])
            except:
                pass
            continue
            
        # Check Memory stats (next lines)
        if current_step is not None:
            if line.startswith("Memories:"):
                # Memories: 50/50000 (0.1%)
                parts = line.split()
                count = int(parts[1].split('/')[0])
                percent = float(parts[2].strip('()%'))
                
                # Check for existing entry to avoid dupes if logs are messy
                # A simple way is to just append, assuming log order
                data['memory_count'].append((current_step, count))
                data['memory_percent'].append((current_step, percent))
                
            elif line.startswith("Avg Surprise:"):
                # Avg Surprise: 0.4532, Threshold: ...
                parts = line.split()
                surp = float(parts[2].strip(','))
                data['avg_surprise'].append((current_step, surp))
                
            elif line.startswith("Sleep Cycles:"):
                # Sleep Cycles: 0
                count = int(line.split()[2])
                data['sleep_cycles'].append((current_step, count))
                current_step = None # Reset
                
    return data

def plot_analysis(data, save_path='analysis_results.png'):
    """Create visualization grid."""
    
    has_pretrain = len(data['ce_loss']) > 0
    has_personal = len(data['memory_count']) > 0
    
    rows = 0
    if has_pretrain: rows += 1
    if has_personal: rows += 2
    
    if rows == 0:
        print("No valid data found to plot.")
        return
        
    fig = plt.figure(figsize=(12, 5 * rows))
    
    plot_idx = 1
    
    # 1. Pre-training Loss
    if has_pretrain:
        ax1 = fig.add_subplot(rows, 1, plot_idx)
        steps = data['steps']
        ax1.plot(steps, data['ce_loss'], label='CE Loss', alpha=0.6)
        ax1.plot(steps, data['pc_loss'], label='PC Loss', alpha=0.6)
        ax1.set_title("Pre-training Loss")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plot_idx += 1
        
    # 2. Personalization - Memory Growth
    if has_personal:
        ax2 = fig.add_subplot(rows, 1, plot_idx)
        mem_steps, mem_counts = zip(*data['memory_count'])
        ax2.plot(mem_steps, mem_counts, 'g-', label='Episodic Memories')
        ax2.set_title("Episodic Memory Growth")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Memory Count")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plot_idx += 1
        
        # 3. Personalization - Surprise & Sleep
        ax3 = fig.add_subplot(rows, 1, plot_idx)
        surp_steps, surp_vals = zip(*data['avg_surprise'])
        sleep_steps, sleep_vals = zip(*data['sleep_cycles'])
        
        ax3.plot(surp_steps, surp_vals, 'orange', label='Avg Surprise')
        ax3.set_ylabel("Surprise (PC Loss)")
        
        # Twin axis for sleep cycles
        ax3b = ax3.twinx()
        ax3b.plot(sleep_steps, sleep_vals, 'b--', label='Sleep Cycles')
        ax3b.set_ylabel("Cumulative Sleep Cycles")
        
        ax3.set_title("Surprise Levels & Sleep Consolidation")
        ax3.set_xlabel("Steps")
        
        lines, labels = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines + lines2, labels + labels2, loc='upper left')
        ax3.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ Analysis plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Episodic Training Logs")
    parser.add_argument('logfile', type=str, help='Path to log file (or captured stdout)')
    args = parser.parse_args()
    
    if not Path(args.logfile).exists():
        print(f"Error: File {args.logfile} not found.")
        sys.exit(1)
        
    print(f"Analyzing {args.logfile}...")
    data = parse_log_file(args.logfile)
    
    print(f"Found {len(data['steps'])} pre-training steps.")
    print(f"Found {len(data['memory_count'])} personalization steps.")
    
    plot_analysis(data)

if __name__ == "__main__":
    main()
