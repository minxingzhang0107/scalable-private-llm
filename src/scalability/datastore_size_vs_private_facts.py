#!/usr/bin/env python3
"""
Analyze all datastores in /usr/xtmp/mz238/ directory (Simple version without FAISS)
For each datastore, extract:
1. Total number of (context, next-token) pairs (from targets file)
2. Total size in bytes
"""

import os
import numpy as np
import re

def get_datastore_size_from_name(datastore_name):
    """
    Extract dataset size from datastore name.
    E.g., 'datastore_100k_A5000' -> '100k'
         'datastore_1m_A5000' -> '1m'
    """
    # Match patterns like 100k, 1m, 820k, etc.
    match = re.search(r'datastore_(\d+[km]?)_', datastore_name)
    if match:
        return match.group(1)
    return "unknown"

def get_directory_size(directory):
    """Calculate total size of directory in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def format_bytes(bytes_size):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def analyze_datastore(datastore_path):
    """
    Analyze a single datastore and return statistics
    """
    faiss_index_path = os.path.join(datastore_path, "datastore.index")
    targets_path = os.path.join(datastore_path, "datastore_targets.npy")
    
    # Check if files exist
    if not os.path.exists(faiss_index_path):
        return None, "FAISS index not found"
    if not os.path.exists(targets_path):
        return None, "Targets file not found"
    
    try:
        # Load targets to get number of pairs
        targets = np.load(targets_path)
        num_pairs = len(targets)
        
        # Get file sizes
        faiss_size = os.path.getsize(faiss_index_path)
        targets_size = os.path.getsize(targets_path)
        
        # Get total directory size
        total_bytes = get_directory_size(datastore_path)
        
        return {
            'num_pairs': num_pairs,
            'total_bytes': total_bytes,
            'total_bytes_formatted': format_bytes(total_bytes),
            'faiss_size': faiss_size,
            'targets_size': targets_size
        }, None
        
    except Exception as e:
        return None, str(e)

def main():
    base_dir = "/usr/xtmp/mz238"
    
    print("=" * 80)
    print("DATASTORE ANALYSIS (Simple Version)")
    print("=" * 80)
    print(f"Base directory: {base_dir}\n")
    
    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        return
    
    # Find all datastore directories
    datastore_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith("datastore_"):
            datastore_dirs.append((item, item_path))
    
    if not datastore_dirs:
        print("❌ No datastore directories found")
        return
    
    # Sort by name
    datastore_dirs.sort()
    
    print(f"Found {len(datastore_dirs)} datastore(s)\n")
    
    # Analyze each datastore
    results = []
    for datastore_name, datastore_path in datastore_dirs:
        dataset_size = get_datastore_size_from_name(datastore_name)
        
        print(f"📊 Analyzing: {datastore_name}")
        print(f"   Dataset size: {dataset_size}")
        
        stats, error = analyze_datastore(datastore_path)
        
        if error:
            print(f"   ❌ Error: {error}\n")
            continue
        
        print(f"   ✅ Number of pairs: {stats['num_pairs']:,}")
        print(f"   ✅ Total size: {stats['total_bytes_formatted']} ({stats['total_bytes']:,} bytes)")
        print(f"      - FAISS index: {format_bytes(stats['faiss_size'])} ({stats['faiss_size']:,} bytes)")
        print(f"      - Targets file: {format_bytes(stats['targets_size'])} ({stats['targets_size']:,} bytes)")
        print()
        
        results.append({
            'name': datastore_name,
            'dataset_size': dataset_size,
            'num_pairs': stats['num_pairs'],
            'total_bytes': stats['total_bytes'],
            'faiss_bytes': stats['faiss_size'],
            'targets_bytes': stats['targets_size']
        })
    
    # Summary table
    if results:
        print("=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Datastore Name':<40} {'Dataset Size':<15} {'Pairs':<15} {'Total Size':<20}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['name']:<40} {result['dataset_size']:<15} {result['num_pairs']:<15,} {format_bytes(result['total_bytes']):<20}")
        
        print("=" * 80)
        
        # Detailed CSV with breakdown
        csv_path = "results/analysis/datastore_analysis_detailed.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'w') as f:
            f.write("Datastore Name,Dataset Size,Number of Pairs,Total Bytes,FAISS Index Bytes,Targets Bytes,Total Size (formatted),FAISS Size (formatted),Targets Size (formatted)\n")
            for result in results:
                f.write(f"{result['name']},{result['dataset_size']},{result['num_pairs']},{result['total_bytes']},{result['faiss_bytes']},{result['targets_bytes']},{format_bytes(result['total_bytes'])},{format_bytes(result['faiss_bytes'])},{format_bytes(result['targets_bytes'])}\n")
        
        print(f"\n✅ Detailed results saved to: {csv_path}")
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("OVERALL STATISTICS")
        print("=" * 80)
        total_pairs = sum(r['num_pairs'] for r in results)
        total_storage = sum(r['total_bytes'] for r in results)
        print(f"Total datastores: {len(results)}")
        print(f"Total pairs across all datastores: {total_pairs:,}")
        print(f"Total storage across all datastores: {format_bytes(total_storage)} ({total_storage:,} bytes)")
        if results:
            avg_pairs = total_pairs / len(results)
            avg_storage = total_storage / len(results)
            print(f"Average pairs per datastore: {avg_pairs:,.0f}")
            print(f"Average storage per datastore: {format_bytes(avg_storage)}")
        print("=" * 80)

if __name__ == "__main__":
    main()