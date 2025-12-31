#!/usr/bin/env python3
"""
Validate dataset quality
"""
import json
import sys
from pathlib import Path

def validate_dataset(file_path):
    """Validate dataset quality"""
    
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        return None, None
    
    issues = []
    stats = {
        "total": 0,
        "empty_instruction": 0,
        "empty_output": 0,
        "too_short": 0,
        "too_long": 0,
        "invalid_json": 0
    }
    
    print(f"Validating {file_path}...")
    
    with open(file_path) as f:
        for i, line in enumerate(f, 1):
            try:
                item = json.loads(line)
                stats["total"] += 1
                
                # Check required fields
                if not item.get("instruction", "").strip():
                    stats["empty_instruction"] += 1
                    issues.append(f"Line {i}: Empty instruction")
                
                if not item.get("output", "").strip():
                    stats["empty_output"] += 1
                    issues.append(f"Line {i}: Empty output")
                
                # Check length
                output_len = len(item.get("output", ""))
                if output_len < 10:
                    stats["too_short"] += 1
                elif output_len > 4000:
                    stats["too_long"] += 1
                    issues.append(f"Line {i}: Output too long ({output_len} chars)")
                
            except json.JSONDecodeError:
                stats["invalid_json"] += 1
                issues.append(f"Line {i}: Invalid JSON")
    
    # Print report
    print(f"\n{'='*60}")
    print(f"Dataset Validation Report")
    print(f"{'='*60}")
    print(f"File: {file_path}")
    print(f"Total examples: {stats['total']}")
    print(f"Empty instructions: {stats['empty_instruction']}")
    print(f"Empty outputs: {stats['empty_output']}")
    print(f"Too short: {stats['too_short']}")
    print(f"Too long: {stats['too_long']}")
    print(f"Invalid JSON: {stats['invalid_json']}")
    
    # Calculate quality score
    if stats["total"] > 0:
        quality_score = (
            (stats["total"] - stats["empty_instruction"] - stats["empty_output"] - stats["invalid_json"]) 
            / stats["total"] * 100
        )
        print(f"\nQuality Score: {quality_score:.1f}%")
    
    if issues:
        print(f"\n{'='*60}")
        print(f"Issues Found ({len(issues)})")
        print(f"{'='*60}")
        for issue in issues[:10]:  # Show first 10
            print(issue)
        if len(issues) > 10:
            print(f"... and {len(issues) - 10} more issues")
    else:
        print(f"\n✅ No issues found!")
    
    return stats, issues

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/codealpaca.jsonl"
    validate_dataset(file_path)
