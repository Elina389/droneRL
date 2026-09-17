#!/usr/bin/env python3
"""
Simple PDF converter using available system tools.
"""

import os
import sys
import subprocess
from pathlib import Path

def try_wkhtmltopdf():
    """Try using wkhtmltopdf if available."""
    try:
        # Check if wkhtmltopdf is available
        result = subprocess.run(['which', 'wkhtmltopdf'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            return False
        
        # Convert HTML to PDF
        cmd = [
            'wkhtmltopdf',
            '--page-size', 'A4',
            '--margin-top', '0.75in',
            '--margin-right', '0.75in', 
            '--margin-bottom', '0.75in',
            '--margin-left', '0.75in',
            '--encoding', 'UTF-8',
            'Consensus_Dynamics_Implementation_Report.html',
            'Consensus_Dynamics_Implementation_Report.pdf'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PDF generated using wkhtmltopdf")
            return True
        else:
            print(f"❌ wkhtmltopdf failed: {result.stderr}")
            return False
            
    except Exception as e:
        return False

def create_text_summary():
    """Create a text summary as fallback."""
    try:
        # Read the markdown file
        with open("Consensus_Dynamics_Implementation_Report.md", "r") as f:
            content = f.read()
        
        # Create a simple text report
        text_report = f"""
CONSENSUS DYNAMICS IMPLEMENTATION REPORT
========================================

This is a text summary of the consensus dynamics implementation.
For the full formatted version, see:
- HTML: Consensus_Dynamics_Implementation_Report.html  
- Markdown: Consensus_Dynamics_Implementation_Report.md

EXECUTIVE SUMMARY:
=================
Successfully implemented matrix-based consensus dynamics and graph Laplacian 
coordination into existing RL drone swarm algorithms.

KEY ACHIEVEMENTS:
================
✅ Mathematical Framework: Complete implementation of consensus dynamics (ẋ = -Lx)
✅ Graph Theory Integration: Adjacency matrices and Laplacian computation
✅ Enhanced RL Algorithms: PPO with matrix-based coordination for 2D and 3D
✅ Modular Architecture: Separated weights, configurations, and algorithms
✅ Convergence Guarantees: Exponential convergence with rate λ₂

CORE FILES CREATED:
==================
📂 consensus_dynamics.py       - Core mathematical framework
📂 config.py                   - All configuration parameters  
📂 weights_manager.py          - Separated weight management
📂 consensus_rl_algorithms.py  - Enhanced RL algorithms
📂 train_consensus_models.py   - Unified training script

MATHEMATICAL FOUNDATION:
=======================
Adjacency Matrix: aᵢⱼ = 1 if ||xᵢ - xⱼ|| < r, else 0
Graph Laplacian: L = D - A (where D is degree matrix, A is adjacency)
Consensus Dynamics: ẋ = -Lx
Convergence: ||x(t) - x̄|| ≤ e^(-λ₂t)||x(0) - x̄||

USAGE:
======
Train 2D: python train_consensus_models.py --model 2d --consensus
Train 3D: python train_consensus_models.py --model 3d --consensus
Ablation: python train_consensus_models.py --ablation --model both

For complete technical details, mathematical derivations, and implementation 
specifics, please view the HTML or markdown versions of this report.

Generated: {Path.cwd()}
"""
        
        with open("Consensus_Dynamics_SUMMARY.txt", "w") as f:
            f.write(text_report)
        
        print("✅ Text summary created: Consensus_Dynamics_SUMMARY.txt")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create text summary: {e}")
        return False

def main():
    """Main conversion function."""
    print("🔄 Converting implementation report to PDF...")
    
    # Check if HTML exists
    if not Path("Consensus_Dynamics_Implementation_Report.html").exists():
        print("❌ HTML file not found!")
        return False
    
    print(f"📄 HTML file found: {Path('Consensus_Dynamics_Implementation_Report.html').stat().st_size/1024:.1f} KB")
    
    # Try different conversion methods
    if try_wkhtmltopdf():
        if Path("Consensus_Dynamics_Implementation_Report.pdf").exists():
            size = Path("Consensus_Dynamics_Implementation_Report.pdf").stat().st_size
            print(f"📄 PDF created successfully: {size/1024:.1f} KB")
            return True
    
    print("\n💡 PDF conversion failed. Creating alternatives:")
    
    # Create text summary as fallback
    create_text_summary()
    
    print("\n📋 Available formats:")
    for ext in ['.html', '.md', '.txt']:
        filename = f"Consensus_Dynamics_Implementation_Report{ext}"
        if ext == '.txt':
            filename = "Consensus_Dynamics_SUMMARY.txt"
        
        if Path(filename).exists():
            size = Path(filename).stat().st_size
            print(f"   ✅ {filename} ({size/1024:.1f} KB)")
    
    print("\n💡 Suggestions for PDF creation:")
    print("   1. Open the HTML file in Chrome/Safari and Print → Save as PDF")
    print("   2. Install wkhtmltopdf: brew install wkhtmltopdf")
    print("   3. Use online HTML-to-PDF converters")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)