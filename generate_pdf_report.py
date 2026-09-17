#!/usr/bin/env python3
"""
Generate PDF report from markdown documentation.

This script converts the implementation documentation to a professional PDF report.
"""

import os
import sys
from pathlib import Path

def generate_pdf_with_pandoc():
    """Generate PDF using pandoc (if available)."""
    try:
        import subprocess
        
        input_file = "Consensus_Dynamics_Implementation_Report.md"
        output_file = "Consensus_Dynamics_Implementation_Report.pdf"
        
        # Pandoc command with professional styling
        cmd = [
            "pandoc", 
            input_file,
            "-o", output_file,
            "--pdf-engine=xelatex",
            "-V", "geometry:margin=1in",
            "-V", "fontsize=11pt",
            "-V", "documentclass=article",
            "--toc",
            "--toc-depth=3",
            "--number-sections",
            "--highlight-style=tango"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ PDF generated successfully: {output_file}")
            return True
        else:
            print(f"❌ Pandoc failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error with pandoc: {e}")
        return False

def generate_pdf_with_markdown():
    """Generate PDF using markdown and weasyprint (if available)."""
    try:
        import markdown
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
        
        # Read markdown file
        with open("Consensus_Dynamics_Implementation_Report.md", "r") as f:
            markdown_content = f.read()
        
        # Convert to HTML
        html_content = markdown.markdown(
            markdown_content, 
            extensions=['toc', 'codehilite', 'tables', 'fenced_code']
        )
        
        # Add CSS styling
        css_style = """
        @page {
            margin: 1in;
            size: letter;
        }
        
        body {
            font-family: 'Times New Roman', serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Arial', sans-serif;
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }
        
        h1 {
            font-size: 24pt;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        
        h2 {
            font-size: 18pt;
            color: #e74c3c;
            border-bottom: 1px solid #e74c3c;
        }
        
        h3 {
            font-size: 14pt;
            color: #8e44ad;
        }
        
        code {
            font-family: 'Courier New', monospace;
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 10pt;
        }
        
        pre {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
            overflow-x: auto;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        
        .toc {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 15px;
            margin: 20px 0;
        }
        
        blockquote {
            border-left: 4px solid #3498db;
            padding-left: 15px;
            margin: 1em 0;
            color: #555;
        }
        """
        
        # Create complete HTML document
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Consensus Dynamics Implementation Report</title>
            <style>{css_style}</style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Generate PDF
        font_config = FontConfiguration()
        HTML(string=full_html).write_pdf(
            "Consensus_Dynamics_Implementation_Report.pdf",
            stylesheets=[CSS(string=css_style)],
            font_config=font_config
        )
        
        print("✅ PDF generated successfully with weasyprint: Consensus_Dynamics_Implementation_Report.pdf")
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies for weasyprint: {e}")
        return False
    except Exception as e:
        print(f"❌ Error with weasyprint: {e}")
        return False

def generate_pdf_basic():
    """Generate a basic PDF using reportlab."""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib.colors import HexColor
        
        # Read markdown content
        with open("Consensus_Dynamics_Implementation_Report.md", "r") as f:
            content = f.read()
        
        # Create PDF document
        doc = SimpleDocTemplate(
            "Consensus_Dynamics_Implementation_Report.pdf",
            pagesize=letter,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=18
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=HexColor('#2c3e50')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=HexColor('#e74c3c')
        )
        
        # Story (content) list
        story = []
        
        # Parse content and add to story
        lines = content.split('\n')
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('# '):
                # Main title
                if current_paragraph:
                    story.append(Paragraph(' '.join(current_paragraph), styles['Normal']))
                    current_paragraph = []
                title = line[2:].strip()
                story.append(Paragraph(title, title_style))
                story.append(Spacer(1, 12))
                
            elif line.startswith('## '):
                # Section heading
                if current_paragraph:
                    story.append(Paragraph(' '.join(current_paragraph), styles['Normal']))
                    current_paragraph = []
                heading = line[3:].strip()
                story.append(Paragraph(heading, heading_style))
                story.append(Spacer(1, 6))
                
            elif line.startswith('### '):
                # Subsection heading
                if current_paragraph:
                    story.append(Paragraph(' '.join(current_paragraph), styles['Normal']))
                    current_paragraph = []
                subheading = line[4:].strip()
                story.append(Paragraph(subheading, styles['Heading3']))
                
            elif line.startswith('```'):
                # Code block
                if current_paragraph:
                    story.append(Paragraph(' '.join(current_paragraph), styles['Normal']))
                    current_paragraph = []
                # Skip code blocks for basic PDF (would need more complex parsing)
                continue
                
            elif line.startswith('---'):
                # Horizontal rule
                if current_paragraph:
                    story.append(Paragraph(' '.join(current_paragraph), styles['Normal']))
                    current_paragraph = []
                story.append(Spacer(1, 12))
                
            elif line == '':
                # Empty line - end paragraph
                if current_paragraph:
                    story.append(Paragraph(' '.join(current_paragraph), styles['Normal']))
                    story.append(Spacer(1, 6))
                    current_paragraph = []
                    
            else:
                # Regular text
                if line:
                    current_paragraph.append(line)
        
        # Add any remaining paragraph
        if current_paragraph:
            story.append(Paragraph(' '.join(current_paragraph), styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        print("✅ Basic PDF generated successfully with reportlab: Consensus_Dynamics_Implementation_Report.pdf")
        return True
        
    except ImportError as e:
        print(f"❌ Missing reportlab dependency: {e}")
        print("   Install with: pip install reportlab")
        return False
    except Exception as e:
        print(f"❌ Error with reportlab: {e}")
        return False

def main():
    """Try different PDF generation methods in order of preference."""
    
    print("🔄 Generating PDF report from implementation documentation...")
    
    # Check if markdown file exists
    if not Path("Consensus_Dynamics_Implementation_Report.md").exists():
        print("❌ Markdown file not found: Consensus_Dynamics_Implementation_Report.md")
        return False
    
    # Try methods in order of preference
    methods = [
        ("pandoc (best quality)", generate_pdf_with_pandoc),
        ("weasyprint (good quality)", generate_pdf_with_markdown),
        ("reportlab (basic quality)", generate_pdf_basic),
    ]
    
    for method_name, method_func in methods:
        print(f"\n🔄 Trying {method_name}...")
        if method_func():
            print(f"✅ Success! PDF generated using {method_name}")
            
            # Check file size and provide info
            if Path("Consensus_Dynamics_Implementation_Report.pdf").exists():
                size = Path("Consensus_Dynamics_Implementation_Report.pdf").stat().st_size
                print(f"📄 PDF file size: {size/1024:.1f} KB")
                print(f"📂 Location: {Path.cwd()}/Consensus_Dynamics_Implementation_Report.pdf")
            
            return True
    
    print("\n❌ All PDF generation methods failed.")
    print("💡 Alternatives:")
    print("   1. Install pandoc: brew install pandoc")
    print("   2. Install weasyprint: pip install weasyprint")
    print("   3. Install reportlab: pip install reportlab")
    print("   4. Use the markdown file directly with any markdown viewer")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)