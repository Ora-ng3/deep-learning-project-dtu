import pandas as pd
import numpy as np
import subprocess
import os
from pathlib import Path

# Paths
CURRENT_DIR = Path(__file__).parent.resolve()
CSV_PATH = CURRENT_DIR / "window_size_analysis.csv"
TEX_OUTPUT_PATH = CURRENT_DIR / "window_size_analysis.tex"

def generate_tex_content(df_filtered):
    # Prepare data strings for PGFPlots
    raw_data_str = ""
    for _, row in df_filtered.iterrows():
        raw_data_str += f"{row['window_size']} {row['proportion']}\n"
        
    smoothed_data_str = ""
    for _, row in df_filtered.iterrows():
        if not pd.isna(row['smoothed']):
            smoothed_data_str += f"{row['window_size']} {row['smoothed']}\n"

    tex_content = r"""
\documentclass[border=10pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}

\begin{document}
\begin{tikzpicture}
    \begin{axis}[
        width=12cm,
        height=8cm,
        xlabel={Window Size},
        ylabel={TP / (TP + FP + FN)},
        title={True Positive Proportion vs Window Size (FCNN)},
        grid=major,
        legend pos=south east,
        xmin=0, xmax=41,
        ymin=0.6, ymax=0.85, % Adjust based on your data range if needed
        axis lines=left,
    ]
    
    % Raw Data (Points only)
    \addplot[
        only marks,
        mark=*,
        mark size=1.5pt,
        color=black!60,
        opacity=0.6
    ]
    table {
""" + raw_data_str + r"""
    };
    \addlegendentry{Raw Data}
    
    % Smoothed Curve
    \addplot[
        smooth,
        thick,
        color=red
    ]
    table {
""" + smoothed_data_str + r"""
    };
    \addlegendentry{Smoothed (MA-5)}
    
    \end{axis}
\end{tikzpicture}
\end{document}
"""
    return tex_content

def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return

    # Load data
    df = pd.read_csv(CSV_PATH)

    # Filter data (1 to 40)
    df_filtered = df[(df['window_size'] >= 1) & (df['window_size'] <= 40)].copy()

    # Calculate smoothed curve (Moving Average)
    df_filtered['smoothed'] = df_filtered['proportion'].rolling(window=5, center=True, min_periods=1).mean()

    # Generate TeX content
    print("Generating TeX file...")
    tex_content = generate_tex_content(df_filtered)
    
    with open(TEX_OUTPUT_PATH, "w") as f:
        f.write(tex_content)
    print(f"TeX file saved to {TEX_OUTPUT_PATH}")
    
    # Compile with pdflatex
    print("Compiling with pdflatex...")
    try:
        subprocess.run(["pdflatex", TEX_OUTPUT_PATH.name], cwd=CURRENT_DIR, check=True)
        print(f"Successfully generated {TEX_OUTPUT_PATH.stem}.pdf")
        
        # Cleanup
        extensions_to_remove = ['.aux', '.log']
        for ext in extensions_to_remove:
            f = CURRENT_DIR / (TEX_OUTPUT_PATH.stem + ext)
            if f.exists():
                os.remove(f)
                
    except subprocess.CalledProcessError as e:
        print(f"Error compiling LaTeX: {e}")
    except FileNotFoundError:
        print("Error: pdflatex not found. Please ensure TeX Live or MiKTeX is installed.")

if __name__ == "__main__":
    main()
