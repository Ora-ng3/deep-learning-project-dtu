import os
import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_diagram.py <path_to_python_script>")
        print("Example: python generate_diagram.py pyexamples/test_simple.py")
        return

    script_path = Path(sys.argv[1]).resolve()
    if not script_path.exists():
        print(f"Error: File {script_path} not found.")
        return

    script_dir = script_path.parent
    script_name = script_path.stem # e.g. test_simple

    print(f"Processing {script_name} in {script_dir}...")

    # 1. Run the python script to generate .tex
    # We need to run it from its directory so the relative imports (sys.path.append('../')) work
    try:
        print(f"Running python {script_path.name}...")
        subprocess.run([sys.executable, script_path.name], cwd=script_dir, check=True)
        print(f"Successfully generated {script_name}.tex")
    except subprocess.CalledProcessError as e:
        print(f"Error running python script: {e}")
        return

    # 2. Run pdflatex
    tex_file = script_name + ".tex"
    try:
        print(f"Running pdflatex {tex_file}...")
        # Run pdflatex
        subprocess.run(["pdflatex", tex_file], cwd=script_dir, check=True)
        print(f"Successfully generated {script_name}.pdf")
    except subprocess.CalledProcessError as e:
        print(f"Error running pdflatex: {e}")
        return
    except FileNotFoundError:
        print("Error: pdflatex not found. Please ensure TeX Live or MiKTeX is installed and in your PATH.")
        return

    # 3. Cleanup
    extensions_to_remove = ['.aux', '.log', '.vscodeLog', '.tex']
    print("Cleaning up...")
    for ext in extensions_to_remove:
        f = script_dir / (script_name + ext)
        if f.exists():
            try:
                os.remove(f)
            except OSError as e:
                print(f"Could not remove {f}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
