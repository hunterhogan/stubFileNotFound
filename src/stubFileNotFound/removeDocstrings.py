"""AI generated without edits, ugly, and IDC."""
import os
import re

def remove_docstrings_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove triple-quoted docstrings (both """ and ''')
    # This regex handles multiline docstrings
    content = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
    content = re.sub(r"'''.*?'''", '', content, flags=re.DOTALL)

    # Clean up any resulting blank lines (optional)
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pyi'):
                filepath = os.path.join(root, file)
                print(f"Processing {filepath}")
                remove_docstrings_from_file(filepath)

# Run it
process_directory("./stubs/Z0Z_")
print("Done!")
