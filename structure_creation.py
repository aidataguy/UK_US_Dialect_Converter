import os
import nbformat as nbf

def create_project_structure():
    """Create the project directory structure."""
    
    # Define the directory structure
    directories = [
        'data/raw',
        'data/processed',
        'notebooks',
        'models',
        'src/config',
        'src/data',
        'src/models',
        'src/training',
        'src/utils'
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    # Create __init__.py files
    init_locations = [
        'src',
        'src/config',
        'src/data',
        'src/models',
        'src/training',
        'src/utils'
    ]
    
    for location in init_locations:
        init_file = os.path.join(location, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Initialize package\n')

    print("Project structure created successfully!")

def create_notebook(file_path):
    nb = nbf.v4.new_notebook()
    
    # Notebook content with placeholders
    cells = [
        nbf.v4.new_markdown_cell("""# Dialect Conversion Model

This notebook implements a model to convert text between UK and US dialects."""),
        
        nbf.v4.new_markdown_cell("""## 1. Data Loading & Preprocessing"""),
        nbf.v4.new_code_cell("""# Load necessary libraries
import pandas as pd

# Load dataset
df = pd.read_csv('../data/sample_data.csv')
df.head()"""),
        
        nbf.v4.new_markdown_cell("""## 2. Model Selection & Justification"""),
        nbf.v4.new_markdown_cell("""We will use a Transformer-based model such as T5 or fine-tuned GPT for dialect conversion."""),
        
        nbf.v4.new_markdown_cell("""## 3. Model Training"""),
        nbf.v4.new_code_cell("""# Define and train model"""),
        
        nbf.v4.new_markdown_cell("""## 4. Model Evaluation"""),
        nbf.v4.new_code_cell("""# Evaluate model performance"""),
        
        nbf.v4.new_markdown_cell("""## 5. Inference"""),
        nbf.v4.new_code_cell("""# Function to perform dialect conversion
def convert_dialect(text):
    return text  # Placeholder"""),
        
        nbf.v4.new_markdown_cell("""## 6. Conclusion & Next Steps"""),
    ]
    
    nb.cells.extend(cells)
    
    with open(file_path, 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_project_structure()