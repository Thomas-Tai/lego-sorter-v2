# Lego Sorter V2

An AI-powered sorter for LEGO Spike Essential parts, developed following professional project management and software engineering practices based on Google's standards.

## 🎯 Project North Star

To deliver a commercial-grade, open-source Lego sorting machine that not only solves a real-world problem but also serves as a comprehensive portfolio piece showcasing excellence in full-stack engineering and disciplined project management.

## ✨ Features

- **Automated Sorting:** Sorts LEGO Spike Essential small parts with >85% accuracy.
- **Modular Architecture:** Built with a scalable, testable, and maintainable software architecture.
- **Data Pipeline:** Includes a robust tool to process official Rebrickable data into a project-specific database.
- **(...more to come)**

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Git

### 1. Installation

1. Clone the repository:
    
    ```
    git clone https://github.com/Thomas-Tai/lego-sorter-v2.git
    cd lego-sorter-v2
    
    ```
    
2. Create and activate a virtual environment:
    
    ```
    # For Windows:
    python -m venv venv
    venv\Scripts\activate
    
    # For Mac/Linux:
    python3.11 -m venv venv
    source venv/bin/activate
    
    ```
    
3. Install dependencies:
    
    ```
    pip install -r requirements.txt
    
    ```
    

### 2. Data Setup (Important!)

This project relies on a local SQLite database built from the official Rebrickable data. You must run the data importer tool at least once to generate this database.

1. **Download the Rebrickable CSV Data:**
    - Go to the official download page: [**https://rebrickable.com/downloads/**](https://rebrickable.com/downloads/)
    - Download the following compressed files: `sets.csv.gz`, `inventories.csv.gz`, `inventory_parts.csv.gz`, `parts.csv.gz`, `colors.csv.gz`.
2. **Prepare the Data:**
    - Unzip all the downloaded `.gz` files.
    - Place all the resulting `.csv` files inside the `data/raw/` directory in this project.
3. **Run the Importer:**
    - Ensure your virtual environment is activated.
    - Run the execution script from the project's root directory:
        
        ```
        # For Windows:
        .\venv\Scripts\python.exe run_importer.py
        
        # For Mac/Linux:
        python run_importer.py
        
        ```
        
    - This script will create the `data/processed/lego_parts.sqlite` file, which is required for all subsequent steps.

## Usage

(To be filled in after the main application is developed)

## Development

To install development tools (linters, testers), run:

```
pip install -r requirements-dev.txt

```

To run tests:

```
# For Windows:
.\venv\Scripts\python.exe -m pytest

# For Mac/Linux:
python -m pytest

```