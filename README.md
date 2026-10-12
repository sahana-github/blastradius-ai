# BlastRadius AI

Predicts what could break when a developer changes code, by building a
dependency graph of code, database tables, API endpoints and tests.

## Setup
    pip install -r requirements.txt
    mkdir data
    git clone https://github.com/fastapi/full-stack-fastapi-template.git data/full-stack-fastapi-template

## Usage
    python graph/graph_builder.py build data/full-stack-fastapi-template/backend
    python graph/graph_builder.py impact Item --high