# Run RAG System from command line
# Usage:
#   .\run.ps1                           - Run full pipeline (index + query + compare)
#   .\run.ps1 --query "Your question"   - Quick query with MMR
#   .\run.ps1 --compare "Your question" - Compare Naive vs MMR vs QUBO
#   .\run.ps1 --notebook                - Execute full Jupyter notebook
#   .\run.ps1 --phase A                 - Run indexing only
#   .\run.ps1 --phase C                 - Run diversity comparison only

& "$PSScriptRoot\.venv\Scripts\python.exe" "$PSScriptRoot\run_notebook.py" @args
