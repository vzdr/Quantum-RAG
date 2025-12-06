@echo off
REM Run RAG System from command line
REM
REM QUICK START (uses existing indexed data):
REM   run --query "Who is Frodo?"           - Quick query with MMR
REM   run --compare "What is the One Ring?" - Compare Naive vs MMR vs QUBO
REM
REM FULL PIPELINE:
REM   run                           - Run full pipeline (index + query + compare)
REM   run --phase A                 - Run indexing only
REM   run --phase C                 - Run diversity comparison only
REM   run --notebook                - Execute full Jupyter notebook (slow!)

.venv\Scripts\python.exe run_notebook.py %*
