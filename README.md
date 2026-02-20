---
title: Climate Policy Advisor RAG
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# 🌍 Climate Policy Advisor RAG

A Retrieval-Augmented Generation (RAG) application that acts as a **Professional Environmental Policy Advisor**.

## Overview
This assistant analyzes historical and projected climate data across multiple countries to provide structured, analytical policy recommendations. It focuses on governance strategies, mitigation/adaptation efforts, and data-driven reasoning.

## How it Works
1.  **Retrieval**: Uses **FAISS** to find relevant country-specific climate metrics from a dataset of 1,000+ records.
2.  **Generation**: Uses **Google Flan-T5** to synthesize a professional advisory report based on the retrieved data and expert policy guidelines.

## Dataset Features
- **Average Temperature**
- **CO2 Emissions**
- **Sea Level Rise**
- **Rainfall**
- **Population**
- **Renewable Energy Share**
- **Extreme Weather Events**
- **Forest Area**

## Tech Stack
- **Framework**: Gradio
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Language Model**: `google/flan-t5-base`
- **Vector DB**: FAISS
- **Processing**: Pure Python (no API dependencies, 100% local/offline-capable)

## Usage
Simply enter a policy-related question (e.g., about mitigation strategies for a specific country) and receive a detailed report including governance, mitigation, and adaptation advice.
