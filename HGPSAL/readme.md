#  HGPSAL Project Setup Guide

This guide provides the necessary steps to set up and run the HGPSAL project. Follow the instructions below to ensure a smooth setup process.

## Prerequisites

- Python installed on your system
- `virtualenv` installed

## Installation Steps

### Step 1: Install `virtualenv`

To create a virtual environment for the project, you need to install `virtualenv`:

`pip install virtualenv`

### Step 2: Create a Virtual Environment

Create a virtual environment for your project. The instructions differ slightly based on your operating system.

Windows:

`virtualenv env_hgpsal`

Linux:

`python -m venv env_hgpsal`

### Step 3: Activate the Virtual Environment
Activate the virtual environment to isolate your project dependencies from your global Python environment.

Windows:

`env_hgpsal\Scripts\activate`

Linux:

`source env_hgpsal/bin/activate`

### Step 4: Install Required Libraries

With the virtual environment activated, install all necessary libraries specified in the requirements.txt file:

`pip install -r requirements.txt`

### Step 5: Run the Project
Finally, run the project using the following command:

`python run_hgpsal.py`





