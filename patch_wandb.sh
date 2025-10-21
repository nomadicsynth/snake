#!/bin/bash
set -e
patch '.venv/lib/python3.11/site-packages/wandb/_pydantic/field_types.py' 'wandb_pydantic.patch'
