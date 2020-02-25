import os
import importlib
# automatically import any Python files in the tasks/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith(".py") and not file.startswith("_"):
        task_name = file[: file.find(".py")]
        module = importlib.import_module("fennlp.tasks." + task_name)
