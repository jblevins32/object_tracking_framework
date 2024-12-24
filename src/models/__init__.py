import os
import importlib
import sys

# Dynamically import all .py files in the models directory
current_dir = os.path.dirname(__file__)
for filename in os.listdir(current_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = filename[:-3]
        module = importlib.import_module(f".{module_name}", package="models")
        globals().update({name: cls for name, cls in module.__dict__.items() if isinstance(cls, type)})

# Optionally, implement the factory
def get_model(model_name):
    if model_name in globals():
        model = globals()[model_name]
        return model()
    raise ValueError(f"Unknown model name: {model_name}")