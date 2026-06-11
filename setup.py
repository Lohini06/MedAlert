from setuptools import setup, find_packages

setup(
    name="medalert",
    version="1.0.0",
    description="Multimodal Adverse Drug Event Prediction using BioBERT + XGBoost Fusion",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "shap>=0.43.0",
        "gradio>=4.0.0",
        "loguru>=0.7.0",
    ],
)