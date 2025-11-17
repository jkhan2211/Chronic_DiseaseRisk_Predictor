# Description 

The src/ directory contains all core application logic for the Disease Feature Classifier project. It is organized to separate experiments, production-ready code, model operations, and backend API components. This structure ensures clean development workflows, modular updates, easier debugging, and a clear path from research → feature engineering → model development → API deployment.


# Folder Structure

```
src/
├── api/
│   ├── main.py
│   ├── inference.py
│   ├── schemas.py
│   └── requirements.txt
│   📌 Backend API — exposes ML models via REST endpoints.
│
├── features/
│   📌 Feature engineering scripts, transformations, and utilities.
│
├── models_operations/
│   📌 Model training, evaluation, saving/loading, pipelines.
│
├── Experiments_JunaidKhan/
│   📌 Junaid’s prototype notebooks, tests, and experimental models.
│
├── Experiments_NO/
│   📌 NO team member’s experiments, drafts, and exploration notebooks.
│
├── R&D/
│   📌 Research and development space for trying new ideas, algorithms, and approaches.
│
└── README.md
    📌 This file — explains the purpose and layout of the `src/` directory.
```