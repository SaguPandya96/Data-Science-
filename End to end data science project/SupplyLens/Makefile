.PHONY: install download-data validate-data train reports test notebook dashboard validate-project

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	python -m pip install -e .

download-data:
	python scripts/download_data.py

validate-data:
	python scripts/validate_data.py

train:
	python scripts/train.py

reports:
	python scripts/build_reports.py
	python scripts/build_readme.py

test:
	python -m pytest

notebook:
	python scripts/build_notebook.py

dashboard:
	python -m streamlit run app/app.py

validate-project:
	python scripts/validate_project.py
