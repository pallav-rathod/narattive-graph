install:
	pip install -r requirements.txt && pip install -e .
demo:
	python scripts/run_pipeline.py --source synthetic && streamlit run app/dashboard.py
test:
	pytest -q
