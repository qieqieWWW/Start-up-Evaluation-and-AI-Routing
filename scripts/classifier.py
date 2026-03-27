try:
	# When project root is on sys.path
	from scripts.mas_blackboard.classifier import ComplexityClassifier
except ModuleNotFoundError:
	# When launched from inside scripts/ (e.g., some Streamlit entry modes)
	from mas_blackboard.classifier import ComplexityClassifier

__all__ = ["ComplexityClassifier"]
