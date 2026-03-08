# DSPy Programming Guide

This guide provides an overview of how to use the DSPy framework for building and optimizing LLM pipelines.

## Getting Started

1. **Installation**: Install DSPy via pip:

   ```
   pip install dspy
   ```

2. **Basic Usage**: Import DSPy and create a simple pipeline:

   ```python
   import dspy

   pipeline = dspy.Pipeline()
   ```

## Key Concepts

- **Pipelines**: A sequence of data processing steps.
- **Adapters**: Components that connect DSPy with external systems (e.g., databases, APIs).
- **Models**: Pre-trained LLMs that can be fine-tuned for specific tasks.

## Example

Here's a simple example of a DSPy pipeline that uses a pre-trained model. Always use `dspy.adapters.baml_adapter.BAMLAdapter` for the adapter if we are using Pydantic classes as complex signatures.

```python
import dspy
from dspy.adapters.baml_adapter import BAMLAdapter


# Get Gemini API key from environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is

# Setup the LLM with the BAMLAdapter
lm = dspy.LM("gemini/gemini-2.0-flash", api_key=GEMINI_API_KEY)
dspy.configure(lm=lm, adapter=BAMLAdapter())


# Define the format of the input and output data via a DSPy signature
class ExtractInfo(dspy.Signature):
    """Extract structured information from text."""

    text: str = dspy.InputField()
    title: str = dspy.OutputField()
    headings: list[str] = dspy.OutputField()
    entities: list[dict[str, str]] = dspy.OutputField(desc="a list of entities and their metadata")

# Create a simple module using `dspy.Predict` - there are many types like `dspy.ChainOfThought`.
module = dspy.Predict(ExtractInfo)

# Run the module with its input parameters
text = "Apple Inc. announced its latest iPhone 14 today." \
    "The CEO, Tim Cook, highlighted its new features in a press release."
response = module(text=text)

# See the output fields...
print(response.title)
print(response.headings)
print(response.entities)
```

## Advanced Features

- **Custom Adapters**: Create your own adapters to connect to different data sources. We use [BAMLAdapter](https://github.com/prrao87/dspy/blob/main/dspy/adapters/baml_adapter.py) for all DSPy modules.
- **Model Fine-tuning**: Fine-tune pre-trained models on your own data for better performance.

## Conclusion

DSPy is a powerful framework for building LLM pipelines. With its flexible architecture and easy-to-use API, you can quickly create and deploy your own NLP applications.
