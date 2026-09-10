"""Dump SERF MLflow traces to JSONL for offline mistake reconstruction."""

import json
import pathlib
import sys

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5001")
client = MlflowClient()

out_path = pathlib.Path(sys.argv[1])
page = None
count = 0
with out_path.open("w", encoding="utf-8") as handle:
    while True:
        res = client.search_traces(locations=["1"], max_results=200, page_token=page)
        for trace in res:
            spans = trace.data.spans
            root = None
            for span in spans:
                if span.parent_id is None:
                    root = span
                    break
            if root is None and spans:
                root = spans[0]
            record = {
                "trace_id": trace.info.trace_id,
                "request_time": trace.info.request_time,
                "state": str(trace.info.state),
                "name": trace.info.tags.get("mlflow.traceName"),
                "token_usage": trace.info.trace_metadata.get("mlflow.trace.tokenUsage"),
                "span_names": [s.name for s in spans],
                "inputs": root.inputs if root is not None else None,
                "outputs": root.outputs if root is not None else None,
            }
            handle.write(json.dumps(record, default=str) + "\n")
            count += 1
        page = res.token
        if not page:
            break
print("dumped", count, "traces to", out_path)
