"""Exercise 2: batch prediction. Inside create_app() in src/tariff_service/app.py, replace the
body of predict_batch with:

    @app.post("/predict/batch", response_model=BatchOut)
    def predict_batch(batch: BatchIn, request: Request) -> BatchOut:
        return BatchOut(predictions=[predict_one(request, d) for d in batch.decisions])

A faster version calls top_k once for all texts instead of once per decision.
"""
