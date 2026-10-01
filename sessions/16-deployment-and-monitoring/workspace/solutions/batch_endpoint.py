"""Exercise 2: batch prediction. Inside create_app() in src/sentiment_service/app.py, replace the
body of predict_batch with:

    @app.post("/predict/batch", response_model=BatchOut)
    def predict_batch(batch: BatchIn, request: Request) -> BatchOut:
        return BatchOut(predictions=[predict_one(request, review) for review in batch.reviews])

A faster version calls model.predict_proba once for all texts instead of once per review.
"""
