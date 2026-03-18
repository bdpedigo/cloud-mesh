```mermaid
graph LR
    mesh["Mesh Input (raw surface mesh)"]
    hks_pipeline["HKS Processing Pipeline"]
    hks_features["HKS Feature Outputs"]
    bucket["Object Storage Bucket"]

    mesh --> hks_pipeline
    hks_pipeline --> hks_features
    hks_features --> bucket
```