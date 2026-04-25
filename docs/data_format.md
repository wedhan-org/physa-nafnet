# Data Format Guidelines

## NPY structure

Each `.npy` file should contain a serialized Python dictionary:

```python
{
    "input": np.ndarray,   # shape: [2, H, W], channels: T1, T2
    "label": np.ndarray,   # shape: [1, H, W], acquired STIR
    "case_id": "Case_001",
    "slice_id": 12
}

Split CSV

The split CSV must contain:

filename,case_id,split
Case_001_slice_001.npy,Case_001,train
Case_001_slice_002.npy,Case_001,train
Case_002_slice_001.npy,Case_002,val
Case_003_slice_001.npy,Case_003,test
