# Detect-an-isolated-camera-zoo

## Requirements:
- Tensoflow p27
- Keras
- Numpy
- Pandas

## Installation

**Note**:
- Place the `inference.py` file in same folder as `model_isolation.json` and `model_wieghts_isolation.h5`.
- Test csv file format with **no headers** and make sure labels are **small case**.

    | Id            | Label         |
    | ------------- |:-------------:|
    | data/10030/jpg/320_180/thumb00175913.jpg      | true |
    | data/10030/jpg/320_180/thumb00175443.jpg     | false      |
- output format:
```javascript
{'accuracy':accuracy, 'recall':recall, 'precision':precision}
```

```python
from inference import inference
result = inference(<csv_file>)
```
