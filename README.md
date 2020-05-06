# 480 Project: Implementation and analysis of learning-based frequency estimation algorithms

1. Prepare the datasets.
```bash
python3 Dataset.py
```
2. Train the model.
```bash
python3 Model.py
```
3. Run hashing-based alrgoithm experiments (Count Sketch, Count-Min, Single hash function). Change the "SingleHash" at line 71, 72 to "CountSketch" or "CountMinSketch".

```bash
python3 HashExperiments.py
```
4. Run non-hashing algorithm experiment (Space-Saving).
```bash
python3 NonHashExperiment.py
```
