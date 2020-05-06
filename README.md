# 480 Project: Implementation and analysis of learning-based frequency estimation algorithms

1. Prepare the datasets. This will generate "test.txt", "training.txt", and "validation.txt" files.
```bash
python3 Dataset.py
```
2. Train the model. This will generate a "model.h5" file.
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
