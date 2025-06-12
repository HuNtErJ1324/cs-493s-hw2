import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Make a table with the following columns:
# - a: Value of a (integers from 1 to p)
# - b: Value of b (integers from 1 to p)
# - c: The resulting value from operation a o c, mod p
# - p: Set value for p (97 or 113)
# - o: Operator (addition (+), subtraction (-), or division (/))

# Step 1: Create numpy arrays, each containing values for a column
p = np.concatenate((np.repeat(97, 98*98*3), np.repeat(113, 114*114*3)))
a = np.concatenate((np.tile(np.repeat(np.arange(98), 98), 3), np.tile(np.repeat(np.arange(114), 114), 3)))
b = np.concatenate((np.tile(np.arange(98), 98*3), np.tile(np.arange(114), 114*3)))
o = np.concatenate((np.repeat("+", 98*98), np.repeat("-", 98*98), np.repeat("/", 98*98), 
                   np.repeat("+", 114*114), np.repeat("-", 114*114), np.repeat("/", 114*114)))

# Step 2: Find c using NumPy operations
c = np.array([])
for p_val in (97, 113):
    # Templates
    a_template = np.repeat(np.arange(p_val+1), p_val+1)
    b_template = np.tile(np.arange(p_val+1), p_val+1)

    # Addition with mod
    c_addition = np.mod(np.add(a_template, b_template), p_val)
 
    # Subtraction with mod
    c_subtraction = np.mod(np.subtract(a_template, b_template), p_val)

    # Division with mod
    c_division = np.mod(np.floor_divide(a_template, b_template), p_val)
    
    # Put resulting values in c
    c = np.concatenate((c, c_addition, c_subtraction, c_division))
    

# Step 4: Put them into one Pandas DataFrame and publish as csv file
# with the columns converted to characters for use with the Transformers model
# & removing rows with division by zero
data = pd.DataFrame({
    "p": p.astype(str),
    "o": o.astype(str),
    "a": a.astype(str),
    "b": b.astype(str),
    "c": c.astype(int).astype(str)
}).query("not (o == '/' and b == '0')")
data.to_csv('data.csv', index=False)

# Step 5: Create train, test, and validation splits
# (50% train, 25% test, 25% validation)
data_train, data_test = train_test_split(data, test_size=0.5)
data_test, data_val = train_test_split(data_test, test_size=0.5)
data_train.to_csv('data_train.csv', index=False)
data_test.to_csv('data_test.csv', index=False)
data_val.to_csv('data_val.csv', index=False)