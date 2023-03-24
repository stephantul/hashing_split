# hashingsplit

Stable train test splits using hashing.

Train test splits are normally done randomly, and are made reproducible using a fixed random seed. This approach fails, however if the cardinality of the data changes, i.e., data is added or removed: suddenly, points can be swapped from train to test and vice versa. This is annoying, as previous hyperparameter searches or other choices could be made on what is now test data.

An alternative is to use the hash of the training data as a signal to decide which items belong to train or test. This works as follows: a hashing function assigns a random N-bit integer to an instance. Because hash functions are uniform, any bit can be set with equal probability. Therefore, the N-bit integer modulo a given number X will give rise to a unform distribution over X, i.e., every number from 0 to X will have an equal probability of being selected. 

For example, if we want to assign 1 out of every 10 items to the test set, we use 10 as our modulo number. As our hash function is uniform, approximately 1 out of 10 hashes modulo 10 will have 0 as a result. These items go to test, while the others go to train.

In pseudocode:

```python

for item in X:
    if hash(item) % 10 == 0:
        X_test.append(item)
    else:
        X_train.append(item)
```

Unlike other train test splitting methods, the assignment of items does not change when items are added to, or removed from, the training set.

## Usage

The package exposes a single function: `hash_split`. This is used to split items on the basis of their hashes. Currently, we made it so you have to pass in a list of `y`. If you don't need `y`, please pass in some dummy data. It doesn't affect the output in any way.

```
from hashingsplit import hash_split

# Generate some dummy data:
X = []
y = []
for x in range(1000):
    instance = [{"text": f"document_{x} + {chr(x)}", "attribute": list(range(x)), "set": {3, 2, 1}}, 0, 1.0, 1.8, {3, 2, 1}]
    X.append(instance)
    y.append(int(x < 500))

X_train, X_test, y_train, y_test = hash_split(X, y, test_size=.1)

# Create X2 by adding another 1000 items to X
X2 = [*X]
y2 = [*y]
for x in range(len(X), len(X) + 1000):
    instance = [{"text": f"document_{x} + {chr(x)}", "attribute": list(range(x)), "set": {3, 2, 1}}, 0, 1.0, 1.8, {3, 2, 1}]
    X2.append(instance)
    y2.append(int(x < 500))

X_train2, X_test2, y_train2, y_test2 = hash_split(X2, y2, test_size=.1)

# This split should be stable when items are added: 
assert all(item in X_train2 for item in X_train)
assert all(item in X_test2 for item in X_test)
```

