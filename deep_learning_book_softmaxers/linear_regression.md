---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
--- 

# Linear Regression with scikit-learn

```{admonition} Objectives
- Have fun with penguins!
```

This chapter teaches linear regression using `scikit-learn` with a dataset from 
[Deep Learning from HEP](https://hsf-training.github.io/deep-learning-intro-for-hep/).

```{warning}
Penguins can be very cute.
```

Download the dataset from [here](https://github.com/hsf-training/deep-learning-intro-for-hep/blob/main/deep-learning-intro-for-hep/data/penguins.csv) 
and place it in a `data/` folder.

The dataset contains the basic measurements on 3 species of penguins!   

```{code-cell}
print(2 + 2)
```

![A penguin](https://hsf-training.github.io/deep-learning-intro-for-hep/_images/culmen_depth.png)

For our regression problem, let's ask, "Given a flipper length (mm), what is the penguin's most likely body mass (g)?"

```{code-cell} ipython3
import pandas as pd
penguins_df = pd.read_csv("data/penguins.csv")
penguins_df
```

```{code-cell} ipython3
regression_features, regression_targets = penguins_df.dropna()[["flipper_length_mm", "body_mass_g"]].values.T
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

def plot_regression_problem(ax, xlow=170, xhigh=235, ylow=2400, yhigh=6500):
    ax.scatter(regression_features, regression_targets, marker=".")
    ax.set_xlim(xlow, xhigh)
    ax.set_ylim(ylow, yhigh)
    ax.set_xlabel("flipper length (mm)")
    ax.set_ylabel("body mass (g)")

plot_regression_problem(ax)

plt.show()
```


=====================

Let's use Scikit-Learn's `LinearRegression`

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression
import numpy as np
```

```{code-cell} ipython3
best_fit = LinearRegression().fit(regression_features[:, np.newaxis], regression_targets)
```

```{code-cell} ipython3
fig, ax = plt.subplots()

def plot_regression_solution(ax, model, xlow=170, xhigh=235):
    model_x = np.linspace(xlow, xhigh, 1000)
    model_y = model(model_x)
    ax.plot(model_x, model_y, color="tab:orange")

plot_regression_solution(ax, lambda x: best_fit.predict(x[:, np.newaxis]))
plot_regression_problem(ax)

plt.show()
```

```{code-cell} ipython3
print("slope:", best_fit.coef_[0])
print("intercept:", best_fit.intercept_)
```