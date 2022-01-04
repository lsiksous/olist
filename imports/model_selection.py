import pandas as pd
from patsy import dmatrices

def stratified_split(data: pd.DataFrame,
                     target: str,
                     n_samples: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified data split.
    Splits data into training and testing dataframes whilst stratifying
    to maintain target variable proportions.
    Args:
        data (DataFrame): Full dataframe including predictors as well as
            target variable.
        target (str): The name of the column containing the target
            variable.
        n_samples (int): The number of samples to include in the test
            set from each target label category.
    Returns:
        tuple: The training and testing dataframes.
    """
    n = min(n_samples, data[target].value_counts().min())
    test_df = data.groupby(target).apply(lambda x: x.sample(n))
    test_df.index = test_df.index.droplevel(0)
    train_df = data[~data.index.isin(test_df.index)]
    return train_df, test_df

def bootstrap(df):
    cols = df.columns
    pmax = 1
    while pmax > 0.05:
        formula = 'freq ~ '

        for x in cols:
            formula = formula + f'{x} + '

        formula = formula[:-2]

        y, X = dmatrices(formula, data=df, return_type='dataframe')

        res = sm.OLS(y, X).fit()
        pmax = max(res.pvalues)
        idx = res.pvalues.index.tolist()

        if pmax > 0.05:
            for j in range(0, len(res.pvalues)):
                if (res.pvalues[j].astype(float) == pmax):
                    var = idx[j]
                    for i in cols:
                        if i in var:
                            print(f'{i} has been discarded')
                            cols.remove(i)
                
    print(res.summary())
    return res, cols

