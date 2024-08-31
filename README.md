# THETA

 THETA : A Tau Heterogeneity Evaluation in Alzheimer's Disease

## Citation

When using code to generate THETA please include a reference to the paper:

*https://jnm.snmjournals.org/content/early/2024/07/25/jnumed.123.267273 *

## Usage

#### Prerequisites

- 18F-Flourtaucipir is the tracer that was used to generate the ground truth based on the FDA's approval of its applicability for late stage Alzhimer's disease diagnosis [1].
- The MCALT-ADIR122 [2] atlas was used to generate the regional SUVR values and the right and left sides were averaged for use in the modeling.
- The SUVR values were generated using the cerebellar crus median uptake as a reference.

##### Model training

- Check the *requirements.txt* file for all the packages and their version needed to run the models.
- All models are for a binary classification problem for tau positivity status prediction.

  - A trained model is available for direct use. Check the *Pretrained models/on Mayo Data.* More models will be made available in the future in this directory.
  - Note: The pretrained models will be made available upon accpetance.
- There are two python scripts containing everything one would need to run models as well as generate THETA scores. The description of each function is clearly layed out and should be easy to follow.

  - *utility.py* contains all the helper functions needed to run SHAP [3], calculate regional THETA indices and THETA scores.
  - *main.py* is where you import data and run the models.
  - ```
    #to run simply type this in your terminal or inside a Jupyter notebook
    python3 main.py
    ```

* The *main.py* script provides two training techniques to choose from based on performance and preference. The two options are:

  - *Autogluon* which is a auto-ML technique. We recommend using this package as it offers an extensive set of machine learning models with hyperparameter search options, preprocessing, and more [4].

    - You can load the pretrained model and use it directly:
      ```
      predictor = TabularPredictor.load(os.path.join(os.getcwd(), 'trained models/on Mayo data'), require_version_match=False)
      ```
    - or you can also train a predictor from sctrach:
      ```
      predictor = TabularPredictor(
      label="True_Label", path=os.path.join(os.getcwd(),'Pretrained models/from scratch'), eval_metric="accuracy", problem_type="binary"
      ).fit(
      train_data.reset_index(drop=True),
      # auto_stack=True,
      # holdout_frac=0.20,
      # hyperparameters="default",
      # use_bag_holdout=True,
      # num_bag_sets=4,
      # num_bag_folds=10,
      # num_stack_levels=2,
      # verbosity=2,
      # presets='best_quality'
      )
      ```
  - *Voting classifier* where individual models are trained and their predictions are voted.

  ```
    estimators = [
    ("rf", RandomForestClassifier()),
    ("xgb", xgb.XGBClassifier()),
    ("lgb", lgb.LGBMClassifier()),
    ]

    voting_clf = VotingClassifier(estimators=estimators, voting="soft")

    params = {
    "rf__n_estimators": [500,800,1200],
    "rf__criterion": ['gini', 'entropy', 'log_loss'],

    "lgb__extra_trees": [True],
    "lgb__objective": ["binary"],

    "xgb__n_estimators": [100,500,800,1200],
    }

    voting_grid = GridSearchCV(
    estimator=voting_clf,
    param_grid = params,
    cv=5,
    scoring="accuracy",
    verbose=1,
    n_jobs=-1,
    )

    voting_grid.fit(X_train, y_train.values.ravel())

    y_pred = voting_grid.predict(X_test)
    performance_stats_voting = classification_report(y_test.values.ravel(), y_pred, output_dict=True)
    print('voting',performance_stats_voting)
  ```

- ***Reminder***: Before you begin training please format your data as described in the above prerequisites as data is not provided with this repository.

### Output

- A single pandas DataFrame of all the THETA scores is generated at the end of the training cycle.

### References

[1] [https://www.accessdata.fda.gov/drugsatfda_docs/label/2020/212123s000lbl.pdf](https://www.accessdata.fda.gov/drugsatfda_docs/label/2020/212123s000lbl.pdf)

[2] Schwarz CG, et al. *A large-scale comparison of cortical thickness and volume methods for measuring Alzheimer's disease severity.* NeuroImage: Clinical. 2016

[3] [https://shap.readthedocs.io/en/latest/](https://shap.readthedocs.io/en/latest/) **Paper**: Lundberg, Scott M and Lee, Su-In. *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems 30: 2017

[4] [https://auto.gluon.ai/stable/index.html](https://auto.gluon.ai/stable/index.html) **Paper**: Erickson et al. 2020. *AutoGluon-Tabular: Robust and Accurate AutoML for Structured Data https://doi.org/10.48550/arXiv.2003.06505*
