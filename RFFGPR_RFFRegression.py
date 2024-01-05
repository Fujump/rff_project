import os
import sys

import docopt
import matplotlib.pyplot as mpl
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import os


def generate_california_housing_dataset(args):
    """
    Create California housing dataset instance.
    """
    # Load California Housing data from sklearn.

    data = sklearn.datasets.fetch_california_housing()

    # Split data to train and test.
    Xs_train, Xs_test, ys_train, ys_test \
        = sklearn.model_selection.train_test_split(data["data"], data["target"], test_size=0.2, random_state=int(args["--seed"]))

    # Data standardization.
    scaler   = sklearn.preprocessing.StandardScaler().fit(Xs_train)
    Xs_train = scaler.transform(Xs_train)
    Xs_test  = scaler.transform(Xs_test)

    return (Xs_train, Xs_test, ys_train, ys_test, data.feature_names)


def main(args):
    """
    Main procedure.
    """

    dim_kernels = [8,16,32,64,128,256]
#     dim_kernels = [80, 160, 320, 640, 1024, 4096]
    R2 = []
    
    for dim_kernel in dim_kernels:
     # Fix seed for random fourier feature calclation
        rfflearn.seed(int(args["--seed"]))

    # Prepare training data
        Xs_train, Xs_test, ys_train, ys_test, feature_names = generate_california_housing_dataset(args)
    #  dim_kernel = 128
    # Create model instance.
        if args["--model_type"] == "rffgpr": 
            model = rfflearn.RFFGPR(dim_kernel=dim_kernel, std_kernel=0.05, std_error=0.1)
            method = "RFFGPR"
        else                               : 
            model = rfflearn.RFFRegression(dim_kernel=dim_kernel, std_kernel=0.05)
            method = "RFFRegression"
        print(method)
        # Train the model.
        model.fit(Xs_train, ys_train)

        # Conduct prediction for the test data
        pred_test = model.predict(Xs_test)
        score_r2  = sklearn.metrics.r2_score(ys_test, pred_test)
        print("- R2 score of the model: ", score_r2)

        # Calculate feature importance (SHAP and permutation importance).
        shap_values = rfflearn.shap_feature_importance(model, Xs_test)
        print("shap_values: ",shap_values)
        print("shap_size: ", shap_values.shape)
        perm_values = rfflearn.permutation_feature_importance(model, Xs_test, ys_test)
        print("perm_values",perm_values)
        print("perm_size: ", perm_values.shape)

        # Draw regression result.
        mpl.figure(figsize=(6, 3.5))
        mpl.scatter(pred_test, ys_test, alpha=0.2)
        mpl.plot([0, 5], [0, 5], "--", color="#666666")
        mpl.title(f"{method}(dim = {dim_kernel}, R2 = %.4f)" % score_r2)
        R2.append(round(score_r2, 4))
        mpl.xlabel("Predicted MedianHouseVal")
        mpl.ylabel("True MedianHouseVal")
        mpl.grid()
        mpl.tight_layout()
        mpl.savefig(f'{method}_{dim_kernel}_figure_california_housing_regression.svg')

        # Visualize SHAP importance.
        mpl.figure(figsize=(6, 3.5))
        rfflearn.shap_plot(shap_values, Xs_test, feature_names, show=False)
        mpl.xlim(-4, 4)
        mpl.tight_layout()
        #mpl.savefig("figure_california_housing_shap_importance.svg")
        mpl.savefig(f'{method}_{dim_kernel}_figure_california_housing_shap_importance.svg')


        # Visualize permurtation importance.
        mpl.figure(figsize=(6, 3.5))
        rfflearn.permutation_plot(perm_values, feature_names, show=False)
        mpl.tight_layout()
        #mpl.savefig("figure_california_housing_permutation_importance.svg")
        mpl.savefig(f'{method}_{dim_kernel}_figure_california_housing_permutation_importance.svg')

        # Show all figures.
        mpl.show()
    print("R2 = ", R2)


if __name__ == "__main__":

    # Parse input arguments.
    args = docopt.docopt(__doc__)

    # Add path to 'rfflearn/' directory.
    # The followings are not necessary if you copied 'rfflearn/' to the current
    # directory or other directory which is included in the Python path.
    current_dir = os.path.dirname(__file__)
    module_path = os.path.join(current_dir, "../../")
    sys.path.append(module_path)

    import rfflearn.cpu as rfflearn

    # Run main procedure.
    main(args)


# Author: Tetsuya Ishikawa <tiskw111@gmail.com>
# vim: expandtab tabstop=4 shiftwidth=4 fdm=marker
