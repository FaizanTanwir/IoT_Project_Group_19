try:
    from time import ticks_us, ticks_diff
except ImportError:
    from time import time_ns

    def ticks_us(): return int(time_ns() * 1000)
    def ticks_diff(a, b): return a - b

class RandomForestClassifier:
    """
    # RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_name=RandomForestClassifier, class_weight=None, criterion=gini, estimator=DecisionTreeClassifier(), estimator_params=('criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'random_state', 'ccp_alpha', 'monotonic_cst'), max_depth=5, max_features=sqrt, max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, monotonic_cst=None, n_estimators=10, n_jobs=None, num_outputs=2, oob_score=False, package_name=everywhereml.sklearn.ensemble, random_state=None, template_folder=everywhereml/sklearn/ensemble, verbose=0, warm_start=False)
    """

    def __init__(self):
        """
        Constructor
        """
        self.latency = 0
        self.predicted_value = -1

        self.votes = [0.00000000000, 0.00000000000]

    def predict(self, x):
        """
        Predict output from input vector
        """
        self.predicted_value = -1
        started_at = ticks_us()

        self.votes = [0.00000000000, 0.00000000000]

        idx, score = self.tree0(x)
        self.votes[idx] += score
        
        idx, score = self.tree1(x)
        self.votes[idx] += score
        
        idx, score = self.tree2(x)
        self.votes[idx] += score
        
        idx, score = self.tree3(x)
        self.votes[idx] += score
        
        idx, score = self.tree4(x)
        self.votes[idx] += score
        
        idx, score = self.tree5(x)
        self.votes[idx] += score
        
        idx, score = self.tree6(x)
        self.votes[idx] += score
        
        idx, score = self.tree7(x)
        self.votes[idx] += score
        
        idx, score = self.tree8(x)
        self.votes[idx] += score
        
        idx, score = self.tree9(x)
        self.votes[idx] += score

        # get argmax of votes
        max_vote = max(self.votes)
        self.predicted_value = next(i for i, v in enumerate(self.votes) if v == max_vote)

        self.latency = ticks_diff(ticks_us(), started_at)
        return self.predicted_value

    def latencyInMicros(self):
        """
        Get latency in micros
        """
        return self.latency

    def latencyInMillis(self):
        """
        Get latency in millis
        """
        return self.latency // 1000

    def tree0(self, x):
        """
        Random forest's tree #0
        """
        if x[0] < 25.050000190734863:
            if x[1] < 30.0:
                return 0, 0.37142857142857144
            else:
                if x[0] < 19.949999809265137:
                    return 0, 0.37142857142857144
                else:
                    if x[1] < 60.5:
                        return 1, 0.6285714285714286
                    else:
                        return 0, 0.37142857142857144
        else:
            return 0, 0.37142857142857144

    def tree1(self, x):
        """
        Random forest's tree #1
        """
        if x[0] < 25.050000190734863:
            if x[1] < 30.199999809265137:
                return 0, 0.38857142857142857
            else:
                if x[1] < 65.0:
                    if x[1] < 39.85000038146973:
                        return 1, 0.6114285714285714
                    else:
                        if x[2] < 1014.3999938964844:
                            return 1, 0.6114285714285714
                        else:
                            return 0, 0.38857142857142857
                else:
                    return 0, 0.38857142857142857
        else:
            return 0, 0.38857142857142857

    def tree2(self, x):
        """
        Random forest's tree #2
        """
        if x[2] < 1011.25:
            if x[2] < 1010.75:
                return 0, 0.37142857142857144
            else:
                if x[1] < 61.0:
                    if x[2] < 1011.1499938964844:
                        if x[0] < 26.100000381469727:
                            return 1, 0.6285714285714286
                        else:
                            return 0, 0.37142857142857144
                    else:
                        return 0, 0.37142857142857144
                else:
                    return 0, 0.37142857142857144
        else:
            if x[2] < 1015.9500122070312:
                if x[2] < 1013.2749938964844:
                    if x[1] < 45.10000038146973:
                        return 0, 0.37142857142857144
                    else:
                        if x[2] < 1012.0499877929688:
                            return 1, 0.6285714285714286
                        else:
                            return 1, 0.6285714285714286
                else:
                    if x[1] < 29.5:
                        return 0, 0.37142857142857144
                    else:
                        if x[2] < 1014.9500122070312:
                            return 1, 0.6285714285714286
                        else:
                            return 1, 0.6285714285714286
            else:
                return 0, 0.37142857142857144

    def tree3(self, x):
        """
        Random forest's tree #3
        """
        if x[2] < 1010.75:
            return 0, 0.4228571428571429
        else:
            if x[0] < 19.949999809265137:
                return 0, 0.4228571428571429
            else:
                if x[0] < 25.050000190734863:
                    if x[2] < 1014.6999816894531:
                        if x[2] < 1013.1999816894531:
                            return 1, 0.5771428571428572
                        else:
                            return 1, 0.5771428571428572
                    else:
                        return 1, 0.5771428571428572
                else:
                    return 0, 0.4228571428571429

    def tree4(self, x):
        """
        Random forest's tree #4
        """
        if x[1] < 60.5:
            if x[2] < 1011.0499877929688:
                if x[2] < 1010.6999816894531:
                    return 0, 0.3485714285714286
                else:
                    if x[0] < 26.100000381469727:
                        return 1, 0.6514285714285715
                    else:
                        return 0, 0.3485714285714286
            else:
                if x[1] < 29.699999809265137:
                    return 0, 0.3485714285714286
                else:
                    if x[0] < 20.050000190734863:
                        return 0, 0.3485714285714286
                    else:
                        if x[2] < 1011.3000183105469:
                            return 1, 0.6514285714285715
                        else:
                            return 1, 0.6514285714285715
        else:
            return 0, 0.3485714285714286

    def tree5(self, x):
        """
        Random forest's tree #5
        """
        if x[0] < 25.100000381469727:
            if x[1] < 29.550000190734863:
                return 0, 0.38285714285714284
            else:
                if x[2] < 1010.6499938964844:
                    return 0, 0.38285714285714284
                else:
                    if x[1] < 39.85000038146973:
                        return 1, 0.6171428571428571
                    else:
                        if x[0] < 20.949999809265137:
                            return 0, 0.38285714285714284
                        else:
                            return 1, 0.6171428571428571
        else:
            return 0, 0.38285714285714284

    def tree6(self, x):
        """
        Random forest's tree #6
        """
        if x[2] < 1010.8999938964844:
            return 0, 0.4057142857142857
        else:
            if x[0] < 20.0:
                return 0, 0.4057142857142857
            else:
                if x[1] < 29.300000190734863:
                    return 0, 0.4057142857142857
                else:
                    if x[1] < 57.64999961853027:
                        if x[2] < 1012.0499877929688:
                            return 1, 0.5942857142857143
                        else:
                            return 1, 0.5942857142857143
                    else:
                        if x[1] < 58.04999923706055:
                            return 0, 0.4057142857142857
                        else:
                            return 1, 0.5942857142857143

    def tree7(self, x):
        """
        Random forest's tree #7
        """
        if x[1] < 60.14999961853027:
            if x[2] < 1010.8500061035156:
                return 0, 0.3942857142857143
            else:
                if x[1] < 30.0:
                    return 0, 0.3942857142857143
                else:
                    if x[0] < 19.949999809265137:
                        return 0, 0.3942857142857143
                    else:
                        if x[2] < 1012.0499877929688:
                            return 1, 0.6057142857142858
                        else:
                            return 1, 0.6057142857142858
        else:
            return 0, 0.3942857142857143

    def tree8(self, x):
        """
        Random forest's tree #8
        """
        if x[1] < 60.25:
            if x[0] < 25.25:
                if x[2] < 1012.9500122070312:
                    return 1, 0.6171428571428571
                else:
                    if x[1] < 30.050000190734863:
                        return 0, 0.38285714285714284
                    else:
                        if x[1] < 46.95000076293945:
                            return 1, 0.6171428571428571
                        else:
                            return 0, 0.38285714285714284
            else:
                return 0, 0.38285714285714284
        else:
            return 0, 0.38285714285714284

    def tree9(self, x):
        """
        Random forest's tree #9
        """
        if x[0] < 24.84999942779541:
            if x[2] < 1015.8000183105469:
                if x[1] < 30.050000190734863:
                    return 0, 0.4057142857142857
                else:
                    if x[1] < 60.39999961853027:
                        if x[2] < 1015.1999816894531:
                            return 1, 0.5942857142857143
                        else:
                            return 1, 0.5942857142857143
                    else:
                        return 0, 0.4057142857142857
            else:
                return 0, 0.4057142857142857
        else:
            if x[2] < 1010.75:
                return 0, 0.4057142857142857
            else:
                if x[1] < 57.5:
                    return 0, 0.4057142857142857
                else:
                    if x[1] < 61.0:
                        return 1, 0.5942857142857143
                    else:
                        return 0, 0.4057142857142857