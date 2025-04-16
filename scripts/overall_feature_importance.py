# get overall feature importance from the individual feature importance for each model

log_reg_importance = {
    "V4": 2.376649,
    "V14": 2.365543,
    "V12": 1.744938,
    "V8": 1.447278,
    "V10": 1.047009,
    "V11": 0.923021,
    "V16": 0.621731,
    "V7": 0.579844,
    "V27": 0.558874,
    "V3": 0.509725
}
decision_tree_importance = {
    "V17": 0.670272,
    "V14": 0.103365,
    "V10": 0.062442,
    "V26": 0.015868,
    "V1": 0.015091,
    "V24": 0.012102,
    "V4": 0.011924,
    "V7": 0.010044,
    "V16": 0.009585,
    "V21": 0.008911
}

random_forest_importance = {
    "V17": 0.205837,
    "V14": 0.161764,
    "V10": 0.130345,
    "V12": 0.095081,
    "V11": 0.077235,
    "V16": 0.050647,
    "V4": 0.039059,
    "V9": 0.031908,
    "V3": 0.029760,
    "V18": 0.028890
}

features = list(set(log_reg_importance.keys()).union(set(decision_tree_importance.keys())).union(set(random_forest_importance.keys())))

overall_importance = {}

for feature in features:
    importance_sum = 0
    if feature in log_reg_importance:
        importance_sum += log_reg_importance[feature]
    if feature in decision_tree_importance:
        importance_sum += decision_tree_importance[feature]
    if feature in random_forest_importance:
        importance_sum += random_forest_importance[feature]
    overall_importance[feature] = importance_sum
overall_importance = sorted(overall_importance.items(), key=lambda x: x[1], reverse=True) # sort
overall_importance = dict(overall_importance)

for feature, importance in overall_importance.items():
    print(feature, importance)