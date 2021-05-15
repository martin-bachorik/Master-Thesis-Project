import pandas as pd
import matplotlib.pyplot as plt
from SourceCode.identification.stochastic.regressors import ARX

from SourceCode import templates
templates.Template()

if __name__ == "__main__":
    # "ARX METHOD"
    p_lags = 20  # No. regressors for both na: y(t-n) and nb: u(t-n)
    model = ARX(p_lags, n_diff=None, scale_factor=False)
    data = pd.read_csv('./SourceCode/data/MultiStep/T10s/random_open_data.csv')
    model.train(data)
    arx_param = model.parameters

    test_data = pd.read_csv('./SourceCode/data/MultiStep/T10s/Step10s_10steps_1_7.csv')
    model.test_online(test_data, save=False, save_csv=None)
    # model.test(test_data, save=False, save_csv=None)

    print("weight parameters of b: {}\n"
          "weight parameters of a: {}".format(arx_param['weights'][:p_lags], arx_param['weights'][p_lags:]))
    print("bias parameters: {}".format(arx_param['bias']))
    print(model.test_rmse)

    plt.show()
