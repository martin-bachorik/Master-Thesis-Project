from SourceCode.identification.deterministic.strejc import Strejc
import matplotlib.pyplot as plt

from SourceCode import templates
templates.Template()

if __name__ == "__main__":
    data = Strejc.strejc(path='./SourceCode/Data/OneStep/Sample01s/all2/pch ({}).csv')
    Strejc.drawFigures(data, save='yes', test_path='./SourceCode/data/MultiStep/T10s/Step10s_10steps_1_7.csv')

    plt.show()
