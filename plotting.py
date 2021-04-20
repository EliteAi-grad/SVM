import numpy as np
# Plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class Plotting(object):
    def set_legend(self):
      legendElements = [
            Line2D([0], [0], linestyle='none',marker='o', color='red',markerfacecolor='red', markersize=9),
            Line2D([0], [0], linestyle='none',marker='o', color='green', markerfacecolor='green', markersize=9),
            Line2D([0], [0], linestyle='-', marker='.', color='black', markerfacecolor='black',markersize=0),
            Line2D([0], [0], linestyle='--', marker='.', color='blue', markerfacecolor='black', markersize=0),
            Line2D([0], [0], linestyle='none', marker='.', color='black', markerfacecolor='black', markersize=9)
      ]
      return legendElements

    def plot_margin(self,X1, X2, objFit):
        fig = plt.figure()  # create a figure object
        ax = fig.add_subplot(1, 1, 1)
        # Format plot area:
        ax = plt.gca()
        ax = plt.axes(facecolor='#FFFD03')  # background color.
        # Axis limits.
        x1_min, x1_max = X1.min(), X1.max()
        x2_min, x2_max = X2.min(), X2.max()
        ax.set(xlim=(x1_min, x1_max), ylim=(x2_min, x2_max))
        # Labels
        plt.xlabel('$x_1$', fontsize=9)
        plt.ylabel('$x_2$', fontsize=9)
        legendElements = self.set_legend()
        
        myLegend = plt.legend(legendElements,   ['Negative', 'Positive','Decision Boundary','Margin','Support Vectors'],fontsize="7",loc='lower center', bbox_to_anchor=(0.7, 0.98))
        # plot points
        plt.plot(X1[:, 0], X1[:, 1], marker='o',markersize=5, color='red',linestyle='none')
        plt.plot(X2[:, 0], X2[:, 1], marker='o',markersize=4, color='green',linestyle='none')
        plt.scatter(objFit.sv[:, 0], objFit.sv[:, 1], s=60, color="blue")   # The points designating the support vectors.
        
        if  objFit.kernel  == 'polynomial' or objFit.kernel  == 'gaussian':
            # Non-linear margin line needs to be generated. Will use a contour plot.
            _X1, _X2 = np.meshgrid(np.linspace(x1_min, x1_max, 50), np.linspace(x1_min, x1_max, 50))
            X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(_X1), np.ravel(_X2))])

            if objFit.kernel == 'polynomial' or objFit.kernel == 'gaussian':
                Z = objFit.helper(X).reshape(_X1.shape)
            else:
                print("unknown fit_type")
                return

            plt.contour(_X1, _X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
            plt.contour(_X1, _X2, Z + 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
            plt.contour(_X1, _X2, Z - 1, [0.0], colors='grey', linestyles='--', linewidths=1, origin='lower')
        else:
            # Linear margin line needs to be generated.
            w = objFit.w
            c = objFit.b
            _y1 = (-w[0] * x1_min - c ) / w[1]
            _y2 = (-w[0] * x1_max - c ) / w[1]
            plt.plot([x1_min, x1_max], [_y1, _y2], "k")

            #upper margin
            _y3 = (-w[0] * x1_min - c + 1) / w[1]
            _y4 = (-w[0] * x1_max - c  + 1) / w[1]
            plt.plot([x1_min, x1_max], [_y3, _y4], "k--")

            #lower_argin
            _y5 = (-w[0] * x1_min - c - 1 ) / w[1]
            _y6 = (-w[0] * x1_max - c - 1 ) / w[1]
            plt.plot([x1_min, x1_max], [_y5, _y6], "k--")

        plt.show(block=False)

