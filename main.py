# -*- coding: utf-8 -*-

import time, datetime, os
#os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

import sys
#import design
#from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QDialog, QApplication, QPushButton, QVBoxLayout, QHBoxLayout, QSlider, QLineEdit, QLabel, QTableWidgetItem, QSizePolicy
from PyQt5.QtGui import QIcon, QPixmap

#from matplotlib.widgets import Slider, Button, RadioButtons
# Libraries for drawing figures
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.ticker import FuncFormatter, MaxNLocator
# Ensure using PyQt5 backend
import matplotlib as mpl
mpl.use('QT5Agg')
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
#plt.style.use('seaborn')
#plt.tight_layout(pad = 1.25)
from matplotlib.ticker import FuncFormatter

# Libraries for stochastic simulation
import numpy as np
from numpy import log, exp, sqrt
from scipy.stats import skew, kurtosis
from math import ceil, floor
from scipy.special import binom
from scipy.stats import norm
#import bs_call_base as bsm

from Simulation import merton_jump_diffusion_simulate as mjd, heston_stochastic_volatility_simulate as hsv

class HelpPage1(QDialog):
    def __init__(self):
        super(HelpPage1, self).__init__()
        uic.loadUi('help_dialog1.ui', self)

class HelpPage2(QDialog):
    def __init__(self):
        super(HelpPage2, self).__init__()
        uic.loadUi('help_dialog2.ui', self)

class Window(QtWidgets.QDialog):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        # load the .ui file created in Qt Creator directly
        print(f"screen width = {screen_width}")
        if screen_width >= 1920:
            Interface = uic.loadUi('mod4.ui', self)
        else:
            Interface = uic.loadUi('mod4.ui', self)

        # other useful info
        # default values for all parameters
        ## Common parameters
        self.N = int(self.comboBox_SimSize_Merton.currentText()) # simulation size
        
        self.Δt = 0.002
        ## Merton
        self.S0 = 100
        self.K = 100
        self.r = 0.02
        self.q = 0.01
        self.σ = 0.20
        self.t = 100
        self.λ = 3
        self.γ = 0.15
        self.δ = 0.10
        self.S = []
        self.SN = []
        self.path_Merton = 0  # this is the path being selected currently
        self.poisson_jumps = []
        self.returns = []
        ## Heston
        self.S0_Heston = 100
        self.K_Heston  = 100
        self.r_Heston  = 0.02
        self.q_Heston  = 0.01
        self.σ_Heston  = 0.20
        self.t_Heston  = 100
        self.κ_Heston  = 0.20
        self.θ_Heston  = 0.20
        self.ρ_Heston  = 0.25
        self.v0_Heston = 0.35
        self.S_Heston = []
        self.SN_Heston = []
        self.path_Heston = 0  # this is the path being selected currently
        self.returns_Heston = []
        self.volatility_Heston = []
        #self.seed = 12345
        self.seed = np.random.randint(0,65535)
        params = {'S0': log(self.S0), 'K': log(self.K), 'r': self.r, 'q': self.q, 'σ': self.σ, 't': self.t, 'λ': self.λ, 'γ': self.γ, 'δ': self.δ,  'Δt': self.Δt, 'seed': self.seed}
        self.params = params

        # Merton model
        self.slider_K.valueChanged.connect(self.on_change_K)
        self.slider_r.valueChanged.connect(self.on_change_r)
        self.slider_q.valueChanged.connect(self.on_change_q)
        self.slider_sigma.valueChanged.connect(self.on_change_sigma)
        self.slider_t.valueChanged.connect(self.on_change_t)
        self.slider_lambda.valueChanged.connect(self.on_change_lambda)
        self.slider_gamma.valueChanged.connect(self.on_change_gamma)
        self.slider_delta.valueChanged.connect(self.on_change_delta)

        self.button_Simulate.clicked.connect(self.simulate_Merton)

        self.slider_t1.valueChanged.connect(self.on_change_t)

        # Heston model
        self.slider_K_Heston.valueChanged.connect(self.on_change_K_Heston)
        self.slider_r_Heston.valueChanged.connect(self.on_change_r_Heston)
        self.slider_q_Heston.valueChanged.connect(self.on_change_q_Heston)
        self.slider_sigma_Heston.valueChanged.connect(self.on_change_sigma_Heston)
        self.slider_kappa_Heston.valueChanged.connect(self.on_change_kappa_Heston)
        self.slider_theta_Heston.valueChanged.connect(self.on_change_theta_Heston)
        self.slider_v0_Heston.valueChanged.connect(self.on_change_v0_Heston)
        self.slider_rho_Heston.valueChanged.connect(self.on_change_rho_Heston)

        self.button_Simulate_Heston.clicked.connect(self.simulate_Heston)
        self.slider_t1_Heston.valueChanged.connect(self.on_change_t1_Heston)

        self.button_Help_Merton.clicked.connect(self.executeHelpPage1)
        self.button_Help_Heston.clicked.connect(self.executeHelpPage2)

        # pick a simulated path
        self.button_PickSim_Heston.clicked.connect(self.PickSim_Heston)
        self.button_PickSim_Merton.clicked.connect(self.PickSim_Merton)


        # initialize parameter values - Merton
        self.label_S0.setText(f"S0 = {self.S0}")
        self.label_K.setText(f"K = {self.K}")
        self.label_r.setText(f"r = {self.r}")
        self.label_q.setText(f"q = {self.q}")
        self.label_sigma.setText(f"σ = {self.σ}")
        self.label_lambda.setText(f"λ = {self.λ}")
        self.label_gamma.setText(f"γ = {self.γ}")
        self.label_delta.setText(f"δ = {self.δ}")

        # initialize parameter values - Heston
        self.label_S0_Heston.setText(f"S0 = {self.S0_Heston}")
        self.label_K_Heston.setText(f"K = {self.K_Heston}")
        self.label_r_Heston.setText(f"r = {self.r_Heston}")
        self.label_q_Heston.setText(f"q = {self.q_Heston}")
        self.label_kappa_Heston.setText(f"κ = {self.κ_Heston}")
        self.label_theta_Heston.setText(f"θ = {self.θ_Heston}")
        self.label_sigma_Heston.setText(f"σ = {self.σ_Heston}")
        self.label_v0_Heston.setText(f"v0 = {self.v0_Heston}")
        self.label_rho_Heston.setText(f"ρ = {self.ρ_Heston}")

        # set tooltips
        #self.slider_delta.setToolTip('This is a tooltip message.')  
        # set help icon
        icon  = QIcon('information.png')
        self.button_Help_Merton.setIcon(icon)
        self.button_Help_Heston.setIcon(icon)

        # initialize self.comboBox_SimSize is changed
        self.comboBox_SimSize_Heston.setCurrentText('100')
        self.comboBox_SimSize_Merton.setCurrentText('100')
        # not working

        self.tabWidget.tabBarClicked.connect(self.on_click_tabWidget)

        # Make sure window size is set
        self.setGeometry(0, 0, 1366, 768)

    def on_click_tabWidget(self):
        #self.tabWidget.repaint()
        self.tabWidget.update()
        self.oMplCanvas.update()
        self.oMplCanvas_Heston.update()
        #QApplication.repaint()

    def executeHelpPage1(self):
        help_page = HelpPage1()
        help_page.exec_() # modal
        #help_page.run() # non-modal: not working

    def executeHelpPage2(self):
        help_page = HelpPage2()
        help_page.exec_() # modal
        #help_page.show() # non-modal: not working

    # Heston model
    def on_change_t1_Heston(self, value):
        self.t_Heston = value
        self.label_t1_Heston.setText(f"t = {str(self.t_Heston)}")
        self.showGraph_Heston()
    def on_change_K_Heston(self, value):
        self.K_Heston = value
        self.label_K_Heston.setText(f"K = {str(self.K_Heston)}")
    def on_change_sigma_Heston(self, value):
        self.σ_Heston = value/100
        self.label_sigma_Heston.setText(f"σ = {str(self.σ_Heston)}")
    def on_change_r_Heston(self, value):
        self.r_Heston = value/100
        self.label_r_Heston.setText(f"r = {str(self.r_Heston)}")
    def on_change_q_Heston(self, value):
        self.q_Heston = value/100
        self.label_q_Heston.setText(f"q = {str(self.q)}")
    def on_change_kappa_Heston(self, value):
        self.κ_Heston = value/100
        self.label_kappa_Heston.setText(f"κ = {str(self.κ_Heston)}")
    def on_change_theta_Heston(self, value):
        self.θ_Heston = value/100
        self.label_theta_Heston.setText(f"θ = {str(self.θ_Heston)}")
    def on_change_v0_Heston(self, value):
        self.v0_Heston = value/100
        self.label_v0_Heston.setText(f"v0 = {str(self.v0_Heston)}")
    def on_change_rho_Heston(self, value):
        self.ρ_Heston = value/100
        self.label_rho_Heston.setText(f"ρ = {str(self.ρ_Heston)}")



    # Merton model
    def on_change_t(self, value):
        self.t = value
        self.label_t1.setText(f"t = {str(self.t)}")
        self.showGraph()
    def on_change_K(self, value):
        self.K = value
        self.label_K.setText(f"K = {str(self.K)}")
        self.showGraph()
    def on_change_sigma(self, value):
        self.σ = value/100
        self.label_sigma.setText(f"σ = {str(self.σ)}")
    def on_change_r(self, value):
        self.r = value/100
        self.label_r.setText(f"r = {str(self.r)}")
    def on_change_q(self, value):
        self.q = value/100
        self.label_q.setText(f"q = {str(self.q)}")
    # λγδ
    def on_change_lambda(self, value):
        self.λ = value
        self.label_lambda.setText(f"λ = {str(self.λ)}")
    def on_change_gamma(self, value):
        self.γ = value/100 - 0.20
        self.label_gamma.setText(f"γ = {str(self.γ)}")
    def on_change_delta(self, value):
        self.δ = value/100
        self.label_delta.setText(f"λ = {str(self.δ)}")

    def on_click_ShowSimulation(self):
        print("Simulation button clicked")
        self.showGraph()

    def simulate_Merton(self):
        self.N = int(self.comboBox_SimSize_Merton.currentText())
        self.label_Message_Merton.setStyleSheet("color: red")
        self.label_Message_Merton.setText('Processing...')
        self.label_Message_Merton.repaint()
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        #S0 = 100
        #K = 100; r = 0.025; q = 0.005; σ = 0.2; T = 1.0
        S0 = self.S0
        K  = self.K
        r  = self.r
        q  = self.q
        σ  = self.σ
        t  = self.t
        λ  = self.λ
        γ  = self.γ
        δ  = self.δ
        Δt = self.Δt
        params = {'S0': log(S0), 'K': log(K), 'r': r, 'q': q, 'σ': σ, 't': t, 'λ': λ, 'γ': γ, 'δ': δ,  'Δt': Δt, 'seed': self.seed}
        #print(f"params = {params}")
        self.params = params
        self.SN = []
        self.poisson_jumps = []
        for i in range(self.N):
            log_stock_prices, poisson_jump = mjd(self.params)
            self.SN.append(log_stock_prices)
            self.poisson_jumps.append(poisson_jump)
        self.SN = np.array(self.SN)
        self.poisson_jumps = np.array(self.poisson_jumps)
        self.label_Message_Merton.setText('')
        self.showGraph()
        QApplication.restoreOverrideCursor()

    def simulate_Heston(self):
        self.N = int(self.comboBox_SimSize_Heston.currentText())
        self.label_Message_Heston.setStyleSheet("color: red")
        self.label_Message_Heston.setText('Processing...')
        self.label_Message_Heston.repaint()
        QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        #S0 = 100
        #K = 100; r = 0.025; q = 0.005; σ = 0.2; T = 1.0
        S0 = self.S0_Heston
        K  = self.K_Heston
        r  = self.r_Heston
        q  = self.q_Heston
        σ  = self.σ_Heston
        t  = self.t_Heston
        κ  = self.κ_Heston
        θ  = self.θ_Heston
        ρ  = self.ρ_Heston
        v0 = self.v0_Heston
        Δt = self.Δt
        params = {'S0': log(S0), 'K': log(K), 'r': r, 'q': q, 'σ': σ, 'κ': κ, 'θ': θ, 'ρ': ρ, 'v0': v0, 'Δt': Δt, 'seed': self.seed}
        #print(f"params = {params}")
        self.params_Heston = params
        self.SN_Heston = []
        self.volatility_Heston = []
        for i in range(self.N):
            log_stock_prices, volatility = hsv(self.params_Heston)
            self.SN_Heston.append(log_stock_prices)
            self.volatility_Heston.append(volatility)
        self.SN_Heston = np.array(self.SN_Heston)
        self.volatility_Heston = np.array(self.volatility_Heston)
        self.label_Message_Heston.setText('')
        self.showGraph_Heston()
        QApplication.restoreOverrideCursor()


    def showGraph(self):

        mpl = self.oMplCanvas.canvas
        [s1, s2, s3, s4] = mpl.axes

        S0 = self.S0
        K  = self.K
        r  = self.r
        q  = self.q
        σ  = self.σ
        t  = self.t
        λ  = self.λ
        γ  = self.γ
        δ  = self.δ
        Δt = self.Δt
        self.S = self.SN[self.path_Merton]


        s1.clear()
        s1.grid(True)
        s1.set_title('Metron Jump Diffusion $S_t$', fontsize=14, color='brown')
        s1.yaxis.set_tick_params(labelright=False, labelleft=True)
        s1.axvline(x=self.t, color="blue", alpha=0.5, linewidth=2)
        s1.plot(self.S, 'b', lw=0.5, label='present value')

        s2.clear()
        s2.set_title(r'$ln{\frac{S_{t+Δt}}{S_t}}$', fontsize=14, color='brown')
        s2.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
        s2.axvline(x=self.t, color="blue", alpha=0.5, linewidth=2)
        s2.grid(True)
        M = self.S[t]
        returns = [ log(self.S[t+1]) - log(self.S[t]) for t in range(0, self.S.size-1)]
        s2.plot(returns, 'r', lw=0.5, label='return')
        #s2.axvline(x=slope, color="g", alpha=0.75, linewidth=2)
        #s2.plot(x, z_put, 'b-.', lw=1.5, label='payoff')

        ###########################################
        # Plot s3: return distribution histogram 
        ###########################################
        returns = []
        
        for i in range(self.N):
            ret = [ log(self.SN[i,t+1]) - log(self.SN[i,t]) for t in range(0, self.S.size-1)]
            returns.append(ret)
        self.returns = returns
        #returns = [ log(self.S[t+1]) - log(self.S[t]) for t in range(0, self.S.size-1)]
        #returns_dist_at_t = [ log(self.SN[:, t+1]) - log(self.SN[:, t]) for t in range(0, self.SN.size-1)]
        returns = [ log(self.SN[:,self.t]) - log(self.SN[:,self.t-1])] 
        s3.clear()
        s3.hist(returns, density=True, bins=20, orientation='vertical', alpha=0.40, color='green')
        s3.set_title(r"Return Distribution", fontsize=11, color='brown')
        s3.yaxis.set_ticks_position("right")
        s3.xaxis.set_major_formatter(FuncFormatter('{0:.1%}'.format))

        s3.grid(True)
        s3.tick_params(axis = 'both', which = 'major', labelsize = 6)
        s3.set_xlabel('Return', fontsize=8, color='brown')

        ###############################################################
        # Plot s4: No. of jumps distribution histogram 
        ###############################################################
        s4.set_visible(True)
        s4.clear()
        data_simulated_poisson = [ self.poisson_jumps[n, :self.t+1].sum() for n in range(0,self.N) ]
        s4.hist( data_simulated_poisson, bins=range(0, max  (6,max(data_simulated_poisson)+1)), \
            density=True, rwidth=0.5, align='left', facecolor='orange', alpha=0.75)
        s4.set_title(r"Poisson Jump Distribution", fontsize=11, color='brown')
        #s4.xaxis.set_major_formatter(FuncFormatter('{0}'.format))
        s4.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
        s4.xaxis.set_major_locator(MaxNLocator(integer=True))
        s4.grid(True)
        s4.tick_params(axis = 'both', which = 'major', labelsize = 6)
        s4.set_xlabel('No. of jumps', fontsize=8, color='brown')

        mpl.fig.set_visible(True)
        mpl.draw()


    def PickSim_Merton(self):
        self.path_Merton = np.random.randint(0, self.N)
        self.showGraph()        
        return None

    def PickSim_Heston(self):
        '''
        Out of the many simulated paths (self.N). choose one and display it with showGraph_Heston()
        numpy.random.randint(low, high=None, size=None, dtype='l')
            Return random integers from low (inclusive) to high (exclusive).
        '''
        self.path_Heston = np.random.randint(0, self.N)
        self.showGraph_Heston()        
        return None

    def showGraph_Heston(self):

        mpl = self.oMplCanvas_Heston.canvas
        [s1, s2, s3, s4] = mpl.axes

        S0 = self.S0_Heston
        K  = self.K_Heston
        r  = self.r_Heston
        q  = self.q_Heston
        σ  = self.σ_Heston
        t  = self.t_Heston
        κ  = self.κ_Heston
        θ  = self.θ_Heston
        ρ  = self.ρ_Heston
        v0 = self.v0_Heston
        Δt = self.Δt
        self.S_Heston = self.SN_Heston[self.path_Heston]

        s1.clear()
        s1.set_title('Heston Stochastic Volatility Model $S_t$', fontsize=11, color='brown')
        s1.yaxis.set_tick_params(labelright=False, labelleft=True)
        s1.axvline(x=self.t_Heston, color="blue", alpha=0.5, linewidth=2)
        s1.grid(True)
        s1.plot(self.S_Heston, 'b', lw=0.5, label='present value')

        s2.clear()
        s2.set_title(r'$ln{\frac{S_{t+Δt}}{S_t}}$', fontsize=14, color='brown')
        s2.axvline(x=self.t_Heston, color="blue", alpha=0.5, linewidth=2)
        s2.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
        #s2.axvline(x=self.t, color="blue", alpha=0.5, linewidth=1)
        s2.grid(True)
        M = self.S_Heston[t]
        # these are the returns for the given path
        returns = [ log(self.S_Heston[t+1]) - log(self.S_Heston[t]) for t in range(0, self.S_Heston.size-1)]
        s2.plot(returns, 'r', lw=0.5, label='return')
        #s2.axvline(x=slope, color="g", alpha=0.75, linewidth=2)
        #s2.plot(x, z_put, 'b-.', lw=1.5, label='payoff')

        ###########################################
        # Plot s3: return distribution histogram 
        ###########################################
        returns = []
        
        for i in range(self.N):
            ret = [ log(self.SN_Heston[i,t+1]) - log(self.SN_Heston[i,t]) for t in range(0, self.S_Heston.size-1)]
            returns.append(ret)
        self.returns_Heston = returns
        #returns = [ log(self.S_Heston[t+1]) - log(self.S_Heston[t]) for t in range(0, self.S_Heston.size-1)]
        #returns_dist_at_t = [ log(self.SN[:, t+1]) - log(self.SN[:, t]) for t in range(0, self.SN.size-1)]
        # we want the simulated return distribution at time a particular time t in range(0, self.S_Heston.size-1)
        returns = [ log(self.SN_Heston[:,self.t_Heston]) - log(self.SN_Heston[:,self.t_Heston-1])] 

        s3.clear()
        s3.hist(returns, density=False, bins=20, orientation='vertical', alpha=0.40, color='green')
        s3.set_title(r"Return Distribution", fontsize=11, color='brown')
        s3.yaxis.set_ticks_position("right")
        s3.xaxis.set_major_formatter(FuncFormatter('{0:.1%}'.format))

        s3.grid(True)
        s3.tick_params(axis = 'both', which = 'major', labelsize = 6)
        s3.set_xlabel('Return', fontsize=8, color='brown')

        ###############################################################
        # Plot s4: stochastic volatility distribution histogram 
        ###############################################################
        volatility_at_t = self.volatility_Heston[:,self.t_Heston]
        s4.clear()
        s4.hist(volatility_at_t, bins=25, density=True, orientation='vertical', alpha=0.70, color='orange')
        s4.set_title(r"Stochastic Volatility Distribution", fontsize=11, color='brown')
        s4.xaxis.set_major_formatter(FuncFormatter('{0:.1}'.format))
        s4.grid(True)
        s4.tick_params(axis = 'both', which = 'major', labelsize = 6)
        s4.set_xlabel('volatility', fontsize=8, color='brown')

        ### draw all graphs
        mpl.fig.set_visible(True)
        mpl.draw()



# This creates the GUI window:
if __name__ == '__main__':

    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet("QToolTip { color: black; background-color: #FFFFE0; border: 1px solid white; }")
    ### load appropriately sized UI based on screen resolution detected
    screen = app.primaryScreen()
    screen_width=screen.size().width()
    window = Window()
    w = window
    window.show()
    sys.exit(app.exec_())

