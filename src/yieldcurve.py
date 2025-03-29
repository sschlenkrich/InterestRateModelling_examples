
import QuantLib as ql

import matplotlib.pyplot as plt
import numpy as np
import pandas

class YieldCurve:

    # Python constructor
    def __init__(self, terms, rates):
        today = ql.Settings.instance().evaluationDate
        self.terms = [ '0d' ] + terms
        self.dates = [ ql.WeekendsOnly().advance(today,ql.Period(term),ql.ModifiedFollowing) for term in self.terms ]
        self.rates = [rates[0]] + rates
        # use rates as backward flat interpolated continuous compounded forward rates
        self.yts = ql.ForwardCurve(self.dates,self.rates,ql.Actual365Fixed(),ql.NullCalendar())

    # zero coupon bond
    def discount(self,dateOrTime):
        return self.yts.discount(dateOrTime,True)

    def forwardRate(self,time):
        return self.yts.forwardRate(time,time,ql.Continuous,ql.Annual,True).rate()
  
    # plot zero rates and forward rate
    def plot(self,stepsize=0.1):
        times = [ k*stepsize for k in range(int(round(30.0/stepsize,0))+1) ]
        continuousForwd = [ self.yts.forwardRate(time,time,ql.Continuous,ql.Annual,True).rate() for time in times ]
        continuousZeros = [ self.yts.zeroRate(time,ql.Continuous,ql.Annual,True).rate() for time in times ]
        annualZeros     = [ self.yts.zeroRate(time,ql.Compounded,ql.Annual,True).rate() for time in times ]
        # print(times, continuousForwd, continuousZeros, annualZeros)
        plt.plot(times,continuousForwd, label='Cont. forward rate')
        plt.plot(times,continuousZeros, label='Cont. zero rate')
        plt.plot(times,annualZeros,     label='Annually comp. zero rate')
        plt.legend()
        plt.xlabel('Maturity')
        plt.ylabel('Interest rate')
        plt.show()

    # return a table with curve data
    def table(self):
        table = pandas.DataFrame( [ self.terms, self.dates, self.rates ] ).T
        table.columns = [ 'Terms', 'Dates', 'Rates' ]
        return table

    def referenceDate(self):
        return self.referenceDate()


class FlatForwardCurve:
    """
    A simple flat yield curve mainly used for testing.
    """

    # Python constructor
    def __init__(self, rate):
        self.rate = rate

    def discount(self, T):
        return np.exp(-self.rate * T)

    def forwardRate(self,time):
        return self.rate
