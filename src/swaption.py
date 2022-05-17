
import numpy as np
import QuantLib as ql

from src.helpers import bachelier
from src.helpers import bachelier_vega
from src.swap import Swap

class Swaption:

    # Python constructor
    def __init__(self, underlying_swap, expiryDate, normalVolatility):
        self.underlying_swap = underlying_swap
        self.exercise = ql.EuropeanExercise(expiryDate)
        self.swaption = ql.Swaption(self.underlying_swap.swap,self.exercise,ql.Settlement.Physical)
        self.normalVolatility = normalVolatility
        volHandle = ql.QuoteHandle(ql.SimpleQuote(normalVolatility))
        engine = ql.BachelierSwaptionEngine(self.underlying_swap.discHandle,volHandle,ql.Actual365Fixed())
        self.swaption.setPricingEngine(engine)

    def fixed_rate(self):
        return self.underlying_swap.fixedRate

    def call_or_put(self):
        return +1.0 if self.underlying_swap.payerOrReceiver==ql.VanillaSwap.Payer else -1.0

    def npv(self):
        return self.swaption.NPV()

    def fairRate(self):
        return self.underlying_swap.fairRate()

    def annuity(self):
        return self.underlying_swap.annuity()
    
    def npv_via_bachelier(self):
        """
        Calculate NPV manually using Bachelier formula.
        This method is intended to validate QuantLib's engine.
        """
        refDate  = self.underlying_swap.discHandle.referenceDate()
        T = ql.Actual365Fixed().yearFraction(refDate,self.exercise.dates()[0])
        CallOrPutOnS = 1.0 if self.underlying_swap.payerOrReceiver==ql.VanillaSwap.Payer else -1.0
        return self.annuity() * bachelier(self.underlying_swap.fixedRate,self.fairRate(),self.normalVolatility,T,CallOrPutOnS)

    def vega(self):
        refDate  = self.underlying_swap.discHandle.referenceDate()
        T = ql.Actual365Fixed().yearFraction(refDate,self.exercise.dates()[0])
        return self.annuity() * bachelier_vega(self.underlying_swap.fixedRate,self.fairRate(),self.normalVolatility,T) * 1.0e-4  # 1bp scaling

    def bond_option_details(self):
        """
        Calculate expiryTime, (coupon) startTims, payTimes, cashFlows, strike and
        c/p flag as inputs to Hull White analytic formula.
        """
        details = {}
        details['call_or_put']  = 1.0 if self.underlying_swap.payerOrReceiver==ql.VanillaSwap.Receiver else -1.0
        details['strike_price'] = 0.0
        refDate  = self.underlying_swap.discHandle.referenceDate()
        details['expiry_time'] = ql.Actual365Fixed().yearFraction(refDate,self.exercise.dates()[0])
        fixedLeg = [ [ ql.Actual365Fixed().yearFraction(refDate,cf.date()), cf.amount() ]
                     for cf in self.underlying_swap.swap.fixedLeg() ]
        details['fixed_leg'] = np.array(fixedLeg)
        floatLeg = [ [ ql.Actual365Fixed().yearFraction(refDate,ql.as_coupon(cf).accrualStartDate()),
                       ((1 + ql.as_coupon(cf).accrualPeriod()*ql.as_coupon(cf).rate()) *
                        self.underlying_swap.discHandle.discount(ql.as_coupon(cf).accrualEndDate()) /
                        self.underlying_swap.discHandle.discount(ql.as_coupon(cf).accrualStartDate()) - 1.0) *
                       ql.as_coupon(cf).nominal() 
                       ] 
                     for cf in self.underlying_swap.swap.floatingLeg() ]
        details['float_leg'] = np.array(floatLeg)
        payTimes = [ floatLeg[0][0]  ]          +       \
                   [ cf[0] for cf in floatLeg ] +       \
                   [ cf[0] for cf in fixedLeg ] +       \
                   [ ql.Actual365Fixed().yearFraction(refDate,ql.as_coupon(
                     self.underlying_swap.swap.floatingLeg()[-1]).accrualEndDate()) ]
        caschflows = [ -ql.as_coupon(self.underlying_swap.swap.floatingLeg()[0]).nominal() ] +  \
                     [ -cf[1] for cf in floatLeg ] +    \
                     [  cf[1] for cf in fixedLeg ] +    \
                     [ ql.as_coupon(self.underlying_swap.swap.floatingLeg()[0]).nominal() ]
        details['pay_times'  ] = np.array(payTimes)
        details['cash_flows'] = np.array(caschflows)
        return details


def create_swaption(expiryTerm, swapTerm, discCurve, projCurve, strike='ATM', payerOrReceiver=ql.VanillaSwap.Payer, normalVolatility=0.01):
    """
    An easy to use constructor function for convenience.
    """
    today      = discCurve.yts.referenceDate()
    expiryDate = ql.TARGET().advance(today,ql.Period(expiryTerm),ql.ModifiedFollowing)
    startDate  = ql.TARGET().advance(expiryDate,ql.Period('2d'),ql.Following)
    endDate    = ql.TARGET().advance(startDate,ql.Period(swapTerm),ql.Unadjusted)
    if str(strike).upper()=='ATM':
        swap = Swap(startDate,endDate,0.0,discCurve,projCurve)
        strike = swap.fairRate()
    swap = Swap(startDate,endDate,strike,discCurve,projCurve,payerOrReceiver)
    swaption = Swaption(swap,expiryDate,normalVolatility)
    return swaption
