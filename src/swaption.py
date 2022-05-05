
import numpy as np
import QuantLib as ql

from src.helpers import bachelier
from src.helpers import bachelier_vega
from src.swap import Swap

class Swaption:

    # Python constructor
    def __init__(self, underlyingSwap, expiryDate, normalVolatility):
        self.underlyingSwap = underlyingSwap
        self.exercise = ql.EuropeanExercise(expiryDate)
        self.swaption = ql.Swaption(self.underlyingSwap.swap,self.exercise,ql.Settlement.Physical)
        self.normalVolatility = normalVolatility
        volHandle = ql.QuoteHandle(ql.SimpleQuote(normalVolatility))
        engine = ql.BachelierSwaptionEngine(self.underlyingSwap.discHandle,volHandle,ql.Actual365Fixed())
        self.swaption.setPricingEngine(engine)

    def npv(self):
        return self.swaption.NPV()

    def fairRate(self):
        return self.underlyingSwap.fairRate()

    def annuity(self):
        return self.underlyingSwap.annuity()
    
    def npv_via_bachelier(self):
        """
        Calculate NPV manually using Bachelier formula.
        This method is intended to validate QuantLib's engine.
        """
        refDate  = self.underlyingSwap.discHandle.referenceDate()
        T = ql.Actual365Fixed().yearFraction(refDate,self.exercise.dates()[0])
        CallOrPutOnS = 1.0 if self.underlyingSwap.payerOrReceiver==ql.VanillaSwap.Payer else -1.0
        return self.annuity() * bachelier(self.underlyingSwap.fixedRate,self.fairRate(),self.normalVolatility,T,CallOrPutOnS)

    def vega(self):
        refDate  = self.underlyingSwap.discHandle.referenceDate()
        T = ql.Actual365Fixed().yearFraction(refDate,self.exercise.dates()[0])
        return self.annuity() * bachelier_vega(self.underlyingSwap.fixedRate,self.fairRate(),self.normalVolatility,T) * 1.0e-4  # 1bp scaling

    def bondOptionDetails(self):
        """
        Calculate expiryTime, (coupon) startTims, payTimes, cashFlows, strike and
        c/p flag as inputs to Hull White analytic formula.
        """
        details = {}
        details['callOrPut'] = 1.0 if self.underlyingSwap.payerOrReceiver==ql.VanillaSwap.Receiver else -1.0
        details['strike']    = 0.0
        refDate  = self.underlyingSwap.discHandle.referenceDate()
        details['expiryTime'] = ql.Actual365Fixed().yearFraction(refDate,self.exercise.dates()[0])
        fixedLeg = [ [ ql.Actual365Fixed().yearFraction(refDate,cf.date()), cf.amount() ]
                     for cf in self.underlyingSwap.swap.fixedLeg() ]
        details['fixedLeg'] = np.array(fixedLeg)
        floatLeg = [ [ ql.Actual365Fixed().yearFraction(refDate,ql.as_coupon(cf).accrualStartDate()),
                       ((1 + ql.as_coupon(cf).accrualPeriod()*ql.as_coupon(cf).rate()) *
                        self.underlyingSwap.discHandle.discount(ql.as_coupon(cf).accrualEndDate()) /
                        self.underlyingSwap.discHandle.discount(ql.as_coupon(cf).accrualStartDate()) - 1.0) *
                       ql.as_coupon(cf).nominal() 
                       ] 
                     for cf in self.underlyingSwap.swap.floatingLeg() ]
        details['floatLeg'] = np.array(floatLeg)    
        payTimes = [ floatLeg[0][0]  ]          +       \
                   [ cf[0] for cf in floatLeg ] +       \
                   [ cf[0] for cf in fixedLeg ] +       \
                   [ ql.Actual365Fixed().yearFraction(refDate,ql.as_coupon(
                     self.underlyingSwap.swap.floatingLeg()[-1]).accrualEndDate()) ]
        caschflows = [ -ql.as_coupon(self.underlyingSwap.swap.floatingLeg()[0]).nominal() ] +  \
                     [ -cf[1] for cf in floatLeg ] +    \
                     [  cf[1] for cf in fixedLeg ] +    \
                     [ ql.as_coupon(self.underlyingSwap.swap.floatingLeg()[0]).nominal() ]
        details['payTimes'  ] = np.array(payTimes)
        details['cashFlows'] = np.array(caschflows)
        return details
