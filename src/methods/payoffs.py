"""
A set of classes that act as payoffs.

A payoff needs to implement an 'at(x)' method that
takes a model state as input and returns the payoff
for that given state.
"""

class CouponBond:
    # Python constructor
    def __init__(self, model, observation_time, pay_times, cash_flows):
        self.model            = model
        self.observation_time = observation_time
        self.pay_times        = pay_times
        self.cash_flows       = cash_flows

    def at(self, x):
        bond = sum([
            cf * self.model.zero_bond_payoff(x,self.observation_time,T)
            for cf, T in zip(self.cash_flows, self.pay_times)
            ])
        return bond
