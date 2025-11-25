"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=120, gamma=10):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        
        # Enhanced multi-factor strategy with momentum and mean-variance optimization
        for i in range(self.lookback + 1, len(self.price)):
            # Initialize all weights to 0 for this time period
            self.portfolio_weights.loc[self.price.index[i], :] = 0
            
            # Get historical returns window
            returns_window = self.returns[assets].iloc[i - self.lookback : i]
            
            # Calculate multiple momentum timeframes for robustness
            momentum_short = self.price[assets].iloc[i] / self.price[assets].iloc[max(0, i - 20)] - 1  # 1 month
            momentum_long = self.price[assets].iloc[i] / self.price[assets].iloc[max(0, i - 120)] - 1  # 6 months
            
            # Calculate volatilities  
            volatilities = returns_window.std()
            volatilities_short = self.returns[assets].iloc[max(0, i - 20) : i].std()
            
            # Calculate Sharpe-like ratio for each asset (annualized)
            mean_returns = returns_window.mean() * 252
            vol_annual = volatilities * np.sqrt(252)
            sharpe_scores = mean_returns / vol_annual
            
            # Calculate risk-adjusted momentum
            risk_adj_momentum = momentum_long / (volatilities * np.sqrt(252))
            
            # Multi-factor score emphasizing Sharpe and stable momentum
            factor_score = 0.5 * sharpe_scores + 0.3 * risk_adj_momentum + 0.2 * (momentum_short / volatilities_short)
            
            # Select top performing assets - be more selective (top 40%)
            n_select = max(3, int(len(assets) * 0.4))
            top_assets = factor_score.nlargest(n_select).index.tolist()
            
            # Additional filter: only keep assets with positive long-term momentum
            selected_assets = [a for a in top_assets if momentum_long[a] > 0]
            
            if len(selected_assets) >= 2:
                # Use mean-variance optimization on selected assets
                selected_returns = returns_window[selected_assets]
                weights = self.mv_opt(selected_returns, self.gamma)
                
                # Assign weights
                for j, asset in enumerate(selected_assets):
                    self.portfolio_weights.loc[self.price.index[i], asset] = weights[j]
            else:
                # Fall back to defensive portfolio with lowest volatility assets
                n_defensive = min(4, len(assets))
                defensive_assets = volatilities.nsmallest(n_defensive).index
                
                # Use inverse volatility weighting for defensive portfolio
                inv_vol = 1 / volatilities[defensive_assets]
                weights = inv_vol / inv_vol.sum()
                
                for j, asset in enumerate(defensive_assets):
                    self.portfolio_weights.loc[self.price.index[i], asset] = weights.iloc[j]
        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def mv_opt(self, R_n, gamma):
        """Mean-variance optimization using Gurobi"""
        Sigma = R_n.cov().values
        mu = R_n.mean().values
        n = len(R_n.columns)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()
            with gp.Model(env=env, name="portfolio") as model:
                # Decision variable: portfolio weights
                w = model.addMVar(n, name="w", lb=0, ub=1)
                
                # Portfolio expected return
                portfolio_return = mu @ w
                
                # Portfolio variance
                portfolio_variance = w.T @ Sigma @ w
                
                # Objective: maximize return - (gamma/2) * variance
                objective = portfolio_return - (gamma / 2) * portfolio_variance
                model.setObjective(objective, gp.GRB.MAXIMIZE)
                
                # Constraint: weights sum to 1
                model.addConstr(w.sum() == 1, name="budget")
                
                model.optimize()

                # Check optimization status
                if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL:
                    # Extract the solution
                    solution = []
                    for i in range(n):
                        var = model.getVarByName(f"w[{i}]")
                        solution.append(var.X)
                    return solution
                else:
                    # If optimization fails, return equal weights
                    return [1/n] * n

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
