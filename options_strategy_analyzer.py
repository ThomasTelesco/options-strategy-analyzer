# Import math functions like log, sqrt, and exp
import math
import matplotlib.pyplot as plt

# Import the normal distribution from scipy
# This is needed for Black-Scholes probability calculations
from scipy.stats import norm

# Function to calculate Black-Scholes option price
def black_scholes_price(S, K, T, r, sigma, option_type):
    """
    PARAMETERS
    S = current stock price
    K = strike price
    T = time to expiration in years
    r = risk-free rate
    sigma = volatility
    option_type = "call" or "put"
    """

    # -----------------------------
    # BLACK-SCHOLES CALCULATIONS
    # -----------------------------

    # d1 is one of the key Black-Scholes variables
    # It measures how far the stock price is from the strike
    # while adjusting for time, volatility, and interest rates
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    # d2 is closely related to d1
    # It adjusts d1 downward based on volatility over time
    d2 = d1 - sigma * math.sqrt(T)

    # -----------------------------
    # CALL OPTION PRICING
    # -----------------------------
    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

     # -----------------------------
     # PUT OPTION PRICING
     # -----------------------------
    elif option_type == "put":
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # -----------------------------
    # INVALID INPUT HANDLING
    # -----------------------------
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Return the calculated option price
    return price

def calculate_greeks(S, K, T, r, sigma, option_type):
    """
        PARAMETERS

        S = Current stock price
        K = Strike price
        T = Time to expiration (in years)
        r = Risk-free interest rate (decimal form)
        sigma = Annualized volatility (decimal form)
        option_type = "call" or "put"

        RETURNS

        Delta
        Gamma
        Theta
        Vega
        Rho
        """

    # ---------------------------------------------------
    # CALCULATE d1
    # ---------------------------------------------------
    #
    # d1 is the most important variable in Black-Scholes.
    #
    # It measures how far the stock price is above or below
    # the strike price while accounting for:
    #
    # - volatility
    # - time remaining
    # - interest rates
    #
    # Many Greeks are derived directly from d1.
    #

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    # ---------------------------------------------------
    # CALCULATE d2
    # ---------------------------------------------------
    #
    # d2 adjusts d1 downward by one standard deviation
    # of future stock movement.
    #
    # It is used heavily in pricing and rho calculations.
    #

    d2 = d1 - sigma * math.sqrt(T)

    # ---------------------------------------------------
    # DELTA
    # ---------------------------------------------------
    #
    # Delta measures:
    #
    # "How much does the option price change if the stock
    # price moves by $1?"
    #
    # Example:
    #
    # Delta = 0.60
    #
    # If stock rises $1
    # option rises approximately $0.60
    #

    if option_type == "call":
        delta = norm.cdf(d1)

        # ------------------------------------------------
        # THETA (CALL)
        # ------------------------------------------------
        #
        # Theta measures time decay.
        #
        # It answers:
        #
        # "How much value does the option lose each day
        # if everything else stays constant?"
        #
        # Theta is usually negative because options lose
        # value as expiration approaches.
        #
        theta = (
            -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * norm.cdf(d2)
        ) / 365

        # ------------------------------------------------
        # RHO (CALL)
        # ------------------------------------------------
        #
        # Rho measures sensitivity to interest rates.
        #
        # Example:
        #
        # Rho = 0.05
        #
        # If interest rates rise by 1%
        # option value increases about $0.05
        #
        rho = (K * T * math.exp(-r * T) * norm.cdf(d2)) / 100

    elif option_type == "put":
        # Put delta ranges from -1 to 0
        delta = norm.cdf(d1) - 1
        # Theta for put options
        theta = (
            -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * norm.cdf(-d2)
        ) / 365
        # Rho for put options
        rho = (-K * T * math.exp(-r * T) * norm.cdf(-d2)) / 100

    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # ---------------------------------------------------
    # GAMMA
    # ---------------------------------------------------
    #
    # Gamma measures:
    #
    # "How quickly does Delta change?"
    #
    # Example:
    #
    # Delta today = 0.50
    #
    # Gamma = 0.04
    #
    # If stock rises $1:
    #
    # New Delta ≈ 0.54
    #
    # Gamma is highest for options near:
    #
    # - the strike price
    # - expiration
    #

    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))

    # ---------------------------------------------------
    # VEGA
    # ---------------------------------------------------
    #
    # Vega measures sensitivity to volatility.
    #
    # Example:
    #
    # Vega = 0.12
    #
    # If implied volatility rises by 1%
    #
    # option value rises approximately $0.12
    #
    # Higher volatility generally increases
    # option prices.
    #
    vega = (S * norm.pdf(d1) * math.sqrt(T)) / 100

    return delta, gamma, theta, vega, rho

def implied_volatility(market_price, S, K, T, r, option_type):
    """
    Estimate implied volatility using trial-and-error search.

    market_price = actual option price in the market
    S = current stock price
    K = strike price
    T = time to expiration in years
    r = risk-free rate
    option_type = "call" or "put"
    """

    # Start with a low possible volatility
    low_vol = 0.01

    # Start with a high possible volatility
    high_vol = 3.00

    # Repeat enough times to get a good estimate
    for i in range(100):

        # Try the midpoint volatility
        mid_vol = (low_vol + high_vol) / 2

        # Price the option using the midpoint volatility
        estimated_price = black_scholes_price(S, K, T, r, mid_vol, option_type)

        # If our estimated price is too high,
        # then volatility is too high
        if estimated_price > market_price:
            high_vol = mid_vol

        # If our estimated price is too low,
        # then volatility is too low
        else:
            low_vol = mid_vol

    # Return the final midpoint volatility estimate
    return (low_vol + high_vol) / 2

def plot_long_option_payoff(K, premium, option_type):
    """
        PARAMETERS

        K = Strike Price

        premium = Price paid for the option

        option_type = "call" or "put"

        PURPOSE

        Creates a payoff diagram showing
        profit/loss at expiration for a
        long option position.
        """

    # ----------------------------------------
    # GENERATE POSSIBLE STOCK PRICES
    # ----------------------------------------
    #
    # We want to see what happens if
    # the stock finishes at many prices.
    #
    # Example:
    #
    # Strike = 100
    #
    # We generate:
    #
    # 50, 51, 52, ... , 150
    #

    stock_prices = range(int(K * 0.5), int(K * 1.5) + 1)
    # Store profit/loss values
    profits = []

    # ----------------------------------------
    # CALCULATE PROFIT FOR EACH STOCK PRICE
    # ----------------------------------------
    #
    # We loop through every possible
    # expiration stock price.
    #

    for stock_price in stock_prices:
        # ------------------------------------
        # LONG CALL
        # ------------------------------------
        #
        # Profit formula:
        #
        # max(stock - strike, 0) - premium
        #
        if option_type == "call":
            profit = max(stock_price - K, 0) - premium
            # Break-even occurs when
            # profit equals zero
            #
            # Call break-even:
            #
            # Strike + Premium
            #
            break_even = K + premium
            # Maximum loss for a call
            # is the premium paid
            #
            max_loss = premium

        elif option_type == "put":
            profit = max(K - stock_price, 0) - premium
            # Put break-even:
            #
            # Strike - Premium
            #
            break_even = K - premium
            max_loss = premium

        else:
            raise ValueError("option_type must be 'call' or 'put'")

        profits.append(profit)

    # ----------------------------------------
    # CREATE GRAPH WINDOW
    # ----------------------------------------
    #
    plt.figure()
    # Plot:
    #
    # x-axis = stock prices
    # y-axis = profits
    #
    plt.plot(stock_prices, profits)
    # ----------------------------------------
    # PROFIT = 0 LINE
    # ----------------------------------------
    #
    # This is the break-even reference line.
    #

    # Horizontal line where profit = 0
    plt.axhline(0)
    # ----------------------------------------
    # STRIKE PRICE LINE
    # ----------------------------------------
    #
    # Vertical line showing strike.
    #

    plt.axvline(K)

    # ----------------------------------------
    # BREAK-EVEN LINE
    # ----------------------------------------
    #
    # Vertical line showing where
    # profit becomes positive.
    #
    plt.axvline(break_even)

    # ----------------------------------------
    # LABEL THE STRIKE
    # ----------------------------------------
    #
    plt.text(K, min(profits), f"Strike = ${K:.2f}", rotation=90)
    # ----------------------------------------
    # LABEL THE BREAK-EVEN POINT
    # ----------------------------------------
    #
    plt.text(break_even, 0, f"Break-even = ${break_even:.2f}", rotation=90)

    # ----------------------------------------
    # CREATE SUMMARY BOX
    # ----------------------------------------
    #
    summary_text = (
        f"Premium Paid: ${premium:.2f}\n"
        f"Break-even: ${break_even:.2f}\n"
        f"Max Loss: ${max_loss:.2f}"
    )
    # ----------------------------------------
    # DRAW SUMMARY BOX
    # ----------------------------------------
    #
    plt.text(
        min(stock_prices),
        max(profits) * 0.75,
        summary_text,
        bbox=dict(facecolor="white", edgecolor="black")
    )
    # ----------------------------------------
    # GRAPH LABELS
    # ----------------------------------------
    #
    plt.title(f"Long {option_type.capitalize()} Payoff Diagram")
    plt.xlabel("Stock Price at Expiration")
    plt.ylabel("Profit / Loss")
    # Display graph
    plt.show()

def plot_bull_call_spread_payoff(lower_K, higher_K, lower_premium, higher_premium):
        """
        PARAMETERS

        lower_K = strike price of call we BUY

        higher_K = strike price of call we SELL

        lower_premium = premium paid for purchased call

        higher_premium = premium received from sold call

        PURPOSE

        Create a payoff diagram for a Bull Call Spread.
        """

        # --------------------------------------------------
        # GENERATE POSSIBLE STOCK PRICES AT EXPIRATION
        # --------------------------------------------------
        #
        # Example:
        #
        # Buy 100 Call
        # Sell 110 Call
        #
        # Generate stock prices from roughly:
        #
        # 50 to 165
        #
        stock_prices = range(
            int(lower_K * 0.5),
            int(higher_K * 1.5) + 1
        )

        # Store profit values
        profits = []

        # --------------------------------------------------
        # NET PREMIUM
        # --------------------------------------------------
        #
        # Example:
        #
        # Buy call for $7
        # Sell call for $3
        #
        # Net premium:
        #
        # 7 - 3 = 4
        #
        net_premium = lower_premium - higher_premium

        # --------------------------------------------------
        # BREAK-EVEN
        # --------------------------------------------------
        #
        # Bull Call Spread:
        #
        # Lower Strike + Net Premium
        #
        break_even = lower_K + net_premium

        # --------------------------------------------------
        # MAX LOSS
        # --------------------------------------------------
        #
        # Worst case:
        #
        # Both options expire worthless
        #
        # Lose net premium paid
        #
        max_loss = net_premium

        # --------------------------------------------------
        # MAX GAIN
        # --------------------------------------------------
        #
        # Spread width:
        #
        # Higher Strike - Lower Strike
        #
        # Example:
        #
        # 110 - 100 = 10
        #
        # Minus premium paid:
        #
        # 10 - 4 = 6
        #
        max_gain = (higher_K - lower_K ) - net_premium

        # --------------------------------------------------
        # CALCULATE PROFIT AT EACH STOCK PRICE
        # --------------------------------------------------
        #
        for stock_price in stock_prices:
            # ----------------------------------------------
            # LONG CALL PROFIT
            # ----------------------------------------------
            #
            # Option we purchased
            #
            # Example:
            #
            # Stock = 120
            # Strike = 100
            #
            # Value:
            #
            # 120 - 100 = 20
            #
            # Profit:
            #
            # 20 - premium paid
            #
            long_call_profit = (
                    max(stock_price - lower_K, 0)
                    - lower_premium
            )

            # ----------------------------------------------
            # SHORT CALL PROFIT
            # ----------------------------------------------
            #
            # Option we sold
            #
            # We keep premium received
            #
            # But lose money if stock rises
            # above the short strike
            #
            short_call_profit = (
                    higher_premium
                    - max(stock_price - higher_K, 0)
            )

            # ----------------------------------------------
            # TOTAL STRATEGY PROFIT
            # ----------------------------------------------
            #
            total_profit = (
                    long_call_profit
                    + short_call_profit
            )

            profits.append(total_profit)

        # --------------------------------------------------
        # CREATE GRAPH
        # --------------------------------------------------
        #
        plt.figure()

        # Plot profit curve
        plt.plot(stock_prices, profits)

        # Profit = 0 reference line
        plt.axhline(0)

        # Lower strike line
        plt.axvline(lower_K)

        # Higher strike line
        plt.axvline(higher_K)

        # Break-even line
        plt.axvline(break_even)

        # --------------------------------------------------
        # CREATE SUMMARY BOX
        # --------------------------------------------------
        #
        summary_text = (
            f"Buy Call Strike: ${lower_K:.2f}\n"
            f"Sell Call Strike: ${higher_K:.2f}\n"
            f"Net Premium: ${net_premium:.2f}\n"
            f"Break-even: ${break_even:.2f}\n"
            f"Max Loss: ${max_loss:.2f}\n"
            f"Max Gain: ${max_gain:.2f}"
        )

        # --------------------------------------------------
        # DISPLAY SUMMARY BOX
        # --------------------------------------------------
        #
        plt.text(
            min(stock_prices),
            max(profits) * 0.75,
            summary_text,

            bbox=dict(
                facecolor="white",
                edgecolor="black"
            )
        )

        # --------------------------------------------------
        # GRAPH LABELS
        # --------------------------------------------------
        #
        plt.title(
            "Bull Call Spread Payoff Diagram"
        )

        plt.xlabel(
            "Stock Price at Expiration"
        )

        plt.ylabel(
            "Profit / Loss"
        )

        plt.text(
            lower_K,
            min(profits) * 0.5,
            f"Buy Strike = {lower_K:.0f}",
            rotation=90
        )

        plt.text(
            higher_K,
            max(profits) * 0.4,
            f"Sell Strike = {higher_K:.0f}",
            rotation=90
        )

        plt.text(
            break_even,
            max(profits) * 0.1,
            f"Break-even = {break_even:.2f}",
            rotation=90
        )

        plt.axvline(lower_K, linestyle="--")
        plt.axvline(higher_K, linestyle=":")
        plt.axvline(break_even, linestyle="-.")

        # Display graph
        plt.show()

def main():
    print("Options Dashboard - Black-Scholes Pricer")

    # -----------------------------
    # USER INPUTS
    # -----------------------------
    S = float(input("Enter current stock price: "))
    K = float(input("Enter strike price: "))
    T = float(input("Enter time to expiration in years: "))
    r = float(input("Enter risk-free rate as decimal, example 0.05: "))
    sigma = float(input("Enter volatility as decimal, example 0.25: "))
    option_type = input("Call or put? ").lower()

    price = black_scholes_price(S, K, T, r, sigma, option_type)

    print()
    print(f"The {option_type} option price is: ${price:.2f}")

    delta, gamma, theta, vega, rho = calculate_greeks(S, K, T, r, sigma, option_type)

    print()
    print("Option Greeks:")
    print(f"Delta: {delta:.4f}")
    print(f"Gamma: {gamma:.4f}")
    print(f"Theta: {theta:.4f} per day")
    print(f"Vega: {vega:.4f} per 1% volatility change")
    print(f"Rho: {rho:.4f} per 1% interest rate change")

    print()
    use_iv = input("Do you want to calculate implied volatility? yes/no: ").lower()

    if use_iv == "yes":
        market_price = float(input("Enter the market option price: "))

        iv = implied_volatility(market_price, S, K, T, r, option_type)

        print()
        print(f"Implied Volatility: {iv:.2%}")

    print()
    use_graph = input("Do you want to see a payoff graph? yes/no: ").lower()

    if use_graph == "yes":
        plot_long_option_payoff(K, price, option_type)

    print()
    use_spread = input("Do you want to plot a bull call spread? yes/no: ").lower()

    if use_spread == "yes":
        lower_K = float(input("Enter lower strike price: "))
        higher_K = float(input("Enter higher strike price: "))
        lower_premium = float(input("Enter premium paid for lower-strike call: "))
        higher_premium = float(input("Enter premium received for higher-strike call: "))

        plot_bull_call_spread_payoff(
            lower_K,
            higher_K,
            lower_premium,
            higher_premium
        )


if __name__ == "__main__":
    main()
