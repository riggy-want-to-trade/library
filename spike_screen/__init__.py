"""A-share spike & drop screening service.

Screens stock_data/SS and stock_data/SZ daily OHLCV files for stocks that
spiked (>= threshold within a trading-week window, or one of three
"3-day 10%" patterns) and then fell after their peak. Exposed as a
FastAPI REST service; see main.py for endpoints and README.md for usage.
"""

__version__ = "1.0.0"
