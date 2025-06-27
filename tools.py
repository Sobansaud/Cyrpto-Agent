from agents import function_tool
import requests

@function_tool
def get_crypto(coin: str = "bitcoin", currency: str = "usd") -> str:
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin.lower()}&vs_currencies={currency.lower()}"
    response = requests.get(url)
    data = response.json()

    if coin.lower() in data:
        price = data[coin.lower()][currency.lower()]
        return f"{coin.capitalize()} ki current price in {currency.upper()} is {price}"
    else:
        return f"{coin.capitalize()} is not available in {currency.upper()}"
