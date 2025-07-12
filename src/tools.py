import requests
import time
import os

def get_currency_rates(currencies, base="BRL"):
    rates = {}
    access_key = os.getenv('EXCHANGE_KEY')
    # URL base
    url = 'https://api.exchangerate.host/convert'

    print(currencies)

    for dest in currencies:
        currency = dest["currency"]

        # Parâmetros da query
        params = {
            'access_key': access_key,
            'from': currency,
            'to': base,
            'amount': 1
        }

        if currency not in rates and currency != params['to']:
            response = requests.get(url, params=params)
            data = response.json()
            rates[currency] = data['result']

        time.sleep(2)
    return rates

def currency_tool(state):
    print("\n---------- Currency Tool Start ----------")
    destinations = state.get("currencies", [])
    rates = get_currency_rates(destinations)
    print(rates)
    print("---------- Currency Tool End ----------\n")

    return {
        **state,
        "currency_rates": rates
    }