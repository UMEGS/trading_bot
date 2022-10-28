
class BinanceBot():

    def __init__(self, api_key, api_secret,is_testnet=False):
        self.client = Client(api_key, api_secret, testnet=is_testnet)


        # self.balances = self.client.get_account()
        #
        # self.balances = self.balances['balances']
        #
        # self.balances = {x['asset']: x for x in self.balances}
        #
        # self.balances = {x: self.balances[x] for x in self.balances if float(self.balances[x]['free']) > 0}
        #
        # self.balances = {x: self.balances[x] for x in self.balances if x not in ['BTC', 'USDT']}
        #
        # self.balances = {x: self.balances[x] for x in self.balances if float(self.balances[x]['free']) > 0.0001}
        #
        # self.balances = {x: self.balances[x] for x in self.balances if x not in ['BTC', 'USDT']}
        #
        # self.balances = {x: self.balances[x] for x in self.balances if float(self.balances[x]['free']) > 0.0001}
        #
        # self.balances = {x: self.balances[x] for x in self.balances if x not in ['BTC', 'USDT']}
        #
        # self.balances = {x: self.balances[x] for x in self.balances if float(self.balances[x]['free']) > 0.0001}
        #
        # self.balances = {x: self.balances[x] for x in self.balances if x not in ['BTC', 'USDT']}
        #
        # self.balances = {x: self.balances[x] for x in self.balances if float(self.balances[x]['free']) > 0.0001}
        #
        # self.balances = {x: self.balances[x] for x in self.balances if x not in ['BTC', 'USDT']}
        #
        # self.balances = {x: self.balances[x] for x in self.balances if float(self.balances[x]['free']) > 0.0001}
        #
        # self.balances = {x: self.balances[x] for x in self.balances if x not in ['BTC', 'USDT']}
        #
        # self.balances = {x: self.balances[x] for x in self.balances if float(self.balances[x]['free']) > 0.0001}
        #