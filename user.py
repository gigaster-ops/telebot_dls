class user():
    def __init__(self, user_id):
        self.user_id = user_id
        self.user_process = []

    def start_process(self, func, *arg):
        self.user_process.append(func)
        func(arg)
