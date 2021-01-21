
def names_generator():
    i = 0
    while True:
        i += 1
        yield i


names = {}
counter = names_generator()


class BooleanOperator:
    def __init__(self, name: int):
        self.name = name


class And(BooleanOperator):
    def __init__(self, left: BooleanOperator, right: BooleanOperator):
        super().__init__(next(counter))
        self.left = left
        self.right = right
        print(f"{self.name}: {self.left.name} and {self.right.name}")


class Or(BooleanOperator):
    def __init__(self, left: BooleanOperator, right: BooleanOperator):
        super().__init__(next(counter))
        self.left = left
        self.right = right
        print(f"{self.name}: {self.left.name} or {self.right.name}")


class Not(BooleanOperator):
    def __init__(self, param: BooleanOperator):
        if type(param) == Atomic:
            super().__init__(-param.name)
        else:
            super().__init__(next(counter))
        self.param = param
        print(f"{self.name}: not {self.param.name}")


class Imp(BooleanOperator):
    def __init__(self, left: BooleanOperator, right: BooleanOperator):
        super().__init__(next(counter))
        self.left = left
        self.right = right
        print(f"{self.name}: {self.left.name} -> {self.right.name}")


class Equiv(BooleanOperator):
    def __init__(self, left: BooleanOperator, right: BooleanOperator):
        super().__init__(next(counter))
        self.left = left
        self.right = right
        print(f"{self.name}: {self.left.name} <-> {self.right.name}")


class Atomic(BooleanOperator):
    def __init__(self, name: str):
        if name not in names.keys():
            names[name] = next(counter)
            super().__init__(names[name])
        else:
            super().__init__(names[name])
        self.val = name
        print(f"{self.name}: {self.val}")
