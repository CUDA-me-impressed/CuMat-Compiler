class DimError(ValueError):
    pass


class LiteralNode:
    def __init__(self, value):
        self.value = value
        self.dimension = [1]

    def dimcheck(self):
        pass

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f"LiteralNode({self.value})"


class MatrixNode:
    def __init__(self, data, seps):
        self.data = [*data]
        self.seps = [*seps]
        self.dimension = None

    def dimcheck(self):
        if self.dimension:
            return
        for i in self.data:
            i.dimcheck()

        apparent_dim = []
        size = []

        for node, sep in zip(self.data, self.seps):
            if len(node.dimension) > sep:
                raise DimError()
            while len(size) < len(node.dimension) or len(size) < sep:
                size.append(0)
            for i, v in enumerate(node.dimension):
                size[i] += v
            for i in range(sep - 1):
                if i + 1 > len(apparent_dim):
                    apparent_dim.append(size[i])
                if size[i] != apparent_dim[i]:
                    raise DimError()
                size[i] = 0
                if i + 1 >= len(node.dimension):
                    size[i + 1] += 1

        node = self.data[-1]
        sep = len(size)

        if len(node.dimension) > sep:
            raise DimError()
        while len(size) < len(node.dimension) or len(size) < sep:
            size.append(0)
        for i, v in enumerate(node.dimension):
            size[i] += v
        for i in range(sep - 1):
            if i + 1 > len(apparent_dim):
                apparent_dim.append(size[i])
            if size[i] != apparent_dim[i]:
                raise DimError()
            size[i] = 0
            if i + 1 >= len(node.dimension):
                size[i + 1] += 1

        apparent_dim.append(size[-1])

        self.dimension = apparent_dim

    def __str__(self):
        str_ = "["
        for d, s in zip(self.data, self.seps):
            if s == 1:
                separator = ","
            else:
                separator = "\\" * (s - 1)
                separator = " " + separator
            str_ = f"{str_}{d}{separator} "
        str_ += f"{self.data[-1]}]"
        return str_

    def __repr__(self):
        return f"Matrix({self.data}, {self.seps})"


def test():
    values = [LiteralNode(i) for i in range(1, 5)]
    sep1 = [1, 1, 1]
    sep2 = [2, 2, 2]
    sep3 = [1, 2, 1]

    mat41 = MatrixNode(values, sep1)
    # print("41\t\t\t|", mat41, mat41.data, mat41.seps)
    mat14 = MatrixNode(values, sep2)
    # print("14\t\t\t|", mat14, mat14.data, mat14.seps)
    mat22 = MatrixNode(values, sep3)
    # print("22\t\t\t|", mat22, mat22.data, mat22.seps)
    mat14.dimcheck()
    print("14dim\t\t|", mat14.dimension)
    mat41.dimcheck()
    print("41dim\t\t|", mat41.dimension)
    mat22.dimcheck()
    print("22dim\t\t|", mat22.dimension)
    cmat = MatrixNode([mat22, mat22, mat22], [2, 2])
    print(cmat)
    cmat.dimcheck()
    print("c dim\t\t|", cmat.dimension)
    cmat = MatrixNode([mat41, mat41], [1])
    cmat.dimcheck()
    print("c dim\t\t|", cmat.dimension)


if __name__ == '__main__':
    test()
