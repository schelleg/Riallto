import numpy as np

class MTIterator2D:
    def __init__(self, origa, blocksize):
        self.a = np.copy(origa)
        self.rows = self.a.shape[0]
        self.cols = self.a.shape[1]
        self.blocksize = blocksize  
        self.shape = self.a.shape

    def __getitem__(self,s):
        '''Simplified 2D indexing for blocks meant to enforce simple striding for ease of sequence building.  E.g. [0,0] or [0:2,0:2]'''
        assert isinstance(s, tuple) and len(s) == 2, "Only 2D indexing is supported.  E.g. [0,0] or [0:2,0:2]"


        #TODO - add support for slices as use cases arise.  Keep it simple for now
        i, j = s
        assert isinstance(i, int) and isinstance(j, int), "Only 2D integer indexing is supported now.  E.g. [0,0]"
        return self.a[i*self.blocksize:i*self.blocksize+self.blocksize, j*self.blocksize:j*self.blocksize+self.blocksize]


    def __setitem__(self,s,val):
        assert isinstance(s, tuple) and len(s) == 2, "Only 2D indexing is supported.  E.g. [0,0] or [0:2,0:2]"
        #TODO - add support for slices as use cases arise.  Keep it simple for now
        i, j = s
        assert isinstance(i, int) and isinstance(j, int), "Only 2D integer indexing is supported now.  E.g. [0,0]"
        self.a[i*self.blocksize:i*self.blocksize+self.blocksize, j*self.blocksize:j*self.blocksize+self.blocksize] = val


class BlockMatMul:
    def __init__(self):
        pass
    def __call__(self,a,b,c):
        return np.dot(a,b) + c

class BlockMatMulApp:  
    def __init__(self,blocksize):
        self.bmm = BlockMatMul()
        self.blocksize = blocksize
        # super.__init__()
    
    def __call__(self, mema, memb, memc):
        return self.callgraph(mema, memb, memc)

    def callgraph(self, mema, memb, memc):
        a, b, c = [MTIterator2D(m, self.blocksize) for m in [mema, memb, memc]]
        
        # Block Matrix Multiply with write to C once per block

        with open("dutlog.txt", "w") as f:
            for i in range(a.shape[0] // self.blocksize):
                for j in range(b.shape[1] // self.blocksize):
                    for k in range(a.shape[1] // self.blocksize):

                        f.write("\n\nC0 " + str(c[i, j]) + "\nA0 " + str(a[i, k]) + "\n B0" + str(b[k, j]) + "\n")

                        c[i,j] = self.bmm(a[i, k], b[k, j], c[i, j])    


                        f.write(f"C {i} {j} {k}" + "\n")
                        f.write("C1 " + str(c[i, j]) + "\nA1 " + str(a[i, k]) + "\n B1" + str(b[k, j]) + "\n")
                        f.write("C2 " + str(c.a) + "\n")



                # memc[i,j] = c[i,j]
        
        return c.a


# mema = np.random.randint(0, 10, size=(6, 6))
# memb = np.random.randint(0, 10, size=(6, 6))

mema = np.arange(16).reshape((8, 2))
memb = np.arange(12).reshape((2, 6))

memc = np.zeros((mema.shape[0], memb.shape[1]), dtype=np.int32)
block_size = 2






def block_matrix_multiply(A, B, block_size):

    with open("goldenlog.txt", "w") as f:

        # Check if the dimensions are compatible for block matrix multiplication
        if A.shape[1] != B.shape[0]:
            raise ValueError("The number of columns in A must be equal to the number of rows in B.")

        # Initialize the resulting matrix C with zeros
        C = np.zeros((A.shape[0], B.shape[1]), dtype=np.int32)

        # Iterate over blocks
        for i in range(0, A.shape[0], block_size):
            for j in range(0, B.shape[1], block_size):
                for k in range(0, A.shape[1], block_size):                    
                    # Calculate block multiplication

                    f.write("\n\nC0 " + str(C[i:i+block_size, j:j+block_size]) + "\nA0 " + str(A[i:i+block_size, k:k+block_size]) + "\n B0" + str(B[k:k+block_size, j:j+block_size]) + "\n")

                    C[i:i+block_size, j:j+block_size] += np.dot(A[i:i+block_size, k:k+block_size],
                                                                B[k:k+block_size, j:j+block_size])


                    f.write(f"C {i // block_size} {j // block_size} {k // block_size}" + "\n")
                    f.write("C1 " + str(C[i:i+block_size, j:j+block_size]) + "\nA1 " + str(A[i:i+block_size, k:k+block_size]) + "\n B1" + str(B[k:k+block_size, j:j+block_size]) + "\n")
                    f.write("C2 " + str(C) + "\n")


    return C


riallto_bmm = BlockMatMulApp(block_size)
Criallto = riallto_bmm(mema,memb,memc)

# print("\n\nCriallto")
# print(Criallto)

# print("A*B")
# print(mema @ memb)

print(np.array_equal(mema @ memb, riallto_bmm(mema,memb,memc)))
# print("\n\n")
# Example usage:
# Create two matrices A and B
block_size = 2

# Perform block matrix multiplication
C = block_matrix_multiply(mema, memb, block_size)

# Display the results
# print("Matrix A:")
# print(mema)
# print("\nMatrix B:")
# print(memb)
print(np.array_equal(mema @ memb, C))