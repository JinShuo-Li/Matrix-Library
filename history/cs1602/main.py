# Test code for IEEE course final project
# Jiachen Zou, Jinshuo Li, 2025

import minimatrix as mm

#Test your code here!
def main():
    #1: Test the 3*3 matrix
    try:
        matr = [[1,2,3],[6,5,4],[7,8,9]]
        mat = mm.Matrix(data=matr)
    except Exception as e:
        print(f"Failed !: {e}")
        print("The calculation can't continue.")
        quit()
    print(mat.shape())
    a = mat.reshape((1,9))
    print(a.data)
    b = mat.dot(mat)
    print(b.data)
    c = mat.T()
    print(c.data)
    d1 = mat.sum(axis=0)
    d2 = mat.sum(axis=1)
    print(d1.data)
    print(d2.data)
    print(mat.sum())
    print(mat[2,2])
    mat[2,2] = 0
    print(mat.data)
    mat[2,2] = 9
    print(mat.data)
    f = mat**2
    print(f.data)
    g = mat+mat
    print(g.data)
    h = mat-mat
    print(h.data)
    j = mat*mat
    print(j.data)
    print(len(mat))
    print(str(mat))
    print(mat.det())
    try:
        result = mat.inverse()
        print(result.data)
        print(mat.rank())
    except Exception as e:
        print(f"Failed! : {e}")
        pass

    #2: Test arrange
    m24 = mm.arange(0,24,1)
    print(m24.data)
    print((m24.reshape((3,8))).data)
    print((m24.reshape((24,1))).data)
    print((m24.reshape((4,6))).data)

    #3: Test zeros
    print((mm.zeros((3,3))).data)
    print((mm.zeros_like(m24)).data)


    #4: Test ones
    print((mm.ones((3,3))).data)
    print((mm.ones_like(m24)).data)

    #5: Test randoms
    print((mm.nrandom((3,3))).data)
    print((mm.nrandom_like(m24)).data)

    #6: Another problem
    m, n = 1000, 100
    X = mm.nrandom((m, n))
    w = mm.nrandom((n, 1))
    e = mm.nrandom((m, 1))
    Xw = X.dot(w)
    Y_data = []
    for i in range(m):
        row_sum = Xw.data[i][0] + e.data[i][0]
        Y_data.append([row_sum])
    Y = mm.Matrix(data=Y_data)
    X_T_data = X.T()
    X_T = mm.Matrix(data=X_T_data.data)
    X_T_X_data = X_T.dot(X)
    X_T_X = mm.Matrix(data=X_T_X_data.data)

    try:
        X_T_X_inv = X_T_X.inverse()
    except Exception as e:
        print(f"Failed! : {e}")
        return

    X_T_Y_data = X_T.dot(Y)
    X_T_Y = mm.Matrix(data=X_T_Y_data.data)
    
    w_hat_data = X_T_X_inv.dot(X_T_Y)
    w_hat = mm.Matrix(data=w_hat_data.data)
    w_sub = w - w_hat
    for i in range(100):
        ele = abs(w_sub.data[i][0]) / w.data[i][0]
        print(ele)


if __name__ == "__main__":
    main()